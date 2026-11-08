"""Resumable street-view feature extraction for the Macau road-image project.

Run from a PyCharm Jupyter notebook or the project terminal. One journal row is
written per unique image SHA-256; rerunning resumes without repeating inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "street_view_features_v1"
SEG_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
WEIGHTS = ROOT / "outputs" / "street_view_feature_pilot_v1" / "weights" / "yolo11s.pt"
PILOT_IDS = {5241, 808, 3813, 1750, 3410, 2057, 2567, 4372, 4768, 6654, 1888, 4929}
PARAMS = {
    "preprocess_version": "macau_streetview_full_v1",
    "bottom_exclusion_fraction": 0.08,
    "tile_width_px": 1024,
    "tile_step_px": 768,
    "seg_input_height_px": 768,
    "seg_input_width_px": 1024,
    "det_imgsz": 1024,
    "det_conf": 0.25,
    "det_iou_within_tile": 0.50,
    "det_iou_cross_tile": 0.50,
    "pedestrian_min_semantic_pixels": 100,
    "pedestrian_person_ratio": 0.70,
    "rider_max_person_ratio": 0.30,
    "model_seg": SEG_NAME,
    "model_det": "yolo11s.pt",
}
MOTOR_IDS = {2, 3, 5, 7}
DETECTION_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FEATURES = [
    "vegetation_share", "building_share", "sky_share", "motor_vehicle_count",
    "person_count", "pedestrian_count_est", "pedestrian_uncertain_count",
    "rider_count_est", "car_count", "motorcycle_count", "bus_count", "truck_count",
]


def load_inputs(only_ids: set[int] | None = None) -> pd.DataFrame:
    tables = ROOT / "outputs" / "structured_data_v1" / "tables"
    images = pd.read_csv(tables / "images.csv", encoding="utf-8-sig")
    links = pd.read_csv(ROOT / "outputs" / "image_road_linkage_v1" / "image_road_association.csv", encoding="utf-8-sig")
    roads = pd.read_csv(tables / "roads.csv", encoding="utf-8-sig")
    points = pd.read_csv(ROOT / "outputs" / "notebook_analysis" / "采样点区域归属.csv", encoding="utf-8-sig")
    merged = images.merge(
        links[["image_id", "road_id", "fclass", "road_name", "longitude", "latitude", "visual_review_status", "file_match_status"]],
        on="image_id", validate="one_to_one"
    ).merge(
        roads[["road_id", "bridge", "tunnel"]], on="road_id", validate="many_to_one"
    ).merge(
        points[["sample_id", "point_region", "near_boundary_or_outside"]], on="sample_id", validate="one_to_one"
    )
    assert len(merged) == len(images), (len(merged), len(images))
    assert merged["image_id"].is_unique and merged["sha256"].notna().all()
    if only_ids is not None:
        merged = merged[merged["source_objectid"].isin(only_ids)].copy()
    return merged.sort_values("source_objectid").reset_index(drop=True)


def load_journal(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                records[rec["sha256"]] = rec
            except json.JSONDecodeError:
                print(f"Skipping incomplete journal line {line_no}", file=sys.stderr)
    return records


def tile_starts(width: int) -> list[int]:
    size = PARAMS["tile_width_px"]
    step = PARAMS["tile_step_px"]
    if width <= size:
        return [0]
    starts = list(range(0, width - size + 1, step))
    if starts[-1] != width - size:
        starts.append(width - size)
    return sorted(set(starts))


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter) if inter else 0.0


def run_inference(row, context, save_preview: bool, out: Path) -> dict:
    torch = context["torch"]
    F = context["F"]
    nms = context["nms"]
    processor = context["processor"]
    seg_model = context["seg_model"]
    det_model = context["det_model"]
    label = context["label"]
    device = context["device"]
    precision = context["precision"]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    image_path = ROOT / "outputs" / "structured_data_v1" / row.relative_path
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    height, width = rgb.shape[:2]
    valid_h = round(height * (1 - PARAMS["bottom_exclusion_fraction"]))
    rgb = rgb[:valid_h]
    read_sec = time.perf_counter() - t0
    starts = tile_starts(width)
    semantic = np.empty((valid_h, width), dtype=np.uint8)
    all_boxes = []
    seg_sec = det_sec = 0.0
    disagreements = []
    sky_building_conflicts = []
    prev_mask = None
    prev_start = None

    for i, x0 in enumerate(starts):
        x1 = min(x0 + PARAMS["tile_width_px"], width)
        tile = rgb[:, x0:x1]
        ts = time.perf_counter()
        inputs = processor(
            images=Image.fromarray(tile), return_tensors="pt",
            size={"height": PARAMS["seg_input_height_px"], "width": PARAMS["seg_input_width_px"]},
        )
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
        with torch.inference_mode(), autocast:
            logits = seg_model(**{k: v.to(device) for k, v in inputs.items()}).logits
            up = F.interpolate(logits, size=tile.shape[:2], mode="bilinear", align_corners=False)
            tile_mask = up.argmax(1)[0].to("cpu", dtype=torch.uint8).numpy()
        sync()
        seg_sec += time.perf_counter() - ts
        del inputs, logits, up

        if prev_mask is not None:
            overlap = (prev_start + prev_mask.shape[1]) - x0
            if overlap > 0:
                left = prev_mask[:, -overlap:]
                right = tile_mask[:, :overlap]
                disagreements.append(float(np.mean(left != right)))
                conflict = ((left == label["sky"]) & (right == label["building"])) | ((left == label["building"]) & (right == label["sky"]))
                sky_building_conflicts.append(float(np.mean(conflict)))
        own_left = 0 if i == 0 else (starts[i - 1] + PARAMS["tile_width_px"] + x0) // 2
        own_right = width if i == len(starts) - 1 else (x1 + starts[i + 1]) // 2
        semantic[:, own_left:own_right] = tile_mask[:, own_left - x0:own_right - x0]
        prev_mask, prev_start = tile_mask, x0

        td = time.perf_counter()
        det_options = {"half": True} if precision == "fp16" else {}
        pred = det_model.predict(
            tile, device=0 if device == "cuda" else "cpu", imgsz=PARAMS["det_imgsz"],
            conf=PARAMS["det_conf"], iou=PARAMS["det_iou_within_tile"],
            classes=list(DETECTION_NAMES), verbose=False, **det_options,
        )[0]
        sync()
        det_sec += time.perf_counter() - td
        for xyxy, cls, conf in zip(
            pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.cpu().numpy(), pred.boxes.conf.cpu().numpy()
        ):
            xa, ya, xb, yb = xyxy.tolist()
            all_boxes.append({
                "x1": xa + x0, "y1": ya, "x2": xb + x0, "y2": yb,
                "class_id": int(cls), "confidence": float(conf), "tile_x": x0,
            })

    kept = []
    for cls in DETECTION_NAMES:
        group = [b for b in all_boxes if b["class_id"] == cls]
        if not group:
            continue
        coords = torch.tensor([[b[k] for k in ("x1", "y1", "x2", "y2")] for b in group], dtype=torch.float32)
        scores = torch.tensor([b["confidence"] for b in group])
        kept.extend(group[j] for j in nms(coords, scores, PARAMS["det_iou_cross_tile"]).tolist())

    pedestrian = rider = uncertain = 0
    for box in kept:
        if box["class_id"] != 0:
            continue
        xa, ya = max(0, int(box["x1"])), max(0, int(box["y1"]))
        xb, yb = min(width, math.ceil(box["x2"])), min(valid_h, math.ceil(box["y2"]))
        sub = semantic[ya:yb, xa:xb]
        p = int(np.sum(sub == label["person"]))
        r = int(np.sum(sub == label["rider"]))
        if p + r < PARAMS["pedestrian_min_semantic_pixels"]:
            uncertain += 1
        elif p / (p + r) >= PARAMS["pedestrian_person_ratio"]:
            pedestrian += 1
        elif p / (p + r) <= PARAMS["rider_max_person_ratio"]:
            rider += 1
        else:
            uncertain += 1

    counts = Counter(b["class_id"] for b in kept)
    motor_boxes = [b for b in kept if b["class_id"] in MOTOR_IDS]
    possible_overlap = sum(
        1 for i, a in enumerate(motor_boxes) for b in motor_boxes[i+1:]
        if a["tile_x"] != b["tile_x"] and iou(a, b) >= 0.25
    )
    features = {
        "image_width_px": width, "image_height_px": height,
        "valid_pixel_count": width * valid_h, "tile_count": len(starts),
        "vegetation_share": float(np.mean(semantic == label["vegetation"])),
        "building_share": float(np.mean(semantic == label["building"])),
        "sky_share": float(np.mean(semantic == label["sky"])),
        "motor_vehicle_count": int(sum(counts[c] for c in MOTOR_IDS)),
        "person_count": int(counts[0]), "pedestrian_count_est": pedestrian,
        "pedestrian_uncertain_count": uncertain, "rider_count_est": rider,
        "car_count": int(counts[2]), "motorcycle_count": int(counts[3]),
        "bus_count": int(counts[5]), "truck_count": int(counts[7]),
        "raw_detection_count": len(all_boxes), "kept_detection_count": len(kept),
        "cross_tile_vehicle_overlap_pairs": possible_overlap,
        "tile_disagreement_max": max(disagreements, default=0.0),
        "tile_disagreement_mean": float(np.mean(disagreements)) if disagreements else 0.0,
        "sky_building_conflict_max": max(sky_building_conflicts, default=0.0),
        "read_sec": read_sec, "segmentation_sec": seg_sec,
        "detection_sec": det_sec, "inference_sec": seg_sec + det_sec,
    }
    flags = []
    if width != 4096 or height != 1536:
        flags.append("unexpected_image_size")
    if features["tile_disagreement_max"] > 0.25:
        flags.append("tile_disagreement_high")
    if features["sky_building_conflict_max"] > 0.02:
        flags.append("sky_building_seam_conflict")
    if str(row.tunnel).upper() in ("T", "TRUE", "1") and features["vegetation_share"] > 0.02:
        flags.append("tunnel_vegetation_review")
    if features["sky_share"] < 0.005 and features["vegetation_share"] > 0.05:
        flags.append("zero_sky_high_vegetation_review")
    if str(row.bridge).upper() in ("T", "TRUE", "1") and features["building_share"] > 0.15:
        flags.append("bridge_building_review")
    if possible_overlap:
        flags.append("vehicle_overlap_review")
    if uncertain >= 3:
        flags.append("pedestrian_uncertain_review")
    if sum(features[k] for k in ("vegetation_share", "building_share", "sky_share")) > 1.00001:
        flags.append("pixel_share_inconsistent")
    features["model_flags"] = flags

    if save_preview:
        preview_dir = out / "qa_previews"
        preview_dir.mkdir(exist_ok=True)
        vis = rgb.copy()
        colors = {
            label["vegetation"]: (46, 170, 75),
            label["building"]: (240, 128, 87),
            label["sky"]: (64, 162, 230),
        }
        for cid, color in colors.items():
            loc = semantic == cid
            vis[loc] = (vis[loc].astype(np.uint16) * 55 // 100 + np.array(color, dtype=np.uint16) * 45 // 100).astype(np.uint8)
        pic = Image.fromarray(vis)
        draw = ImageDraw.Draw(pic)
        for b in kept:
            color = "#e8218a" if b["class_id"] == 0 else "#f5d427"
            draw.rectangle((b["x1"], b["y1"], b["x2"], b["y2"]), outline=color, width=6)
        pic.thumbnail((1600, 600))
        pic.save(preview_dir / f"{row.image_id}.jpg", quality=85)
    return features


def build_context(precision: str):
    import torch
    import torch.nn.functional as F
    from torchvision.ops import nms
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from ultralytics import YOLO

    assert WEIGHTS.exists(), WEIGHTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision == "fp16" and device != "cuda":
        raise RuntimeError("fp16 mode requires CUDA")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(SEG_NAME)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_NAME).to(device).eval()
    det_model = YOLO(str(WEIGHTS))
    label = {v: k for k, v in seg_model.config.id2label.items()}
    assert all(k in label for k in ("vegetation", "building", "sky", "person", "rider"))
    load_sec = time.perf_counter() - t0
    # Warm-up is deliberately outside per-image timing.
    warm = np.zeros((768, 1024, 3), dtype=np.uint8)
    inputs = processor(images=Image.fromarray(warm), return_tensors="pt", size={"height": 768, "width": 1024})
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
    with torch.inference_mode(), autocast:
        _ = seg_model(**{k: v.to(device) for k, v in inputs.items()}).logits
    det_options = {"half": True} if precision == "fp16" else {}
    _ = det_model.predict(warm, device=0 if device == "cuda" else "cpu", imgsz=1024, verbose=False, **det_options)
    if device == "cuda":
        torch.cuda.synchronize()
    return dict(
        torch=torch, F=F, nms=nms, processor=processor, seg_model=seg_model,
        det_model=det_model, label=label, device=device, precision=precision, load_sec=load_sec,
        seg_commit=getattr(seg_model.config, "_commit_hash", None),
    )


def verify_features(df: pd.DataFrame) -> None:
    assert df["image_id"].is_unique
    ok = df[df["process_status"] == "success"]
    for col in ("vegetation_share", "building_share", "sky_share"):
        assert ok[col].between(0, 1).all(), col
    assert (ok[["vegetation_share", "building_share", "sky_share"]].sum(axis=1) <= 1.00001).all()
    for col in ("motor_vehicle_count", "person_count", "pedestrian_count_est", "pedestrian_uncertain_count"):
        assert (ok[col] >= 0).all(), col
    assert (ok["pedestrian_count_est"] + ok["pedestrian_uncertain_count"] <= ok["person_count"]).all()


def finalize(rows: pd.DataFrame, journal: dict[str, dict], out: Path, run_info: dict) -> None:
    road_counts = rows.groupby("sha256")["road_id"].nunique().to_dict()
    first_image = rows.groupby("sha256")["image_id"].first().to_dict()
    records = []
    for row in rows.itertuples(index=False):
        entry = journal.get(row.sha256)
        if entry is None:
            raise RuntimeError(f"No journal result for {row.image_id}")
        metrics = entry.get("features", {})
        # Recalculate review flags from stored raw metrics. This lets quality
        # thresholds be calibrated without changing the model predictions.
        flags = [f for f in metrics.get("model_flags", []) if f in ("unexpected_image_size", "pixel_share_inconsistent")]
        if metrics.get("tile_disagreement_max", 0) > 0.30:
            flags.append("tile_disagreement_high")
        if metrics.get("sky_building_conflict_max", 0) > 0.15:
            flags.append("sky_building_seam_conflict")
        if metrics.get("cross_tile_vehicle_overlap_pairs", 0) >= 1:
            flags.append("vehicle_overlap_review")
        if metrics.get("pedestrian_uncertain_count", 0) >= 3:
            flags.append("pedestrian_uncertain_review")
        if str(row.tunnel).upper() in ("T", "TRUE", "1") and metrics.get("vegetation_share", 0) > 0.02:
            flags.append("tunnel_vegetation_review")
        if metrics.get("sky_share", 1) < 0.005 and metrics.get("vegetation_share", 0) > 0.05:
            flags.append("zero_sky_high_vegetation_review")
        if str(row.bridge).upper() in ("T", "TRUE", "1") and metrics.get("building_share", 0) > 0.15:
            flags.append("bridge_building_review")
        if road_counts[row.sha256] > 1:
            flags.append("cross_road_exact_duplicate")
        if row.visual_review_status == "priority_manual_review":
            flags.append("source_priority_manual_review")
        if bool(row.near_boundary_or_outside):
            flags.append("region_boundary_review")
        rec = {
            "image_id": row.image_id, "sample_id": row.sample_id, "road_id": row.road_id,
            "source_objectid": int(row.source_objectid), "source_sha256": row.sha256,
            "relative_path": row.relative_path, "duplicate_of_image_id": row.duplicate_of_image_id,
            "cache_hit": row.image_id != first_image[row.sha256],
            "fclass": row.fclass, "road_name": row.road_name,
            "longitude": row.longitude, "latitude": row.latitude,
            "point_region": row.point_region,
            "near_boundary_or_outside": bool(row.near_boundary_or_outside),
            "bridge": row.bridge, "tunnel": row.tunnel,
            "file_match_status": row.file_match_status,
            "association_review_status": row.visual_review_status,
            "process_status": entry["status"],
            **{k: entry.get("features", {}).get(k, np.nan) for k in FEATURES},
            "image_width_px": entry.get("features", {}).get("image_width_px", np.nan),
            "image_height_px": entry.get("features", {}).get("image_height_px", np.nan),
            "valid_pixel_count": entry.get("features", {}).get("valid_pixel_count", np.nan),
            "tile_count": entry.get("features", {}).get("tile_count", np.nan),
            "raw_detection_count": entry.get("features", {}).get("raw_detection_count", np.nan),
            "kept_detection_count": entry.get("features", {}).get("kept_detection_count", np.nan),
            "cross_tile_vehicle_overlap_pairs": entry.get("features", {}).get("cross_tile_vehicle_overlap_pairs", np.nan),
            "tile_disagreement_max": entry.get("features", {}).get("tile_disagreement_max", np.nan),
            "tile_disagreement_mean": entry.get("features", {}).get("tile_disagreement_mean", np.nan),
            "sky_building_conflict_max": entry.get("features", {}).get("sky_building_conflict_max", np.nan),
            "read_sec": 0.0 if row.image_id != first_image[row.sha256] else entry.get("features", {}).get("read_sec", np.nan),
            "segmentation_sec": 0.0 if row.image_id != first_image[row.sha256] else entry.get("features", {}).get("segmentation_sec", np.nan),
            "detection_sec": 0.0 if row.image_id != first_image[row.sha256] else entry.get("features", {}).get("detection_sec", np.nan),
            "inference_sec": 0.0 if row.image_id != first_image[row.sha256] else entry.get("features", {}).get("inference_sec", np.nan),
            "error_type": entry.get("error_type", ""),
            "error_message": entry.get("error_message", ""),
            "quality_flags": ";".join(sorted(set(flags))),
            "quality_status": "failed" if entry["status"] != "success" else ("needs_review" if flags else "auto_pass"),
            "preprocess_version": PARAMS["preprocess_version"],
            "quality_rule_version": "v1.1",
        }
        records.append(rec)
    df = pd.DataFrame(records)
    verify_features(df)

    # Statistical tails are prompts for manual review, never automatic corrections.
    unique = df.drop_duplicates("source_sha256")
    tail_cols = ["vegetation_share", "building_share", "sky_share", "motor_vehicle_count", "person_count"]
    thresholds = {}
    for grade, subset in unique[unique.process_status == "success"].groupby("fclass"):
        thresholds[grade] = {col: float(subset[col].quantile(0.995)) for col in tail_cols}
    tails = []
    for row in df.itertuples(index=False):
        if row.process_status != "success":
            tails.append("")
            continue
        hit = [col for col in tail_cols if getattr(row, col) > thresholds[row.fclass][col]]
        tails.append(";".join(hit))
    df["statistical_tail_fields"] = tails
    review = df[(df.quality_status != "auto_pass") | (df.statistical_tail_fields != "")].copy()
    review["review_priority"] = np.where(df.loc[review.index, "process_status"] != "success", "high",
        np.where(review.quality_flags.str.contains("tunnel_vegetation|zero_sky_high_vegetation|sky_building_seam|source_priority", na=False)
                 | (review.cross_tile_vehicle_overlap_pairs >= 3), "high", "normal"))
    df.to_csv(out / "street_view_features_full.csv", index=False, encoding="utf-8-sig")
    review.to_csv(out / "anomaly_review.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_image_rows": len(df), "n_unique_sha256": int(df.source_sha256.nunique()),
        "n_success_images": int((df.process_status == "success").sum()),
        "n_failed_images": int((df.process_status != "success").sum()),
        "n_auto_pass": int((df.quality_status == "auto_pass").sum()),
        "n_needs_review": int((df.quality_status == "needs_review").sum()),
        "n_review_rows": len(review), "n_statistical_tail_rows": int((df.statistical_tail_fields != "").sum()),
        "n_cross_road_exact_duplicate": int(df.quality_flags.str.contains("cross_road_exact_duplicate").sum()),
        "median_unique_inference_sec": float(unique.loc[unique.process_status == "success", "inference_sec"].median()),
        "mean_unique_inference_sec": float(unique.loc[unique.process_status == "success", "inference_sec"].mean()),
        "sum_unique_inference_sec": float(unique.loc[unique.process_status == "success", "inference_sec"].sum()),
        "fclass_counts": df.fclass.value_counts().to_dict(),
        "flag_counts": dict(Counter(f for flags in df.quality_flags for f in flags.split(";") if f)),
        "statistical_upper_99_5pct_by_fclass": thresholds,
        **run_info,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_image_rows", "n_unique_sha256", "n_failed_images", "n_review_rows", "median_unique_inference_sec")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-ids", type=int, nargs="*")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_inputs(set(args.only_ids) if args.only_ids else None)
    journal_path = out / "unique_progress.jsonl"
    journal = load_journal(journal_path)
    mismatches = {v.get("precision", "fp32") for v in journal.values()} - {args.precision}
    if mismatches:
        raise RuntimeError(f"Journal precision {mismatches} differs from requested {args.precision}; use a separate output directory")
    first_rows = rows.drop_duplicates("sha256")
    pending = first_rows[
        ~first_rows.sha256.isin(journal) |
        (first_rows.sha256.map(lambda s: journal.get(s, {}).get("status") == "failed") & args.retry_failed)
    ]
    print(f"input images={len(rows)} unique={len(first_rows)} cached={len(first_rows)-len(pending)} pending={len(pending)}", flush=True)
    context = None
    started = time.perf_counter()
    processed = 0
    failures = 0
    if len(pending):
        context = build_context(args.precision)
        print(f"models loaded in {context['load_sec']:.2f}s on {context['device']}", flush=True)
        with journal_path.open("a", encoding="utf-8") as sink:
            for row in pending.itertuples(index=False):
                try:
                    image_id_num = int(row.source_objectid)
                    save_preview = image_id_num in PILOT_IDS or int(str(row.sha256)[:8], 16) % 100 == 0
                    features = run_inference(row, context, save_preview, out)
                    entry = {"sha256": row.sha256, "inference_image_id": row.image_id, "status": "success", "precision": args.precision, "features": features}
                except Exception as exc:
                    failures += 1
                    if context["device"] == "cuda":
                        context["torch"].cuda.empty_cache()
                    entry = {
                        "sha256": row.sha256, "inference_image_id": row.image_id,
                        "status": "failed", "precision": args.precision, "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }
                    print(f"ERROR {row.image_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
                sink.flush()
                processed += 1
                journal[row.sha256] = entry
                if processed % 25 == 0:
                    os.fsync(sink.fileno())
                if processed % args.progress_every == 0 or processed == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = processed / elapsed if elapsed else 0
                    eta = (len(pending) - processed) / rate if rate else 0
                    print(f"progress {processed}/{len(pending)} failed={failures} elapsed={elapsed:.1f}s eta={eta/60:.1f}min", flush=True)
            os.fsync(sink.fileno())

    import importlib.metadata
    previous = {}
    previous_path = out / "parameters.json"
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    run_info = {
        "run_wall_sec": time.perf_counter() - started if processed else previous.get("run_wall_sec", 0.0),
        "new_unique_processed": processed if processed else previous.get("new_unique_processed", 0),
        "model_load_sec": context["load_sec"] if context else previous.get("model_load_sec", 0.0),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "device": context["device"] if context else ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        "gpu": context["torch"].cuda.get_device_name(0) if context and context["device"] == "cuda" else previous.get("gpu"),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "ultralytics", "pillow", "numpy", "pandas")},
        "seg_model_commit": context["seg_commit"] if context else previous.get("seg_model_commit"),
        "yolo_weight_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "parameters": PARAMS,
        "inference_precision": args.precision,
    }
    (out / "parameters.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize(rows, journal, out, run_info)


if __name__ == "__main__":
    main()
