# 澳门街景视觉特征｜全量提取 v1

本目录由 `scripts/extract_street_view_features_full.py` 生成。它只处理 `images.csv` 中实际存在的图片；原本无照片的采样点不生成特征行，也不设缺图标记。每个 `image_id` 一行，连接 `sample_id`、`road_id`、道路等级与采样点区域。

## 文件

| 文件 | 用途 |
| --- | --- |
| `street_view_features_full.csv` | 完整逐图特征表；运行结束后生成 |
| `anomaly_review.csv` | 规则标记或统计尾部值的人工复核清单；运行结束后生成 |
| `summary.json` | 总数、质量标记频数、处理时间；运行结束后生成 |
| `parameters.json` | 模型、权重哈希、硬件、切片及阈值参数；运行结束后生成 |
| `unique_progress.jsonl` | 每个不同 SHA-256 一条的断点日志；中途重启可续跑 |
| `qa_previews/` | 定向和确定性抽样的预测叠加图，用于目视核查 |

配套 Notebook：`notebooks/澳门街景全量特征提取与检查.ipynb`。在 PyCharm 中打开项目和 Notebook，使用 `C:\Users\25142\anaconda3\python.exe` 内核；把 `RUN_EXTRACTION` 设为 `True` 可启动或续跑，已有结果时保持 `False` 即可查看图表。

## 特征与质量口径

- `vegetation_share`、`building_share`、`sky_share` 是各类在**有效画面像素**中的比例。统一排除下沿 8% 的车头／相机区域；并非绿地面积、建筑占地或完整绿视率。
- `motor_vehicle_count` 是一张静态街景中检出的汽车、摩托车、公交车、货车实例数；包含停放车辆，不能解释为交通流量。`person_count` 是检测到的人物数；`pedestrian_count_est` 结合语义掩膜估计行人数，待判者单列 `pedestrian_uncertain_count`。
- `quality_status=auto_pass` 仅表示**自动规则未触发**，不代表人工验证准确；`needs_review` 表示质量标记需人工复核，预测值仍保留；`failed` 表示该图处理失败，对应特征为空而不是零。
- `quality_flags` 以分号分隔。`tile_disagreement_high` 与 `sky_building_seam_conflict` 检查重叠切片的语义冲突；`vehicle_overlap_review` 检查跨切片车辆框可能重复；`tunnel_vegetation_review`、`zero_sky_high_vegetation_review`、`bridge_building_review` 检查试验中出现的特殊场景误识别；`cross_road_exact_duplicate`、`source_priority_manual_review`、`region_boundary_review` 来自数据关联和区域复核信息。
- `statistical_tail_fields` 用道路等级内各特征的 99.5 百分位提示极端值，仅用于排查，不能自动判错或删除。统计分布应按 `source_sha256` 去重，避免精确重复图片增加权重。

模型基线与已知局限见 `outputs/street_view_feature_plan_v1/方法选型说明.md` 和 `outputs/street_view_feature_pilot_v1/问题清单.md`。文件级图片—道路编号关联成立，但具体拍摄道路尚未全面独立核实；因此道路级比较需单列待确认样本和跨道路重复图。
