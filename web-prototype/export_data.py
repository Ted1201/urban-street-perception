"""Export the project's verified research tables and small display thumbnails."""
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'dist'
(OUT/'data').mkdir(parents=True,exist_ok=True)
(OUT/'photos').mkdir(exist_ok=True)
def records(frame):
 return json.loads(frame.to_json(orient='records',force_ascii=False))
roads=pd.read_csv(ROOT/'outputs/environment_indicators_v1/街道环境评价指标全表.csv',encoding='utf-8-sig')
images=pd.read_csv(ROOT/'outputs/street_view_features_v1/street_view_features_full.csv',encoding='utf-8-sig')
coords=images.groupby('road_id')[['longitude','latitude']].median().reset_index()
roads=roads.merge(coords,on='road_id',how='left',validate='one_to_one')
photo_cols=['image_id','road_id','source_sha256','vegetation_share','building_share','sky_share','motor_vehicle_count','person_count','quality_status','quality_flags','point_region']
photos=records(images[photo_cols])
baseline=pd.read_csv(ROOT/'outputs/environment_baselines_v1/各模型逐道路预测.csv',encoding='utf-8-sig')
neural=pd.read_csv(ROOT/'outputs/environment_mlp_week13/神经网络逐道路预测.csv',encoding='utf-8-sig')
neural=neural[neural.seed==42].assign(model='MLP')
pc=['task','feature_set','model','road_id','split','y_true','y_pred']
pred=records(pd.concat([baseline[pc],neural[pc]],ignore_index=True))
geo=json.loads((ROOT/'outputs/basic_street_comparison_v1/region_boundaries.geojson').read_text(encoding='utf-8'))
neural_selected=json.loads((ROOT/'outputs/environment_mlp_week13/神经网络实验摘要.json').read_text(encoding='utf-8'))['selected']
data={'neuralSelected':neural_selected,'roads':records(roads),'photos':photos,'predictions':pred,'boundaries':geo,'meta':{'images':len(images),'roads':len(roads),'visual':int(roads.visual_subindex.notna().sum()),'complete':int(roads.environment_index.notna().sum()),'snapshot':'2026-11-08','thumbnail_width':512}}
(OUT/'data/project.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
unique=images.drop_duplicates('source_sha256')
def thumbnail(row):
 dest=OUT/'photos'/f'{row.source_sha256}.jpg'
 if dest.exists():return
 with Image.open(ROOT/'outputs/structured_data_v1'/row.relative_path) as source:
  im=ImageOps.exif_transpose(source).convert('RGB');im.thumbnail((512,512));im.save(dest,'JPEG',quality=52,optimize=True)
with ThreadPoolExecutor(max_workers=6) as pool:
 list(pool.map(thumbnail,unique.itertuples(index=False)))
assert len(list((OUT/'photos').glob('*.jpg')))==images.source_sha256.nunique()
print(json.dumps({'roads':len(roads),'images':len(images),'unique_thumbnails':len(unique),'predictions':len(pred),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
