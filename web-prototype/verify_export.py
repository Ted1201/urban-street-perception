from pathlib import Path
import json,math
from PIL import Image

root=Path(__file__).resolve().parent
d=json.loads((root/'dist/data/project.json').read_text(encoding='utf-8'))
ids={r['road_id'] for r in d['roads']}
assert len(ids)==1537 and len(d['photos'])==5920
assert sum(r['visual_subindex'] is not None for r in d['roads'])==304
assert sum(r['environment_index'] is not None for r in d['roads'])==46
for p in d['photos']:
 assert p['road_id'] in ids and (root/'dist/photos'/f"{p['source_sha256']}.jpg").is_file()
assert len({(p['task'],p['feature_set'],p['model'],p['road_id']) for p in d['predictions']})==3500
for p in d['predictions']:
 assert p['road_id'] in ids and all(math.isfinite(p[k]) for k in ['y_true','y_pred'])
for task,expected in [('visual_proxy',(73,11.116677700791223)),('complete_proxy',(7,13.652093410483262))]:
 fs=d['neuralSelected'][task]['feature_set']
 rows=[p for p in d['predictions'] if p['task']==task and p['feature_set']==fs and p['model']=='MLP' and p['split']=='test']
 assert len(rows)==expected[0]
 assert abs(sum(abs(p['y_true']-p['y_pred']) for p in rows)/len(rows)-expected[1])<1e-6
for p in sorted((root/'dist/photos').glob('*.jpg'))[::250]:
 with Image.open(p) as im: assert im.width<=512 and im.height>0
print('Verified 1537 roads, 5920 image links, 3500 predictions and frozen test metrics.')
