"""Street environment quality proxy indicators; exploratory operational target, not independently validated."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/environment_indicators_v1'
OUT.mkdir(parents=True, exist_ok=True)
roads = pd.read_csv(ROOT/'outputs/structured_data_v1/tables/roads.csv', encoding='utf-8-sig')
images = pd.read_csv(ROOT/'outputs/street_view_features_v1/street_view_features_full.csv', encoding='utf-8-sig')
flags = images.quality_flags.fillna('')
# Exact duplicates across roads cannot be confidently assigned to either road.
base = images[(images.process_status=='success') & ~flags.str.contains('cross_road_exact_duplicate|source_priority_manual_review')].drop_duplicates(['road_id','source_sha256']).copy()
flags = base.quality_flags.fillna('')
base['green_usable'] = ~flags.str.contains('tile_disagreement_high|tunnel_vegetation_review|zero_sky_high_vegetation_review')
base['vehicle_usable'] = ~flags.str.contains('vehicle_overlap_review')
records = []
for road_id, g in base.groupby('road_id'):
    green = g.loc[g.green_usable,'vegetation_share']
    vehicle = g.loc[g.vehicle_usable,'motor_vehicle_count']
    regions = g.loc[~g.near_boundary_or_outside.astype(bool),'point_region']
    unique_regions = sorted(set(regions)-{'未归属/边界'})
    records.append(dict(road_id=road_id, n_unique_images=len(g), n_green_usable=len(green), n_vehicle_usable=len(vehicle),
        green_median=green.median() if len(green)>=3 else np.nan,
        vehicle_median=vehicle.median() if len(vehicle)>=3 else np.nan,
        building_context=g.building_share.median(),sky_context=g.sky_share.median(),person_context=g.person_count.median(),
        region_group=unique_regions[0] if len(unique_regions)==1 else '跨区或边界待确认',
        flagged_image_share=g.quality_status.ne('auto_pass').mean()))
d = roads.merge(pd.DataFrame(records), on='road_id', how='left', validate='one_to_one')
for c in ['n_unique_images','n_green_usable','n_vehicle_usable']:
    d[c]=d[c].fillna(0).astype(int)
d['region_group']=d.region_group.fillna('未纳入区域比较')
d['structure_group']=np.select([d.tunnel.eq('T'), d.bridge.eq('T')], ['隧道','桥梁'], default='普通道路')
d['speed_valid']=pd.to_numeric(d.maxspeed,errors='coerce').where(lambda s:s.gt(0)&s.le(130))
params = {'version':'v1_exploratory','minimum_unique_images_per_visual_metric':3,'scaling':'pooled road-level 5th–95th percentile clipped min-max, 0–100', 'weights':{'green':1/3,'vehicle':1/3,'speed':1/3},'thresholds':[20,40,60,80],'normalization':{}}
def scale(col, reverse=False, log=False):
    s=d[col].astype(float)
    x=np.log1p(s) if log else s
    low,high=x.quantile([.05,.95])
    if pd.isna(low) or pd.isna(high):
        y=pd.Series(np.nan,index=s.index)
    elif high<=low:
        y=pd.Series(np.where(s.notna(),50,np.nan),index=s.index)
    else:
        y=100*((x.clip(low,high)-low)/(high-low))
        if reverse: y=100-y
    params['normalization'][col]={'q05':None if pd.isna(low) else float(low),'q95':None if pd.isna(high) else float(high),'reverse':reverse,'log1p':log,'valid_roads':int(s.notna().sum())}
    return y
d['green_score']=scale('green_median')
d['vehicle_score']=scale('vehicle_median',reverse=True,log=True)
d['speed_score']=scale('speed_valid',reverse=True)
for col in ['building_context','sky_context','person_context']:
    d[col+'_standardized']=scale(col,log=col=='person_context')
d['visual_subindex']=d[['green_score','vehicle_score']].mean(axis=1,skipna=False)
d['environment_index']=d[['green_score','vehicle_score','speed_score']].mean(axis=1,skipna=False)
def group(s):
    return pd.cut(s,[-.001,20,40,60,80,100.001],labels=['相对较低','相对偏低','中间','相对偏高','相对较高'],right=False).astype('string')
d['environment_group']=group(d.environment_index)
d['visual_group']=group(d.visual_subindex)
alternatives=[]
for name,w in [('green_priority',[.5,.25,.25]),('vehicle_priority',[.25,.5,.25]),('speed_priority',[.25,.25,.5])]:
    d[name]=d[['green_score','vehicle_score','speed_score']].mul(w).sum(axis=1,min_count=3)
    alternatives.append(name)
d['sensitivity_score_range']=d[['environment_index',*alternatives]].max(axis=1)-d[['environment_index',*alternatives]].min(axis=1)
d['sensitivity_group_change']=False
for col in alternatives:
    d['sensitivity_group_change'] |= (group(d[col])!=d.environment_group).fillna(False)
d['calculation_status']=np.select([d.environment_index.notna(),d.visual_subindex.notna()],['三项完整','仅视觉子指数'],default='可用视觉样本不足')
assert len(d)==len(roads) and d.road_id.is_unique
assert d.loc[d.environment_index.notna(),['green_score','vehicle_score','speed_score']].notna().all().all()
assert d.loc[d.environment_index.notna(),'environment_index'].between(0,100).all()
d.to_csv(OUT/'街道环境评价指标全表.csv',index=False,encoding='utf-8-sig')
pd.crosstab([d.structure_group,d.fclass],d.environment_group,dropna=False).to_csv(OUT/'结构等级分组统计.csv',encoding='utf-8-sig')
summary={'roads':len(d),'roads_with_retained_images':int(d.n_unique_images.gt(0).sum()),'valid_speed_roads':int(d.speed_valid.notna().sum()),'visual_subindex_roads':int(d.visual_subindex.notna().sum()),'complete_index_roads':int(d.environment_index.notna().sum()),'weight_sensitive_roads':int(d.sensitivity_group_change.sum()),'groups':d.environment_group.value_counts().to_dict()}
(OUT/'计算参数.json').write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'结果摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
ax[0,0].hist(d.visual_subindex.dropna(),bins=np.arange(0,105,5),color='#669e80',alpha=.8,label='视觉子指数')
ax[0,0].hist(d.environment_index.dropna(),bins=np.arange(0,105,5),histtype='step',color='#b36b4f',linewidth=2,label='三项完整指数')
ax[0,0].set(title='指数分布：视觉子指数与完整指数分别统计',xlabel='相对得分（0–100）',ylabel='道路记录数');ax[0,0].legend()
counts=d.environment_group.value_counts().reindex(['相对较低','相对偏低','中间','相对偏高','相对较高'],fill_value=0)
ax[0,1].bar(counts.index,counts.values,color='#759cb4');ax[0,1].set(title='三项完整指数分组',ylabel='道路记录数')
for i,(c,label) in enumerate([('green_score','绿化'),('vehicle_score','低可见车辆'),('speed_score','低标注限速')]):
    ax[1,0].hist(d[c].dropna(),bins=15,histtype='step',linewidth=2,label=label)
ax[1,0].legend();ax[1,0].set(title='分项标准分分布（各自可用样本）',xlabel='相对得分',ylabel='道路记录数')
ax[1,1].hist(d.sensitivity_score_range.dropna(),bins=15,color='#ae96b6');ax[1,1].set(title='权重敏感性：四种方案得分极差',xlabel='最高得分 − 最低得分',ylabel='完整指数道路数')
fig.savefig(OUT/'环境评价指标分布.png',dpi=170)
print(json.dumps(summary,ensure_ascii=False))
