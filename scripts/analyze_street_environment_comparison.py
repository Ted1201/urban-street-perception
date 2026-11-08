"""Descriptive street environment proxy comparisons and research figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/street_environment_comparison_v1'
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'outputs/environment_indicators_v1/街道环境评价指标全表.csv',encoding='utf-8-sig')
im=pd.read_csv(ROOT/'outputs/street_view_features_v1/street_view_features_full.csv',encoding='utf-8-sig')
regions=['澳门半岛','氹仔','路氹','路环']
classes=['primary','secondary','tertiary']
visual=d[d.visual_subindex.notna()].copy()
full=d[d.environment_index.notna()].copy()
ordinary=visual[visual.structure_group=='普通道路']
tables=[]
for key in ['fclass','structure_group','region_group']:
    for name,g in d.groupby(key):
        for metric in ['visual_subindex','environment_index']:
            s=g[metric].dropna()
            tables.append(dict(dimension=key,group=name,metric=metric,total_roads=len(g),valid_roads=len(s),coverage_pct=100*len(s)/len(g),median=s.median(),q25=s.quantile(.25),q75=s.quantile(.75),small_sample=len(s)<10))
stats=pd.DataFrame(tables)
stats.to_csv(OUT/'街道类型指标对比.csv',index=False,encoding='utf-8-sig')
ordinary.groupby('fclass').agg(n=('road_id','size'),visual_median=('visual_subindex','median'),green_median=('green_median','median'),vehicle_median=('vehicle_median','median')).to_csv(OUT/'普通道路等级对比.csv',encoding='utf-8-sig')

sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':110})
fig,ax=plt.subplots(2,2,figsize=(14,10),layout='constrained')
for a,frame,key,order,title in [
 (ax[0,0],visual,'fclass',classes,'视觉子指数：道路等级'),
 (ax[0,1],visual,'structure_group',['普通道路','桥梁','隧道'],'视觉子指数：结构类型'),
 (ax[1,0],visual[visual.region_group.isin(regions)],'region_group',regions,'视觉子指数：四区域'),
 (ax[1,1],full,'fclass',classes,'完整指数：道路等级（仅46条）')]:
    y='environment_index' if frame is full else 'visual_subindex'
    sns.boxplot(data=frame,x=key,y=y,order=order,ax=a,color='#87b4a1',showfliers=True)
    counts=frame[key].value_counts()
    a.set_xticks(range(len(order)),[f'{x}\nn={counts.get(x,0)}' for x in order])
    a.set(title=title,xlabel='',ylabel='相对得分（0–100）',ylim=(-3,103))
fig.savefig(OUT/'02_街道类型箱线图.png',dpi=170)
plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
sns.scatterplot(data=visual,x='green_median',y='vehicle_median',hue='fclass',hue_order=classes,ax=ax[0],alpha=.7)
ax[0].set(title=f'可见绿化与可见车辆（n={len(visual)}）',xlabel='道路可见绿化比例中位数',ylabel='道路每图可见机动车数中位数')
sns.scatterplot(data=full,x='speed_valid',y='visual_subindex',hue='structure_group',ax=ax[1],alpha=.8)
ax[1].set(title=f'标注限速与视觉子指数（n={len(full)}）',xlabel='标注限速（km/h）',ylabel='视觉子指数')
fig.savefig(OUT/'03_街景与道路属性散点图.png',dpi=170)
plt.close(fig)

# Raw-component association only: composite versus own component is circular.
cols=['green_median','vehicle_median','building_context','sky_context','person_context','speed_valid']
labels=['可见绿化','可见车辆','可见建筑','可见天空','可见人物','标注限速']
x=visual[cols]
corr=x.corr(method='spearman',min_periods=10)
counts=x.notna().astype(int).T.dot(x.notna().astype(int))
corr.to_csv(OUT/'Spearman相关系数.csv',encoding='utf-8-sig')
counts.to_csv(OUT/'相关性有效样本数.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(1,2,figsize=(14,6),layout='constrained')
sns.heatmap(corr.set_axis(labels).set_axis(labels,axis=1),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',square=True,ax=ax[0])
ax[0].set_title('原始指标 Spearman 相关（视觉合格道路内）')
sns.heatmap(counts.set_axis(labels).set_axis(labels,axis=1),annot=True,fmt='d',cmap='Blues',square=True,ax=ax[1])
ax[1].set_title('每对指标有效道路数（非因果关系）')
fig.savefig(OUT/'04_相关性与样本量.png',dpi=170)
plt.close(fig)

# Coordinates are representative sampled-image locations, not road geometry.
coord=im[(im.process_status=='success') & ~im.quality_flags.fillna('').str.contains('cross_road_exact_duplicate|source_priority_manual_review')].drop_duplicates(['road_id','source_sha256']).groupby('road_id')[['longitude','latitude']].median()
mapped=d.merge(coord,on='road_id',how='left',validate='one_to_one')
geo=json.loads((ROOT/'outputs/basic_street_comparison_v1/region_boundaries.geojson').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,7),layout='constrained')
for a,metric,title in [(ax[0],'visual_subindex','视觉子指数'),(ax[1],'environment_index','道路属性与视觉完整指数')]:
    for f in geo['features']:
        geometry=f['geometry']; polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
        for polygon in polygons:
            for ring in polygon:
                xy=np.asarray(ring);a.plot(xy[:,0],xy[:,1],color='#b8b8b8',linewidth=.8,zorder=0)
    m=mapped.dropna(subset=[metric,'longitude','latitude'])
    sc=a.scatter(m.longitude,m.latitude,c=m[metric],vmin=0,vmax=100,cmap='viridis',s=18,alpha=.8)
    a.set(title=f'{title}\nn={len(m)}；道路采样点代表位置',xlabel='经度',ylabel='纬度')
    a.set_aspect(1/np.cos(np.deg2rad(22.15)))
    a.ticklabel_format(axis='both',style='plain',useOffset=False)
fig.colorbar(sc,ax=ax,label='相对得分（0–100）',shrink=.5)
fig.savefig(OUT/'01_环境代理指标地图.png',dpi=180,bbox_inches='tight')
plt.close(fig)
mapped[['road_id','longitude','latitude','visual_subindex','environment_index']].to_csv(OUT/'地图道路代表点.csv',index=False,encoding='utf-8-sig')

def md(t):
    t=t.fillna('—').astype(str)
    return '| '+' | '.join(t.columns)+' |\n| '+' | '.join(['---']*len(t.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in t.values)
vstats=stats[stats.metric=='visual_subindex'][['dimension','group','total_roads','valid_roads','coverage_pct','median','q25','q75']].round(2)
pairs=[]
for i in range(len(cols)):
 for j in range(i+1,len(cols)):
    if pd.notna(corr.iloc[i,j]):pairs.append((labels[i],labels[j],corr.iloc[i,j],int(counts.iloc[i,j])))
pairs=sorted(pairs,key=lambda z:abs(z[2]),reverse=True)
group_medians=visual.groupby('fclass').visual_subindex.median()
region_medians=visual[visual.region_group.isin(regions)].groupby('region_group').visual_subindex.median()
structure_medians=visual.groupby('structure_group').visual_subindex.median()
report=f'''# 澳门街道环境质量代理与街道类型对比：分析初稿

本稿沿用已确定的街道环境质量代理目标。现有数据没有主观幸福度观测，分析对象是可见绿化、静态可见车辆及标注限速形成的代理指标。道路记录是分析单元，不等同于整条命名街道。

## 主要结论

1. 全部道路1,537条，视觉子指数可用{len(visual)}条（{len(visual)/len(d):.1%}），完整指数可用{len(full)}条（{len(full)/len(d):.1%}）。完整指数仅覆盖46条，受到限速缺值和视觉质量筛选共同影响，不能外推为全澳门街道排名。
2. 视觉样本含普通道路{len(ordinary)}条、桥梁{int(visual.structure_group.eq('桥梁').sum())}条、隧道{int(visual.structure_group.eq('隧道').sum())}条。完整样本含桥梁{int(full.structure_group.eq('桥梁').sum())}条，占{full.structure_group.eq('桥梁').mean():.1%}，存在结构类型选择偏差。隧道无合格指数样本，不能判断隧道环境优劣。
3. 道路等级的视觉子指数中位数分别为：primary {group_medians['primary']:.2f}、secondary {group_medians['secondary']:.2f}、tertiary {group_medians['tertiary']:.2f}。这些是当前合格样本的描述性差异；混合了区域、结构类型和图片覆盖差异。另附普通道路内的等级对比表，以便检查桥梁构成影响。
4. 分组统计见下表。区域比较仅使用四区域明确归属道路；跨区或边界待确认道路单列，少于10条的组仅作小样本描述。箱线图展示中位数、四分位范围与离群值，不以图上差异代替显著性或因果判断。
5. 对原始环境指标计算Spearman相关，同时提供每对有效样本量。未将综合分数与其组成项之间的公式相关解释为独立研究发现。关联可能受区域和道路类型混杂，且相邻道路具有空间依赖。

四区视觉子指数中位数为：澳门半岛 {region_medians['澳门半岛']:.2f}、氹仔 {region_medians['氹仔']:.2f}、路氹 {region_medians['路氹']:.2f}、路环 {region_medians['路环']:.2f}。路环样本较高、路氹较低；普通道路中位数 {structure_medians['普通道路']:.2f}，桥梁 {structure_medians['桥梁']:.2f}。这些差异仅适用于已筛选的图片和道路，尚未控制道路等级等混杂因素。

## 视觉子指数分组结果

{md(vstats)}

## 原始指标关联（按绝对相关系数列前5对）

{md(pd.DataFrame(pairs[:5],columns=['指标1','指标2','Spearman_r','有效道路数']).round(3))}

相关性以304条视觉合格道路为母样本；涉及限速时仅46条完整记录参与。分项质量筛选及至少3张不同图片的门槛沿用指标计算规则。建筑、天空、人物是背景描述值，仍可能含模型误识别。未开展多重显著性检验，不作显著性宣称。

可见建筑与天空占比的强负相关（约−0.84）部分反映固定画面中的像素份额竞争及空间围合，不能直接解释为建筑对环境质量的负面影响。机动车数与人物数呈正相关（约0.44），可作为街道活动共现的探索线索；尚不能区分道路类型、时段或拍摄位置造成的共同变化。

## 地图及图表解释

- 01地图：用道路保留图片坐标中位数表示道路代表位置，展示采样空间分布；不是道路中心线或连续空间插值。底图沿用已有四区判界文件，边界依据见 `outputs/basic_street_comparison_v1/boundary_sources.json`，并非新取得的官方精确行政边界。
- 02箱线图：分别展示视觉与完整指数，不能混合排名；n=0表示无可用指数，不代表0分。
- 03散点图：绿化与机动车的原值关系，以及标注限速与视觉子指数的关系；可见车辆数含停放车辆。
- 04相关图：左图为原指标相关系数，右图为对应有效样本量，少于10条不计算相关。

## 结论适用范围与后续核验

本稿是探索性描述分析。三种替代权重使31/46条完整指数道路改变分组，评分解释应保留权重不确定性；现阶段不宜给出精确优劣名次。图片未形成时段均衡的道路普查，分割和检测仍有已知错误。下一步宜独立抽查图片与道路归属、完善有效限速，并用独立人工环境评价验证代理目标。若开展推断，应处理道路邻近性和样本选择问题。
'''
(OUT/'分析结果初稿与主要结论.md').write_text(report,encoding='utf-8')
assert int(stats[(stats.dimension=='fclass')&(stats.metric=='visual_subindex')].valid_roads.sum())==len(visual)
assert int(counts.loc['speed_valid','green_median'])==len(full)
print(json.dumps({'visual':len(visual),'complete':len(full),'figures':4,'pairs':pairs[:5]},ensure_ascii=False))
