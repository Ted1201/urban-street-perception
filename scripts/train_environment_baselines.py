"""Spatially held-out baselines for environmental proxy targets."""
from pathlib import Path
import json,time,platform
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/environment_baselines_v1'
OUT.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(ROOT/'outputs/environment_indicators_v1/街道环境评价指标全表.csv',encoding='utf-8-sig')
loc=pd.read_csv(ROOT/'outputs/street_environment_comparison_v1/地图道路代表点.csv',encoding='utf-8-sig')
d=raw[raw.visual_subindex.notna()].merge(loc[['road_id','longitude','latitude']],on='road_id',validate='one_to_one').reset_index(drop=True)
assert len(d)==304 and d.osm_id.is_unique and d[['longitude','latitude']].notna().all().all()
d['spatial_group']=np.floor(d.longitude/.005).astype(int).astype(str)+'_'+np.floor(d.latitude/.005).astype(int).astype(str)
# First seed meeting predeclared sample-size gates; no target values or model scores inspected.
for seed in range(42,142):
    trval,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(d,groups=d.spatial_group))
    t,v=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1000).split(d.iloc[trval],groups=d.iloc[trval].spatial_group))
    tr,va=trval[t],trval[v]
    if len(tr)>=140 and len(va)>=40 and len(te)>=40 and d.iloc[tr].environment_index.notna().sum()>=20 and d.iloc[va].environment_index.notna().sum()>=5 and d.iloc[te].environment_index.notna().sum()>=5:
        break
else: raise RuntimeError('No spatial split satisfies sample-size gates')
d['split']='';d.loc[tr,'split']='train';d.loc[va,'split']='validation';d.loc[te,'split']='test'
assert d.groupby('spatial_group').split.nunique().max()==1
cat=['fclass','structure_group','oneway']
sets={'road_only':cat+['layer'],'road_context':cat+['layer','building_context','sky_context','person_context']}
forbidden={'green_median','vehicle_median','speed_valid','maxspeed','green_score','vehicle_score','speed_score','visual_subindex','environment_index','green_priority','vehicle_priority','speed_priority'}
assert all(not (set(cols)&forbidden) for cols in sets.values())
def pipeline(cols,estimator):
    numeric=[c for c in cols if c not in cat]
    return Pipeline([('preprocess',ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
    ])),('model',estimator)])
def metrics(y,p):
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'n':len(y)}
rows=[];predictions=[];normalizers={};winners={};models={};labels=[]
for task,components in [('visual_proxy',['green_median','vehicle_median']),('complete_proxy',['green_median','vehicle_median','speed_valid'])]:
    z=d[d.environment_index.notna()].copy() if task=='complete_proxy' else d.copy()
    train=z[z.split=='train'];params={};scores=[]
    # Label normalizers learned from task training roads only, not all-data v1 scores.
    for col in components:
        xx=np.log1p(z[col]) if col=='vehicle_median' else z[col].astype(float)
        fit=xx[z.split=='train'];low,high=fit.quantile([.05,.95]);params[col]={'q05':float(low),'q95':float(high),'log1p':col=='vehicle_median','reverse':col!='green_median'}
        s=100*((xx.clip(low,high)-low)/(high-low)) if high>low else pd.Series(50.,index=z.index)
        if col!='green_median':s=100-s
        scores.append(s)
    z['target']=pd.concat(scores,axis=1).mean(axis=1,skipna=False)
    assert z.target.notna().all() and z.target.between(0,100).all()
    normalizers[task]=params
    labels.append(z[['road_id','split','target']].assign(task=task))
    candidates=[('mean',{},DummyRegressor()),('linear',{},LinearRegression()),
        ('random_forest',{'min_samples_leaf':3},RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=4)),
        ('random_forest',{'min_samples_leaf':8},RandomForestRegressor(n_estimators=300,min_samples_leaf=8,max_features=.8,random_state=42,n_jobs=4)),
        ('gradient_boosting',{'max_depth':1},GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=1,min_samples_leaf=5,random_state=42)),
        ('gradient_boosting',{'max_depth':2},GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,min_samples_leaf=5,random_state=42))]
    selected=[]
    train=z[z.split=='train'];valid=z[z.split=='validation'];test=z[z.split=='test']
    for fs,cols in sets.items():
        fitted=[]
        for family,hp,estimator in candidates:
            from sklearn.base import clone
            model=pipeline(cols,clone(estimator));start=time.perf_counter();model.fit(train[cols],train.target)
            vm=metrics(valid.target,model.predict(valid[cols]));elapsed=time.perf_counter()-start
            fitted.append((family,hp,model,vm,elapsed))
        for family in ['mean','linear','random_forest','gradient_boosting']:
            family_best=min((x for x in fitted if x[0]==family),key=lambda x:x[3]['MAE'])
            fam,hp,model,vm,elapsed=family_best
            selected.append((vm['MAE'],fs,fam,hp,model))
            for split,sub in [('train',train),('validation',valid),('test',test)]:
                pp=model.predict(sub[cols]);m=metrics(sub.target,pp)
                rows.append(dict(task=task,feature_set=fs,model=fam,parameters=json.dumps(hp),split=split,fit_sec=elapsed,**m))
                predictions.extend(dict(task=task,feature_set=fs,model=fam,road_id=r,split=split,y_true=float(y),y_pred=float(p)) for r,y,p in zip(sub.road_id,sub.target,pp))
    # Winner fixed on validation MAE, includes the null model as a valid winner.
    best=min(selected,key=lambda x:x[0]);val_mae,fs,fam,hp,model=best
    winners[task]={'feature_set':fs,'model':fam,'parameters':hp,'validation_MAE':val_mae,'counts':z.split.value_counts().to_dict(),'test_metrics':metrics(test.target,model.predict(test[sets[fs]]))}
    joblib.dump({'pipeline':model,'features':sets[fs],'target_normalizers':params,'task':task},OUT/f'{task}_selected_model.joblib')
results=pd.DataFrame(rows);results.to_csv(OUT/'基线模型评价指标.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(predictions).to_csv(OUT/'各模型逐道路预测.csv',index=False,encoding='utf-8-sig')
pd.concat(labels).to_csv(OUT/'训练尺度目标值.csv',index=False,encoding='utf-8-sig')
d[['road_id','osm_id','spatial_group','longitude','latitude','split']].to_csv(OUT/'训练验证测试划分.csv',index=False,encoding='utf-8-sig')
config={'seed':seed,'split_rule':'0.005 degree grid groups; approximate 60/20/20; no spatial buffer','features':sets,'target_normalization':'train-only q05/q95 clipped minmax; equal weights; vehicle log1p','normalizers':normalizers,'selected':winners,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},'grid_counts':d.groupby('split').spatial_group.nunique().to_dict()}
(OUT/'模型参数与结果摘要.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
for i,task in enumerate(['visual_proxy','complete_proxy']):
    a=axes[i,0];b=axes[i,1];s=results[(results.task==task)&results.split.isin(['validation','test'])].copy();s['label']=s.feature_set+' / '+s.model
    pp=s.pivot(index='label',columns='split',values='MAE');pp.plot.barh(ax=a,color=['#bc896f','#70a38c']);a.set(title=task+'：验证与测试 MAE',xlabel='绝对误差（目标分）',ylabel='')
    w=winners[task];pr=pd.DataFrame(predictions);pr=pr[(pr.task==task)&(pr.feature_set==w['feature_set'])&(pr.model==w['model'])&(pr.split=='test')]
    b.scatter(pr.y_true,pr.y_pred,color='#5c92b1');b.plot([0,100],[0,100],'--',color='gray');b.set(title=f"验证集选定模型：{w['model']}；测试n={len(pr)}",xlabel='测试目标值',ylabel='预测值',xlim=(-5,105))
fig.savefig(OUT/'基线模型评价与预测.png',dpi=170)
plt.close(fig)
def markdown_table(df):
    df=df.round(3).astype(str)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(x)+' |' for x in df.values)
test_results=results[results.split=='test'][['task','feature_set','model','n','MAE','RMSE','R2']]
text=f'''# 环境质量代理目标：基线模型结果

已完成均值预测、线性回归、随机森林、梯度提升基线。视觉代理目标304条道路为主分析；完整代理目标46条仅作小样本补充。目标是既定环境代理公式，不是独立环境质量真值。

## 数据划分与防泄漏

以道路记录为样本，304条道路的OSM编号不重复。按经纬度0.005度网格（约500米量级）成组划分，网格不跨训练、验证、测试集，约60%/20%/20%，随机种子{seed}。共同划分先固定，再取完整目标子样本。种子仅按预设样本数量下限选择，未依据目标得分或模型表现选择。没有空间缓冲带，相邻网格的道路仍可能相近，不能宣称完全空间独立；不等同于跨城市泛化。

全部特征缺值填补、数值缩放及类别编码仅在训练集拟合，未知类别忽略。绿化、车辆、限速及其标准分、综合分数、计数覆盖与质量标记不进入自变量。road_only使用道路等级、结构类型、单向属性和layer；road_context另加建筑、天空、人物背景特征。这些视觉背景与标签来自同批图像，可能共享识别误差，解释为同源背景信息的预测关联。

**目标标准化也仅在对应任务训练集拟合**：对训练道路原指标拟合5%/95%分位数，再应用于验证、测试道路。因此`训练尺度目标值.csv`与此前全样本标准化的v1指数可能不同，不能混用评价指标或分组。原v1数据表不改写。低可见车辆取log(1+x)，正负方向和等权公式沿用既定规则。

随机森林比较min_samples_leaf=3/8；梯度提升比较深度1/2；线性回归和均值预测无调参。先按验证集MAE为各模型选择参数，再按验证集MAE选定模型与特征集，包含均值预测参照。所有模型仍只拟合训练集，测试集不用于选择或重新调参。测试分数为一次固定划分结果，无多次交叉验证置信区间。

## 验证集选定模型

'''
for task,w in winners.items():
    text+=f"- {task}：{w['counts']}；{w['feature_set']} / {w['model']}；测试 MAE={w['test_metrics']['MAE']:.3f}，RMSE={w['test_metrics']['RMSE']:.3f}，R²={w['test_metrics']['R2']:.3f}。\n"
text+='\n## 各基线的测试集表现\n\n'+markdown_table(test_results)+'\n\nMAE、RMSE越小越好；R²可能为负，表示相对测试样本均值的平方误差参照较差。均值模型实际预测的是训练集均值。报告所有基线测试分数用于比较，不以测试最佳重新选定模型。预测值不截断，以免隐藏误差。完整目标测试集很小，不据此确定模型优劣或宣称已达到应用精度。\n\n结果只针对有合格图像和属性的道路；目标仍是研究者设定的环境质量代理，不能作为独立环境效度验证。后续需要更多标注、独立环境评价和空间重复划分核查。\n\n方法实现参考 [scikit-learn 数据泄漏与 Pipeline 说明](https://scikit-learn.org/1.5/common_pitfalls.html)；该文档支持训练集拟合预处理的流程，不为本项目代理目标提供效度背书。\n'
main_result=results[(results.task=='visual_proxy')&(results.split=='test')]
null_mae=float(main_result[main_result.model=='mean'].MAE.iloc[0])
selected_mae=winners['visual_proxy']['test_metrics']['MAE']
text+=f'\n## 初步结论\n\n主分析验证集选定模型为 {winners["visual_proxy"]["feature_set"]} / {winners["visual_proxy"]["model"]}，测试MAE为{selected_mae:.3f}，训练均值预测MAE为{null_mae:.3f}，误差相对变化为{100*(selected_mae/null_mae-1):.1f}%。模型对既定代理目标提供的预测能力不等于对独立环境质量的验证。完整目标测试集仅7条道路，验证与测试误差差异明显，应等待更多样本再解释稳定性。\n'
(OUT/'基线模型结果与评价说明.md').write_text(text,encoding='utf-8')
assert d.road_id.is_unique and not (set(d.loc[d.split=='train','spatial_group']) & set(d.loc[d.split=='test','spatial_group']))
assert len(results)==48 and len(pd.DataFrame(predictions))==8*(304+46)
print(json.dumps(winners,ensure_ascii=False))
