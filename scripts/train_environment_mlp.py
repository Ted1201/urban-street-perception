"""Week 13: reproducible MLP experiments on the frozen baseline benchmark."""
from pathlib import Path
from copy import deepcopy
import json,time,hashlib,platform
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/environment_baselines_v1'
OUT=ROOT/'outputs/environment_mlp_week13'
OUT.mkdir(parents=True,exist_ok=True)
config=json.loads((BASE/'模型参数与结果摘要.json').read_text(encoding='utf-8'))
features=config['features']
split=pd.read_csv(BASE/'训练验证测试划分.csv',encoding='utf-8-sig')
labels=pd.read_csv(BASE/'训练尺度目标值.csv',encoding='utf-8-sig')
roads=pd.read_csv(ROOT/'outputs/environment_indicators_v1/街道环境评价指标全表.csv',encoding='utf-8-sig')
assert split.road_id.is_unique and split.groupby('spatial_group').split.nunique().max()==1
assert labels.groupby(['task','road_id']).size().max()==1
forbidden={'green_median','vehicle_median','speed_valid','maxspeed','visual_subindex','environment_index','green_score','vehicle_score','speed_score'}
assert all(not (set(cols)&forbidden) for cols in features.values())
SEEDS=[42,52,62];MAX_EPOCHS=600;PATIENCE=50;MIN_DELTA=.0001
specs=[
 ('C01',(16,),'relu',.001,.0001),
 ('C02',(32,),'relu',.001,.0001),
 ('C03',(32,16),'relu',.001,.0001),
 ('C04',(32,),'tanh',.001,.0001),
 ('C05',(32,),'relu',.005,.0001),
 ('C06',(32,),'relu',.001,.01),
 ('C07',(32,),'tanh',.005,.01),
 ('C08',(32,16),'relu',.005,.01),
]
def metrics(y,p):
 return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'n':len(y)}
def preprocessor(cols):
 cats=['fclass','structure_group','oneway'];nums=[c for c in cols if c not in cats]
 return ColumnTransformer([
  ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
  ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
def predict(bundle,x):
 return bundle['target_scaler'].inverse_transform(bundle['network'].predict(bundle['preprocess'].transform(x[bundle['features']])).reshape(-1,1)).ravel()
trials=[];history=[];bundles={};chosen={};errors=[];predictions=[]
with threadpool_limits(limits=1), (OUT/'实验进度.jsonl').open('w',encoding='utf-8') as journal:
 for task in ['visual_proxy','complete_proxy']:
  z=labels[labels.task==task].merge(roads,on='road_id',validate='one_to_one')
  assert z[['road_id','split']].merge(split[['road_id','split']],on='road_id',suffixes=('_label','_fixed')).eval('split_label == split_fixed').all()
  tr=z[z.split=='train'];va=z[z.split=='validation'];te=z[z.split=='test']
  for fs,cols in features.items():
   prep=preprocessor(cols);xt=prep.fit_transform(tr[cols]);xv=prep.transform(va[cols])
   sy=StandardScaler().fit(tr[['target']]);yt=sy.transform(tr[['target']]).ravel()
   for cid,hidden,activation,lr,alpha in specs:
    for seed in SEEDS:
     model=MLPRegressor(hidden_layer_sizes=hidden,activation=activation,solver='adam',alpha=alpha,batch_size=min(32,len(tr)),learning_rate_init=lr,early_stopping=False,shuffle=True,random_state=seed,max_iter=1)
     best=float('inf');best_epoch=0;best_model=None;start=time.perf_counter()
     for epoch in range(1,MAX_EPOCHS+1):
      model.partial_fit(xt,yt)
      vp=sy.inverse_transform(model.predict(xv).reshape(-1,1)).ravel()
      tp=sy.inverse_transform(model.predict(xt).reshape(-1,1)).ravel()
      val_mae=float(mean_absolute_error(va.target,vp))
      history.append(dict(task=task,feature_set=fs,config_id=cid,seed=seed,epoch=epoch,train_MAE=float(mean_absolute_error(tr.target,tp)),validation_MAE=val_mae,train_loss_scaled=float(model.loss_)))
      if val_mae<best-MIN_DELTA:
       best=val_mae;best_epoch=epoch;best_model=deepcopy(model)
      if epoch-best_epoch>=PATIENCE:break
     bundle={'preprocess':prep,'target_scaler':sy,'network':best_model,'features':cols,'task':task,'config_id':cid,'seed':seed,'best_epoch':best_epoch,'target_normalizers':config['normalizers'][task]}
     bundles[(task,fs,cid,seed)]=bundle
     vm=metrics(va.target,predict(bundle,va))
     row=dict(task=task,feature_set=fs,config_id=cid,seed=seed,hidden_layers=str(hidden),activation=activation,learning_rate=lr,alpha=alpha,batch_size=min(32,len(tr)),best_epoch=best_epoch,stopped_epoch=epoch,hit_max_epochs=epoch==MAX_EPOCHS,trainable_parameters=sum(c.size for c in best_model.coefs_)+sum(c.size for c in best_model.intercepts_),fit_sec=time.perf_counter()-start,validation_MAE=vm['MAE'],validation_RMSE=vm['RMSE'],validation_R2=vm['R2'])
     trials.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
   # Choose architecture/hyperparameters from mean validation MAE across fixed seeds.
   frame=pd.DataFrame(trials);sub=frame[(frame.task==task)&(frame.feature_set==fs)]
   bestcid=sub.groupby('config_id').validation_MAE.mean().idxmin()
   chosen[(task,fs)]=bestcid
   for seed in SEEDS:
    bundle=bundles[(task,fs,bestcid,seed)]
    joblib.dump(bundle,OUT/f'{task}_{fs}_seed{seed}_mlp.joblib')
    for sn,ss in [('train',tr),('validation',va),('test',te)]:
     pp=predict(bundle,ss)
     errors.append(dict(task=task,feature_set=fs,model='MLP',config_id=bestcid,seed=seed,split=sn,**metrics(ss.target,pp)))
     predictions.extend(dict(task=task,feature_set=fs,config_id=bestcid,seed=seed,road_id=rid,split=sn,y_true=float(y),y_pred=float(p)) for rid,y,p in zip(ss.road_id,ss.target,pp))
   print(f'{task} / {fs}: selected {bestcid} after {len(specs)*len(SEEDS)} trials',flush=True)
trials=pd.DataFrame(trials);hist=pd.DataFrame(history);errs=pd.DataFrame(errors);pred=pd.DataFrame(predictions)
trials.to_csv(OUT/'神经网络参数实验记录.csv',index=False,encoding='utf-8-sig')
hist.to_csv(OUT/'逐轮训练过程.csv',index=False,encoding='utf-8-sig')
errs.to_csv(OUT/'神经网络误差结果.csv',index=False,encoding='utf-8-sig')
pred.to_csv(OUT/'神经网络逐道路预测.csv',index=False,encoding='utf-8-sig')
grid=trials.groupby(['task','feature_set','config_id'],as_index=False).agg(validation_MAE_mean=('validation_MAE','mean'),validation_MAE_std=('validation_MAE','std'),best_epoch_mean=('best_epoch','mean'))
grid.to_csv(OUT/'参数方案验证集对比.csv',index=False,encoding='utf-8-sig')
baselines=pd.read_csv(BASE/'基线模型评价指标.csv',encoding='utf-8-sig')
common=['task','feature_set','model','split','MAE','RMSE','R2','n']
comparison=pd.concat([baselines[common].assign(seed='baseline'),errs[errs.seed==42][common].assign(seed=42)],ignore_index=True)
comparison.to_csv(OUT/'神经网络与基线对比表.csv',index=False,encoding='utf-8-sig')
stability=errs[errs.split=='test'].groupby(['task','feature_set'],as_index=False).agg(test_MAE_mean=('MAE','mean'),test_MAE_std=('MAE','std'),test_RMSE_mean=('RMSE','mean'),test_R2_mean=('R2','mean'))
stability.to_csv(OUT/'随机种子稳定性.csv',index=False,encoding='utf-8-sig')
summary={'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'min_delta_MAE':MIN_DELTA,'seeds':SEEDS,'main_seed':42,'trials':len(trials),'selection':'mean validation MAE across three seeds; main result fixed seed42; test not used in selection','feature_sets':features,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__},'baseline_input_sha256':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ['训练验证测试划分.csv','训练尺度目标值.csv','基线模型评价指标.csv']},'selected':{}}
for task in ['visual_proxy','complete_proxy']:
 choices=grid[grid.task==task].sort_values(['validation_MAE_mean','feature_set','config_id']).iloc[0]
 fs=choices.feature_set;cid=choices.config_id
 assert chosen[(task,fs)]==cid
 test=errs[(errs.task==task)&(errs.feature_set==fs)&(errs.seed==42)&(errs.split=='test')].iloc[0]
 summary['selected'][task]={'feature_set':fs,'config_id':cid,'validation_MAE_seed_mean':float(choices.validation_MAE_mean),'test_MAE':float(test.MAE),'test_RMSE':float(test.RMSE),'test_R2':float(test.R2),'test_n':int(test.n)}
 joblib.dump(bundles[(task,fs,cid,42)],OUT/f'{task}_selected_mlp.joblib')
(OUT/'神经网络实验摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
for i,task in enumerate(['visual_proxy','complete_proxy']):
 sel=summary['selected'][task];fs=sel['feature_set'];cid=sel['config_id']
 h=hist[(hist.task==task)&(hist.feature_set==fs)&(hist.config_id==cid)&(hist.seed==42)]
 a=axes[i,0];a.plot(h.epoch,h.train_MAE,label='训练MAE');a.plot(h.epoch,h.validation_MAE,label='验证MAE');best=bundles[(task,fs,cid,42)]['best_epoch'];a.axvline(best,linestyle='--',color='gray',label=f'保存轮次 {best}');a.legend();a.set(title=f'{task} / {fs} / {cid}',xlabel='训练轮次',ylabel='误差（目标分）')
 c=comparison[(comparison.task==task)&(comparison.feature_set==fs)&(comparison.split=='test')]
 axes[i,1].bar(c.model,c.MAE,color=['#b7b7b7','#6ba18b','#84a4c0','#ba9f7e','#9b84b5']);axes[i,1].set(title=f'相同特征集测试MAE（n={sel["test_n"]}）',ylabel='MAE（越小越好）');axes[i,1].tick_params(axis='x',rotation=15)
fig.savefig(OUT/'神经网络学习曲线与模型对比.png',dpi=170);plt.close(fig)
def md(df):
 df=df.round(3).astype(str)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(x)+' |' for x in df.values)
report='''# 第13周：多层感知机预测实验

沿用环境质量代理目标及基线实验冻结数据。视觉目标训练176、验证55、测试73条道路；完整目标训练26、验证13、测试7条。数据按约500米空间网格分组，同一网格不跨集合，但相邻网格没有空间缓冲。继续排除评分组成项及其变换作为输入。两套特征为道路属性、道路属性加建筑/天空/人物背景。背景与标签同源，可能共享识别误差，预测能力不等于独立环境效度。

## 模型与参数调整

使用scikit-learn MLPRegressor、Adam优化器。预先固定8个配置，比较16/32单层、32–16双层，ReLU/tanh激活，0.001/0.005学习率，0.0001/0.01 L2正则。配置表中只实现指定组合，并非所有参数的完整笛卡尔积。每个配置固定三个种子42/52/62，两任务×两特征集×八配置×三种子，共96次训练。

训练集拟合缺值填补、类别编码、数值缩放；MLP额外对目标做训练集均值/标准差缩放，评价前还原为0–100目标分尺度。目标原值直接读取基线的训练尺度目标，不再重新拟合分位数。输出预测不截断。

每次partial_fit执行一轮，仅使用训练样本更新权重；每轮记录训练/验证MAE及缩放尺度训练损失。外部验证MAE改善超过0.0001时保存模型，连续50轮无改善停止，最多600轮。关闭内部early_stopping以避免另行随机拆出验证数据；取外部验证最佳轮次权重。验证集参与早停和参数选择，验证误差会偏乐观。

配置由三个固定种子的验证MAE平均值选择；主结果预先固定种子42，另外两个种子用于稳定性展示，不按测试表现挑选随机种子。各任务再依据验证均值确定优先特征集。选定配置后才计算测试指标。测试集沿用此前已报告的基线基准，不是新采集的最终盲测集；没有根据本轮测试结果修改参数。

## 配置明细

'''+md(pd.DataFrame(specs,columns=['配置','隐藏层','激活','学习率','L2']))+'\n\n## 验证集选定结果\n\n'
for task,s in summary['selected'].items():
 report+=f'- {task}：{s["feature_set"]}、{s["config_id"]}；测试 n={s["test_n"]}，MAE={s["test_MAE"]:.3f}，RMSE={s["test_RMSE"]:.3f}，R²={s["test_R2"]:.3f}。\n'
report+='\n## 主种子神经网络与既有基线测试对比\n\n'+md(comparison[comparison.split=='test'][['task','feature_set','model','n','MAE','RMSE','R2']])
report+='\n\n## 随机初始化稳定性\n\n'+md(stability)+'\n\n上述标准差只反映三个随机初始化种子的差异，不是对道路样本或空间划分的不确定性置信区间。完整目标只有7条测试道路，不依据其表现确定模型稳定优劣。\n'
for task,s in summary['selected'].items():
 b=config['selected'][task]['test_metrics']['MAE'];delta=100*(s['test_MAE']/b-1)
 report+=f'\n{task}：神经网络验证选定结果相对既有验证选定基线的测试MAE变化 {delta:+.1f}%。两者可能选用不同特征集，公平的同特征对比另见对比表。负值表示误差减少，正值表示误差增加，不预设神经网络必然优于传统模型。\n'
report+='\n## 文件与复现\n\n参数实验记录保存全部96次训练、验证指标、轮次和耗时；逐轮训练过程保存学习曲线；参数方案验证集对比用于审查选择依据；误差表与逐道路预测保存选定配置的三种子结果；joblib模型包括预处理、目标缩放器和神经网络权重。使用项目中的Jupyter Notebook可重跑。\n\n实现参考[scikit-learn MLPRegressor文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)。实际运行版本记录在实验摘要中。\n'
(OUT/'第13周神经网络实验结果.md').write_text(report,encoding='utf-8')
# Persistence and input-label agreement checks.
for task,s in summary['selected'].items():
 fs=s['feature_set'];saved=joblib.load(OUT/f'{task}_{fs}_seed42_mlp.joblib')
 test=labels[(labels.task==task)&(labels.split=='test')].merge(roads,on='road_id',validate='one_to_one')
 expected=pred[(pred.task==task)&(pred.feature_set==fs)&(pred.seed==42)&(pred.split=='test')].set_index('road_id').loc[test.road_id]
 assert np.allclose(predict(saved,test),expected.y_pred)
 assert np.allclose(test.target,expected.y_true)
assert len(trials)==96 and len(errs)==36
print(json.dumps(summary['selected'],ensure_ascii=False),flush=True)
