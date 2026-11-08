# 第13周神经网络预测实验

已完成96次MLP训练；沿用基线冻结划分与训练尺度目标。神经网络主种子固定42，配置按三个种子的平均验证MAE选择。

建议先看：

- [实验结果说明](第13周神经网络实验结果.md)
- [神经网络与基线对比表](神经网络与基线对比表.csv)
- [神经网络误差结果](03_实验数据与记录/神经网络误差结果.csv)
- [学习曲线与模型对比图](神经网络学习曲线与模型对比.png)
- [全部96次参数记录](03_实验数据与记录/神经网络参数实验记录.csv)
- [逐轮训练过程](03_实验数据与记录/逐轮训练过程.csv)
- [已运行Notebook](05_代码与Notebook/第13周_街道环境神经网络预测实验.ipynb)

主分析模型为 `visual_proxy_selected_mlp.joblib`；完整代理目标小样本补充模型为 `complete_proxy_selected_mlp.joblib`。模型包含训练集拟合的预处理器、目标缩放器与MLP网络。其余带特征集和seed的模型用于种子稳定性复核。

主分析测试MAE 11.12，线性回归10.56；神经网络未在MAE上超过线性基线。RMSE和R²略好，需结合误差指标共同解释。完整代理目标测试仅7条道路。

Notebook 在当前 E:/毕业设计 项目的Jupyter内核中运行；RUN_TRAINING=True会重新执行96次训练，设为False可直接读取已有结果。


## 文件位置


### 01_选定模型

- [complete_proxy_selected_mlp.joblib](01_选定模型/complete_proxy_selected_mlp.joblib)
- [visual_proxy_selected_mlp.joblib](01_选定模型/visual_proxy_selected_mlp.joblib)

### 02_随机种子模型

- [complete_proxy_road_context_seed42_mlp.joblib](02_随机种子模型/complete_proxy_road_context_seed42_mlp.joblib)
- [complete_proxy_road_context_seed52_mlp.joblib](02_随机种子模型/complete_proxy_road_context_seed52_mlp.joblib)
- [complete_proxy_road_context_seed62_mlp.joblib](02_随机种子模型/complete_proxy_road_context_seed62_mlp.joblib)
- [complete_proxy_road_only_seed42_mlp.joblib](02_随机种子模型/complete_proxy_road_only_seed42_mlp.joblib)
- [complete_proxy_road_only_seed52_mlp.joblib](02_随机种子模型/complete_proxy_road_only_seed52_mlp.joblib)
- [complete_proxy_road_only_seed62_mlp.joblib](02_随机种子模型/complete_proxy_road_only_seed62_mlp.joblib)
- [visual_proxy_road_context_seed42_mlp.joblib](02_随机种子模型/visual_proxy_road_context_seed42_mlp.joblib)
- [visual_proxy_road_context_seed52_mlp.joblib](02_随机种子模型/visual_proxy_road_context_seed52_mlp.joblib)
- [visual_proxy_road_context_seed62_mlp.joblib](02_随机种子模型/visual_proxy_road_context_seed62_mlp.joblib)
- [visual_proxy_road_only_seed42_mlp.joblib](02_随机种子模型/visual_proxy_road_only_seed42_mlp.joblib)
- [visual_proxy_road_only_seed52_mlp.joblib](02_随机种子模型/visual_proxy_road_only_seed52_mlp.joblib)
- [visual_proxy_road_only_seed62_mlp.joblib](02_随机种子模型/visual_proxy_road_only_seed62_mlp.joblib)

### 03_实验数据与记录

- [参数方案验证集对比.csv](03_实验数据与记录/参数方案验证集对比.csv)
- [神经网络参数实验记录.csv](03_实验数据与记录/神经网络参数实验记录.csv)
- [神经网络误差结果.csv](03_实验数据与记录/神经网络误差结果.csv)
- [神经网络逐道路预测.csv](03_实验数据与记录/神经网络逐道路预测.csv)
- [逐轮训练过程.csv](03_实验数据与记录/逐轮训练过程.csv)
- [随机种子稳定性.csv](03_实验数据与记录/随机种子稳定性.csv)

### 04_参数与核验

- [实验核验记录.json](04_参数与核验/实验核验记录.json)
- [实验进度.jsonl](04_参数与核验/实验进度.jsonl)
- [神经网络实验摘要.json](04_参数与核验/神经网络实验摘要.json)

### 05_代码与Notebook

- [train_environment_mlp.py](05_代码与Notebook/train_environment_mlp.py)
- [第13周_街道环境神经网络预测实验.ipynb](05_代码与Notebook/第13周_街道环境神经网络预测实验.ipynb)
