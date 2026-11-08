# 街道环境质量代理基线模型

先阅读[基线模型结果与评价说明](基线模型结果与评价说明.md)。主任务训练176、验证55、测试73条道路；完整目标补充任务训练26、验证13、测试7条道路。

- [visual_proxy_selected_model.joblib](01_模型/visual_proxy_selected_model.joblib)
- [complete_proxy_selected_model.joblib](01_模型/complete_proxy_selected_model.joblib)
- [基线模型评价指标.csv](基线模型评价指标.csv)
- [各模型逐道路预测.csv](03_数据划分与预测/各模型逐道路预测.csv)
- [训练尺度目标值.csv](03_数据划分与预测/训练尺度目标值.csv)
- [训练验证测试划分.csv](03_数据划分与预测/训练验证测试划分.csv)
- [模型参数与结果摘要.json](04_参数与核验/模型参数与结果摘要.json)
- [基线模型评价与预测.png](基线模型评价与预测.png)
- [基线模型结果与评价说明.md](基线模型结果与评价说明.md)
- [验证记录.json](04_参数与核验/验证记录.json)
- [train_environment_baselines.py](02_代码与Notebook/train_environment_baselines.py)
- [澳门街道环境代理基线预测.ipynb](02_代码与Notebook/澳门街道环境代理基线预测.ipynb)

Notebook 已在当前项目的 Anaconda/Jupyter 环境执行。模型与脚本依赖当前项目数据；模型文件含训练集拟合的预处理流程和目标标准化参数。

## 文件位置


### 01_模型

- [complete_proxy_selected_model.joblib](01_模型/complete_proxy_selected_model.joblib)
- [visual_proxy_selected_model.joblib](01_模型/visual_proxy_selected_model.joblib)

### 02_代码与Notebook

- [train_environment_baselines.py](02_代码与Notebook/train_environment_baselines.py)
- [澳门街道环境代理基线预测.ipynb](02_代码与Notebook/澳门街道环境代理基线预测.ipynb)

### 03_数据划分与预测

- [各模型逐道路预测.csv](03_数据划分与预测/各模型逐道路预测.csv)
- [训练尺度目标值.csv](03_数据划分与预测/训练尺度目标值.csv)
- [训练验证测试划分.csv](03_数据划分与预测/训练验证测试划分.csv)

### 04_参数与核验

- [模型参数与结果摘要.json](04_参数与核验/模型参数与结果摘要.json)
- [验证记录.json](04_参数与核验/验证记录.json)
