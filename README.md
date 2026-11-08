# 澳门街道环境评价

Urban Street Perception — Macao street environment analysis and prediction.

**状态：开发中。当前为第一版（V1.0）阶段成果，项目尚未完成。** 进度更新：2026-11-09。

本项目使用澳门道路属性与街景图像，提取视觉特征、构建街道环境质量代理指标，并比较传统回归与多层感知机（MLP）的预测表现。配套网页展示道路地图、街景、指标对比和已有实验结果。

当前分数由研究中明确设定的指标公式构成，尚未通过独立居民感知或人工环境评分验证，不能直接视为幸福度、真实交通流量或完整环境质量。

## 已完成的工作

| 方向 | 第一版进度 |
| --- | --- |
| 数据整理 | 整理道路、采样点、图片索引及图片与道路关联 |
| 街景特征 | SegFormer语义分割与YOLO目标检测；5,920条图片记录，5,537张不同内容图片 |
| 指标构建 | 可见绿化、可见机动车和标注限速组成的代理指标；附权重与质量敏感性分析 |
| 街道分析 | 区域、道路等级和结构类型的统计比较与可视化 |
| 基线模型 | 均值预测、线性回归、随机森林、梯度提升 |
| 神经网络 | MLP的8套配置、3个初始化种子、两任务与两特征集，共96次训练 |
| Web展示 | 地图、道路详情、街景图片、道路对比、预测记录、误差指标及CSV导出；支持离线访问 |

## 数据范围与实验结果

网页包含1,537条道路；其中视觉代理指标覆盖304条，完整代理指标覆盖46条。视觉目标的训练/验证/测试样本数为176/55/73，完整目标为26/13/7。

使用约500米网格分组划分，标签标准化、特征填补和数值缩放仅在训练集拟合。评分组成项及其确定性变换不直接作为预测输入。当前没有空间缓冲带，邻近网格仍可能存在空间关联。

主分析：视觉目标、道路属性加视觉背景特征，固定测试集73条道路：

| 模型 | MAE ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: |
| 训练均值预测 | 14.785 | 18.906 | -0.006 |
| 线性回归 | 10.563 | 13.933 | 0.454 |
| MLP（主种子42） | 11.117 | 13.713 | 0.471 |

模型及参数按验证集表现选择。MLP的MAE尚未优于线性回归，不能据此宣称神经网络全面更好。三种初始化种子的差异不等于空间样本不确定性；完整目标测试集仅7条，结果仅作探索性补充。

## 本地查看网页

需要Python 3.10或更新版本。准备网页仅使用Python标准库，不需要Node.js、数据库、云服务器或VPN。

```bash
git clone https://github.com/Ted1201/urban-street-perception.git
cd urban-street-perception
python prepare_web.py
```

然后用浏览器打开 `web-prototype/dist/macau-streets.html`。Windows也可以双击根目录的“离线打开网站.cmd”。

`prepare_web.py`会校验并解压 `assets/thumbnails/` 中的5,537张展示缩略图，再生成内嵌数据的离线页面。首次准备需要额外磁盘空间；下载仓库时需保留所有缩略图分包。

如需HTTP方式：

```bash
python web-prototype/local_server.py
```

浏览器访问 `http://127.0.0.1:8765`。该地址仅供本机使用。网页预测页展示已完成实验的记录，不会实时训练或调用模型。

## 代码与结果目录

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 图像特征提取、环境指标构建、街道比较和模型训练 |
| `notebooks/` | 交互分析与复现实验Notebook |
| `outputs/` | 已整理的派生数据、参数、图表、划分、模型与预测结果 |
| `成果汇总_第一版/` | 第一版报告、数据表、图表与模型归档的GitHub发布子集 |
| `web-prototype/` | 网页源码、数据导出、数据核验及本地服务 |
| `assets/thumbnails/` | 可还原的网页展示缩略图分包及SHA-256清单 |
| `self-hosting/` | 可选的Nginx/Docker部署配置 |
| [后续任务表](后续任务表.md) | 第二版任务、优先级、交付物、完成标准及依赖 |

本仓库未上传原始高分辨率照片、原始RAR数据包、外部预训练权重、临时日志、工作站快捷方式及重复部署压缩包。发布目录与本地完整研究目录存在这些明确差异。

## 模型复现

当前训练结果记录的环境为Python 3.12.4、scikit-learn 1.7.2。建议使用独立环境安装分析依赖：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

使用仓库内派生数据可重新计算指标、比较道路与运行表格模型：

```bash
python scripts/build_environment_indicators.py
python scripts/analyze_street_environment_comparison.py
python scripts/train_environment_baselines.py
python scripts/train_environment_mlp.py
```

训练会改写相应 `outputs/` 文件，重跑前请另存已有结果。MLP读取基线生成的划分及训练尺度目标，应先完成基线步骤。图表字体可能因系统不同而变化；部分Notebook仍保留原工作站路径，需要调整后运行。

从原始照片重新提取视觉特征还需自行准备原始照片、PyTorch/torchvision、Transformers、Ultralytics及预训练权重，并检查 `scripts/extract_street_view_features_full.py` 的路径和设备设置。此部分尚未整理为一键安装流程，根目录依赖文件也不包含完整图像推理环境。

更新网页数据：

```bash
python web-prototype/export_data.py
python prepare_web.py
python web-prototype/verify_export.py
```

导出器重新生成全部缩略图时需要原始图片。仅使用已发布网页无需重跑导出器；数据或样本数量变化后，应同步修改核验脚本中的版本断言。

## 已知不足与后续工作

- 补充独立人工环境评分，以及图像检测、分割的定量抽查。
- 解释各区域覆盖率、缺失原因和样本筛选过程。
- 补充空间分组稳定性、缓冲区实验、特征消融与误差案例。
- 修正参数报告将0.0001显示为0.0的精度问题，统一配置与依赖管理。
- 完善Web研究说明、质量展示和模型解释；真实推理接口尚未实现。
- 根据需求再选择在线托管或带后台的服务器，保留离线演示方式。

完整待办见[后续任务表](后续任务表.md)。本仓库用于记录持续开发的毕业设计进度，不代表最终论文或已经验证可用于实际决策的系统。
