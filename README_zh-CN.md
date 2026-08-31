# StrokeFusion：置信度引导的混合分割方法

[English](README.md) | **简体中文**

<p align="center">
  <strong>ISLES'26 参赛算法的官方实现</strong><br>
  基于原生空间 T1 加权 MRI 的缺血性卒中病灶分割
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg" alt="PyTorch 2.13"></a>
  <a href="https://github.com/MIC-DKFZ/nnUNet"><img src="https://img.shields.io/badge/nnU--Net-2.8.1-4B8BBE.svg" alt="nnU-Net 2.8.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <a href="https://grand-challenge.org/algorithms/strokefusion-confidence-guided-hybrid-segmentation/"><img src="https://img.shields.io/badge/Grand%20Challenge-StrokeFusion-5A4FCF.svg" alt="Grand Challenge algorithm"></a>
</p>

StrokeFusion 是由残差编码器 3D U-Net 与局部细化 Primus-M Transformer
组成的置信度引导集成方法。两个成员均采用公开的 OpenMind
掩码自编码预训练权重初始化，并在官方 ISLES'26 RAW/原生空间数据上训练。
推理时对前景概率进行等权平均；保守的置信度清理只作用于二值掩码，提交的
概率图保持不变。

> 本仓库提供参赛算法完整的训练、评估、推理和 Grand Challenge 打包代码。
> 受许可约束的数据、预训练权重、最终权重和容器归档不会存入 Git。

## 方法

| 组件 | 冻结配置 |
|---|---|
| **ResEnc-L RASS SWA** | nnU-Net v2 ResEnc-L，`160 × 192 × 160` patch，OpenMind-MAE 初始化，仅训练阶段使用 RASS，对第 300/350/400 轮权重求平均 |
| **Primus-M Local-Refinement SWA** | Primus-M，`160³` patch，OpenMind-MAE 初始化，低秩局部 patch-stem 细化、token 存在性辅助损失、仅训练阶段使用 RASS，对第 200/250/300 轮权重求平均 |
| **概率融合** | 使用镜像测试时增强，对两个模型的前景概率等权平均 |
| **置信度引导输出** | 概率图原样提交；以 `0.5` 二值化，仅当 6 连通分量体积 `< 0.002 mL` 且其最大前景概率 `< 0.65` 时移除该分量 |

Transformer 细化模块以零初始化方式工作在原生 Primus token 网格上，新增一条
含深度可分离 `3 × 3 × 3` 卷积的 `864 → 64 → 864` 局部残差路径。
病灶存在性头对该路径进行门控，并在标准 Dice 与交叉熵分割损失之外加入权重
为 `0.05` 的平衡 token 级二元交叉熵目标。

为兼容已训练权重，部分 trainer 类名和 checkpoint 目录仍保留历史内部标识；
这些只是实现标识，并非公开模型名称。

## 仓库结构

```text
configs/                 冻结的模型与训练配置
docker/                  Grand Challenge 推理容器
docs/                    方法、评估、数据与提交说明
external_models/         公开 OpenMind 预训练权重的预期位置
nnunet_extensions/       StrokeFusion 使用的自定义 nnU-Net trainer
plans/                    冻结的 nnU-Net 架构 plans
reproducibility/         冻结的推理元数据
results/                 精简的 Preliminary Evaluation 结果
scripts/                 数据准备、训练、评估与打包脚本
```

## 复现最终方法

### 1. 系统要求

- Linux x86-64
- Conda 或 Miniforge
- Python 3.11
- 支持 CUDA 的 NVIDIA GPU；训练建议至少 24 GB 显存
- 容器测试需要 Docker 与 NVIDIA Container Toolkit
- 足够存放受许可 RAW 数据和 nnU-Net 预处理缓存的磁盘空间

提交的推理容器以 16 GB 显存的 NVIDIA T4 GPU 为目标环境。

### 2. 克隆仓库并创建环境

```bash
git clone git@github.com:KawhiQaQ/ISLES26-StrokeFusion.git
cd ISLES26-StrokeFusion

bash scripts/bootstrap_env.sh isles26
conda activate isles26
```

精简依赖见 `requirements-core.txt`，完整参考环境见 `environment.lock.txt`。

### 3. 准备官方 RAW 数据

下载并解压官方 ISLES'26/ATLAS R3.0 **RAW** 训练数据。不要使用标准化或预处理
版本。将解压目录放置为：

```text
data/raw/ATLAS3_Training_Raw/
```

归档身份和数据清单见 [`docs/dataset.md`](docs/dataset.md)。请勿将数据集解密
密钥写入脚本、Shell 历史、Docker layer 或 Git。

### 4. 下载公开的自监督初始化权重

```bash
hf download MIC-DKFZ/ResEncL-OpenMind-MAE checkpoint_final.pth \
  --local-dir external_models/ResEncL-OpenMind-MAE

hf download MIC-DKFZ/PrimusM-OpenMind-MAE checkpoint_final.pth \
  --local-dir external_models/PrimusM-OpenMind-MAE
```

| 初始化权重 | 本地路径 |
|---|---|
| ResEnc-L OpenMind-MAE | `external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth` |
| Primus-M OpenMind-MAE | `external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth` |

这些权重在 OpenMind/OpenNeuro 脑 MRI 集合上进行无标签预训练，并非来自
ISLES 验证集或测试集。

### 5. 生成冻结划分并预处理

```bash
bash scripts/prepare_data.sh setup
bash scripts/prepare_data.sh plan
bash scripts/prepare_data.sh preprocess
```

`setup` 会根据受许可数据确定性重建按中心分组的五折划分。生成的 manifest、
病例级划分文件和预处理缓存均保留在本地，并由 Git 忽略。

### 6. 在全量数据上训练两个成员

```bash
bash scripts/train_strokefusion.sh
```

脚本会执行完整的冻结流程：

1. 在 fold `all` 上训练 ResEnc-L RASS 400 轮；
2. 平均第 300、350、400 轮权重，得到 ResEnc-L RASS SWA；
3. 在 fold `all` 上训练 Primus-M Local-Refinement 400 轮；
4. 平均第 200、250、300 轮权重，得到 Primus-M Local-Refinement SWA；
5. 将两个验证后的模型保存到 `outputs/final_full_models/`。

若存在 `checkpoint_latest.pth`，训练会安全续跑。

## 权重下载

| 文件 | 下载链接 | 提取码 |
|---|---|---|
| 完整 StrokeFusion 模型包（两个 SWA checkpoint、plans、metadata 及 Grand Challenge Model resource） | [百度网盘](https://pan.baidu.com/s/1U1IR3pC7o73z7Xe1IdFDdw?pwd=z6hx) | `z6hx` |

下载 `model-postprocess.tar.gz` 后，将其放到
`submission_artifacts/model-postprocess.tar.gz`，然后恢复两个 checkpoint：

```bash
bash scripts/restore_final_models.sh
```

下载全部受许可和公开文件后，可验证冻结版本：

```bash
bash scripts/verify_final_release.sh
```

## Grand Challenge 推理容器

容器遵循官方 ISLES'26 `invoke` 接口，输入原生空间 T1 MRI 与卒中元数据，输出
原始几何下的二值病灶掩码和浮点概率图。元数据仅用于满足接口要求，模型本身
不使用这些字段。

```bash
cd docker
bash do_build.sh
bash do_test_run.sh
bash do_save.sh
```

输入输出接口、资源限制和打包约定见 [`docker/README.md`](docker/README.md)
与 [`docs/submission.md`](docs/submission.md)。

## 评估

评估代码实现 Dice、病灶 F1、PR-AUC、绝对体积差和绝对病灶数量差：

```bash
python scripts/evaluate_isles26.py --help
python scripts/evaluate_probability_ensemble.py --help
```

两个公开 Preliminary 病例的结果记录在
[`results/preliminary_metrics.json`](results/preliminary_metrics.json)。它们仅用于
检查容器合规和推理一致性，不用于选模。由受许可训练数据生成的本地模型比较池
和逐病例预测不会公开。

## 可复现性说明

- 监督训练只使用官方 RAW 数据。
- CENTER、DAYS_POST_STROKE 与 CHRONICITY 均不作为模型输入。
- RASS 只在训练时启用，不改变推理图。
- 两个成员的融合权重、阈值和清理条件均已冻结。
- 二值掩码后处理不会修改概率图。
- 文件验证与恢复说明见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 参考文献

- Wald et al., [An OpenMind for 3D Medical Vision Self-Supervised Learning](https://arxiv.org/abs/2412.17041), 2024.
- Wald et al., [Revisiting MAE Pre-training for 3D Medical Image Segmentation](https://arxiv.org/abs/2410.23132), 2024.
- Wald et al., [Primus: Enforcing Attention Usage for 3D Medical Image Segmentation](https://arxiv.org/abs/2503.01835), 2025.
- Isensee et al., [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/abs/2404.09556), 2024.

ISLES'26 挑战赛论文发表后将补充正式引用信息。

## 许可证

本仓库采用 [Apache License 2.0](LICENSE) 开源。ISLES'26 数据集、预训练权重和
训练权重仍分别受其原始许可证与使用条款约束。
