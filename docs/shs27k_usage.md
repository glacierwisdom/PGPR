# SHS27k数据集使用指南

## 1. 数据集概述

SHS27k是一个较小规模的蛋白质-蛋白质相互作用(PPI)数据集，包含约27,000个蛋白质-蛋白质相互作用样本。该数据集主要用于PPI预测模型的训练和评估，特别适合快速原型开发和小规模实验。

### 数据集特点
- **规模**: 约27,000个PPI样本
- **蛋白质数量**: 约27,000个独特蛋白质
- **关系类型**: 二分类（相互作用/不相互作用）
- **来源**: 酵母等模式生物的PPI数据

## 2. 数据集下载与准备

### 2.1 使用数据准备脚本

项目提供了Windows环境下的数据准备脚本，可以一键下载和预处理SHS27k数据集：

```powershell
# 进入项目根目录
cd E:\science\protein_work\project\GAPNPPI

# 运行数据准备脚本
.scriptssetup_data.ps1
```

该脚本将：
1. 创建必要的目录结构
2. 从GitHub下载原始SHS27k数据集
3. 运行数据预处理流程
4. 划分训练集、验证集和测试集

### 2.2 手动下载与处理

如果需要手动处理数据集，可以按照以下步骤操作：

#### 2.2.1 下载原始数据集

```powershell
# 创建原始数据目录
mkdir -p data/raw/shs27k

# 下载SHS27k数据集
Invoke-WebRequest -Uri "https://github.com/luoyunan/DNN-PPI/raw/master/dataset/yeast/SHS27k/SHS27k.csv" -OutFile "data/raw/shs27k/SHS27k.csv"
```

#### 2.2.2 运行预处理

```powershell
# 运行数据下载器进行预处理
python -m data.downloader --dataset shs27k --preprocess
```

#### 2.2.3 划分数据集

```powershell
# 使用默认配置划分数据集
python -m data.splitter --config configs/data.yaml --dataset shs27k
```

## 3. 数据集配置

SHS27k数据集的配置位于 `configs/data.yaml` 文件中。以下是关键配置参数：

```yaml
# 数据集配置
dataset:
  name: "shs27k"  # 数据集名称
  train_file: "shs27k_train.tsv"  # 训练集文件名
  val_file: "shs27k_val.tsv"  # 验证集文件名
  test_file: "shs27k_test.tsv"  # 测试集文件名
  
  # SHS27k 数据集特定配置
  shs27k:
    url: "https://github.com/luoyunan/DNN-PPI/raw/master/dataset/yeast/SHS27k"
    raw_file: "SHS27k.csv"
    processed_file: "shs27k_processed.tsv"
    download_timeout: 30
```

### 3.1 优化的预处理参数

针对SHS27k数据集的特点，配置文件中包含了优化的预处理参数：

```yaml
preprocessing:
  sequence:
    max_length: 512  # 较短的最大序列长度，适合SHS27k
    tokenizer_name: "facebook/esm2_t6_8M_UR50D"  # 较小的ESM模型，适合小规模数据集
  
  graph:
    num_neighbors: 8  # 减少邻居数量，提高效率
    max_path_length: 4  # 减少路径长度，节省计算资源
  
  feature_extraction:
    esm_model: "facebook/esm2_t6_8M_UR50D"  # 适合小规模数据集的模型
    use_go_features: false  # SHS27k可能没有GO注释
```

## 4. 数据格式

### 4.1 原始数据格式

SHS27k原始数据格式（CSV）：
```csv
protein1,protein2,interaction
P1,P2,1
P3,P4,0
...
```

### 4.2 处理后数据格式

处理后的SHS27k数据格式（TSV）：
```tsv
protein_A_id	protein_B_id	sequence_A	sequence_B	relationship_label
P1	P2	M...	A...	1
P3	P4	V...	L...	0
...
```

## 5. 使用示例

### 5.1 数据加载示例

```python
from data.dataset import PPIDataset
from torch.utils.data import DataLoader

# 加载SHS27k训练集
train_dataset = PPIDataset(
    data_path="data/processed/shs27k_train.tsv",
    max_sequence_length=512
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# 遍历数据
for batch in train_loader:
    protein_a = batch["protein_a"]
    protein_b = batch["protein_b"]
    labels = batch["label"]
    # 执行模型训练...
```

### 5.2 模型训练示例

```python
import torch
from models.gapn_ppi import GAPNPPI
from training.trainer import PPITrainer
from configs.config_loader import load_config

# 加载配置
config = load_config("configs/data.yaml")

# 初始化模型
model = GAPNPPI(
    config["model"]["esm"]["model_name"],
    config["model"]["gnn"]["num_layers"],
    config["model"]["gnn"]["hidden_dim"],
    config["model"]["gnn"]["dropout"]
)

# 初始化训练器
trainer = PPITrainer(
    model=model,
    config=config,
    train_data_path="data/processed/shs27k_train.tsv",
    val_data_path="data/processed/shs27k_val.tsv"
)

# 开始训练
trainer.train(
    epochs=10,
    learning_rate=1e-4,
    batch_size=128
)
```

## 6. 评价指标

针对SHS27k数据集的特点，推荐使用以下评价指标：

### 6.1 基本分类指标
- **准确率(Accuracy)**: 分类正确的样本数占总样本数的比例
- **精确率(Precision)**: 预测为正类的样本中实际为正类的比例
- **召回率(Recall)**: 实际为正类的样本中被正确预测的比例
- **F1分数(F1-Score)**: 精确率和召回率的调和平均数

### 6.2 平衡指标
- **马修斯相关系数(MCC)**: 考虑了真阳性、真阴性、假阳性和假阴性的综合指标，特别适合不平衡数据集
- **Cohen's Kappa**: 衡量分类器与随机猜测相比的一致性

### 6.3 性能曲线
- **ROC曲线(Receiver Operating Characteristic)**: 展示不同阈值下的真阳性率和假阳性率
- **AUC-ROC**: ROC曲线下的面积，评估分类器的整体性能
- **PR曲线(Precision-Recall)**: 展示不同阈值下的精确率和召回率
- **AUPRC**: PR曲线下的面积，在不平衡数据集上更有意义

### 6.4 混淆矩阵

混淆矩阵可以直观地展示模型的分类结果：
- **真阳性(TP)**: 实际为正类且被正确预测的样本
- **真阴性(TN)**: 实际为负类且被正确预测的样本
- **假阳性(FP)**: 实际为负类但被错误预测为正类的样本
- **假阴性(FN)**: 实际为正类但被错误预测为负类的样本

## 7. 实验建议

### 7.1 模型选择

由于SHS27k数据集规模较小，建议使用：
- 较小的ESM模型（如 `facebook/esm2_t6_8M_UR50D`）
- 较浅的图神经网络层数（3-5层）
- 较小的隐藏层维度（128-256）

### 7.2 训练设置
- **Batch Size**: 128-256（根据GPU内存调整）
- **学习率**: 1e-4到5e-4
- **训练轮数**: 10-20轮（避免过拟合）
- **正则化**: 使用适当的 dropout（0.1-0.3）和权重衰减

### 7.3 避免过拟合的策略
- 数据增强
- 早停策略
- 模型正则化
- 交叉验证

## 8. 常见问题

### 8.1 数据集下载失败

如果遇到数据集下载失败，可以尝试：
1. 检查网络连接
2. 手动下载数据集并放置在 `data/raw/shs27k/` 目录
3. 检查防火墙设置

### 8.2 序列信息缺失

SHS27k原始数据集可能不包含蛋白质序列信息。项目的预处理流程会自动生成模拟序列或尝试从UniProt获取实际序列。如果需要更准确的序列信息，可以：

```python
from data.uniprot_fetcher import UniProtFetcher

# 创建UniProt获取器
with UniProtFetcher() as fetcher:
    # 获取单个蛋白质序列
    sequence = fetcher.fetch_sequence("P00509")
    
    # 批量获取蛋白质序列
    sequences = fetcher.fetch_sequences(["P00509", "P00510"])
```

### 8.3 内存不足

如果在处理过程中遇到内存不足问题，可以：
1. 减少批次大小
2. 使用较小的ESM模型
3. 减少序列最大长度
4. 增加系统内存或使用更强大的GPU

## 9. 后续工作

- 添加实际的UniProt序列获取功能
- 支持更多数据集格式
- 优化小规模数据集的模型训练策略
- 添加更多针对SHS27k的实验脚本

## 10. 参考文献

- [DNN-PPI: Deep Neural Network for Protein-Protein Interaction Prediction](https://github.com/luoyunan/DNN-PPI)
- [UniProt: Universal Protein Resource](https://www.uniprot.org/)
- [Protein-Protein Interaction Prediction: Methods and Applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6818403/)
