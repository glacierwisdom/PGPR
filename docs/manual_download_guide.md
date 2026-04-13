# SHS27k数据集手动下载指南

由于自动下载可能受到网络环境限制，本指南将帮助您手动下载和准备SHS27k数据集。

## 1. 数据集介绍

SHS27k是一个常用的蛋白质-蛋白质相互作用(PPI)数据集，包含约27,000个蛋白质相互作用对。该数据集主要用于PPI预测模型的训练和评估。

## 2. 下载方式

SHS27k数据集有多种下载来源，您可以选择其中一种方式进行下载：

### 方式1: 从GitHub仓库下载（推荐）

1. 访问GitHub仓库：[https://github.com/luoyunan/DNN-PPI/tree/master/dataset/yeast/SHS27k](https://github.com/luoyunan/DNN-PPI/tree/master/dataset/yeast/SHS27k)
2. 点击 `SHS27k.csv` 文件
3. 点击 `Download raw file` 按钮下载文件

### 方式2: 从Google Drive下载

1. 访问链接：[https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing](https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing)
2. 点击 `Download` 按钮下载文件

### 方式3: 从Hugging Face下载

1. 访问链接：[https://hf-mirror.com/datasets/Synthyra/SHS27k](https://hf-mirror.com/datasets/Synthyra/SHS27k)
2. 点击 `Download dataset` 按钮下载文件

## 3. 文件放置位置

将下载的文件保存到项目指定目录：

```
E:\science\protein_work\project\GAPNPPI\data\raw\shs27k\
```

- 如果下载的是 `SHS27k.csv`，直接保存到上述目录
- 如果下载的是压缩文件（如 `SHS27k.zip`），解压后将 `SHS27k.csv` 文件放到上述目录

## 4. 数据集验证

下载完成后，您可以运行以下脚本来验证数据集是否正确放置：

```bash
python scripts/verify_shs27k.py
```

## 5. 数据集预处理

运行以下命令对数据集进行预处理：

```bash
python -m data.downloader
```

或者直接运行预处理脚本：

```bash
python scripts/preprocess_shs27k.py
```

## 6. 数据划分

预处理完成后，运行数据划分脚本：

```bash
python -m data.splitter --dataset_name shs27k
```

## 7. 常见问题

### Q: 下载的文件格式是什么？
A: SHS27k数据集的标准格式是CSV文件，包含三列：protein1, protein2, interaction

### Q: 如果下载的文件名不同怎么办？
A: 请将文件重命名为 `SHS27k.csv` 后再放置到指定目录

### Q: 如何检查数据集是否完整？
A: 原始SHS27k数据集应包含约27,000个样本。您可以使用以下命令查看：

```bash
python -c "
import pandas as pd
import os
file_path = 'data/raw/shs27k/SHS27k.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path, header=None)
    print(f'数据集样本数: {len(df)}')
    print(f'数据集形状: {df.shape}')
else:
    print('文件不存在')
"
```

### Q: 预处理失败怎么办？
A: 请确保：
1. 文件路径正确
2. 文件格式正确（CSV格式，三列）
3. 有足够的磁盘空间
4. 已安装必要的依赖（pandas等）

## 8. 自动化脚本

项目提供了一个完整的自动化脚本，您可以在手动下载完成后运行：

```bash
python scripts/setup_data.py
```

该脚本将自动完成数据集预处理、划分等步骤。

---

如果您在操作过程中遇到任何问题，请查看项目的其他文档或提交issue。
