# PGPR: Policy-Guided Path Reasoning for PPI Prediction

本仓库实现 PGPR（Policy-Guided Path Reasoning），用于多标签 PPI 预测，并强调 unseen protein 场景下的“训练图证据路径 + LLM 条件预测”。

## 快速开始

### 训练

```bash
python main.py train --config configs/training.yaml
```

### 评估

```bash
python main.py evaluate --config configs/evaluation.yaml --checkpoint path/to/checkpoint.pt
```

### 预测

```bash
python main.py predict --config configs/inference.yaml --checkpoint path/to/checkpoint.pt --protein-a SEQ_A --protein-b SEQ_B
```

### 推理服务

```bash
python main.py serve --config configs/inference.yaml --checkpoint path/to/checkpoint.pt
```

## 论文对应

论文源码在 [paper_emnlp](file:///e:/science/protein_work/PGPR_emnlp/paper_emnlp)（主 tex 为 [acl_latex.tex](file:///e:/science/protein_work/PGPR_emnlp/paper_emnlp/latex/acl_latex.tex)）。

## 入口与结构

- 主入口：[main.py](file:///e:/science/protein_work/PGPR_emnlp/main.py)
- 训练/评估脚本：`experiments/`
- 配置：`configs/`
- 数据与图：`data/`, `graph/`
- LLM 组件：`llm/`
- 训练与奖励：`training/`
