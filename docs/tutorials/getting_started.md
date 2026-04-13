# PGPR 入门教程

本教程面向本仓库当前代码（以 [main.py](file:///e:/science/protein_work/PGPR_emnlp/main.py) 作为统一入口），给出可直接运行的训练/评估/推理命令。

## 1. 配置

核心配置位于 [configs](file:///e:/science/protein_work/PGPR_emnlp/configs)。

## 2. 训练

```bash
python main.py train --config configs/training.yaml
```

## 3. 评估

```bash
python main.py evaluate --config configs/evaluation.yaml --checkpoint path/to/checkpoint.pt
```

## 4. 单次预测

```bash
python main.py predict --config configs/inference.yaml --checkpoint path/to/checkpoint.pt --protein-a SEQ_A --protein-b SEQ_B
```

## 5. 推理服务

```bash
python main.py serve --config configs/inference.yaml --checkpoint path/to/checkpoint.pt
```
