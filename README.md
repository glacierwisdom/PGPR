# PGPR: Policy-Guided Path Reasoning for Protein-Protein Interaction Prediction

## 1. 项目简介 (Project Overview)
PGPR（Policy-Guided Path Reasoning）是一个面向多标签 PPI（Protein-Protein Interaction）预测的“路径推理”框架。核心目标是：在测试蛋白大量“未见/不连通（unseen/disconnected）”的更严格划分场景下，仍然能从训练图中抽取支持性证据路径，并利用 LLM 进行可解释的关系预测。

## 2. 核心特性 (Key Features)
- **训练图证据路径**：推理路径只在训练集构建的 PPI 图上展开，保证路径中间节点均为训练可见蛋白，路径证据“训练支撑（training-supported）”。
- **未见蛋白代理映射**：对测试端点先做序列相似度映射 $m(\cdot)$，将任意蛋白映射到训练图中的非孤立节点集合（优先 BLASTP，相似度不可用时回退到 ESM embedding 余弦相似度）。
- **PPO 策略引导检索**：将“下一跳邻居选择”建模为 RL 策略 $\pi_\\theta$，用 PPO 结合奖励塑形（准确性、长度惩罚、多样性、距离、边置信度等）学习更有用的路径分布。
- **LLM 条件预测**：将多条路径进行结构化 verbalization（hop 级 relation 语句 + 可选置信度），与查询蛋白的功能文本一起组成 prompt，进行 7 类多标签关系预测（可配合 LoRA/分类头）。
- **严格划分评估**：支持 standard / new\_protein / zero\_shot 等评估模式，并可按 BS/ES/NS（两端可见/单端未见/双端未见）分层统计指标。

## 3. 项目结构 (Project Structure)
*   `configs/`: 配置文件目录，包含模型 (`model.yaml`)、训练 (`training.yaml`) 和数据 (`data.yaml`) 的配置。
*   `data/`: 数据与预处理（SHS27k/SHS148k 及其 split）。
*   `graph/`: 训练图构建与路径工具（`builder.py` 等）。
*   `models/`: 预测器与策略相关模块（路径生成/向量化、组件组装等）。
*   `llm/`: prompt 设计与 LLM 包装（`prompt_designer.py`, `wrapper.py`）。
*   `training/`: 训练流程（监督训练 + PPO 更新、回调与奖励等）。
*   `evaluation/`: 评估与结果导出（`evaluator.py`、可视化等）。

## 4. 数据集 (Dataset)
默认适配 **SHS27k**（并包含 SHS148k 的实验配置/数据目录）。
*   **标签空间**：7 个交互类别（如激活、抑制、结合等），为多标签 multi-hot 预测。
*   **划分**：支持不同 split 策略（bfs/dfs/random），并强调 unseen protein 的 ES/NS 场景。

## 5. 快速开始 (Quick Start)

### 环境要求
*   Python 3.8+
*   PyTorch
*   PyTorch Geometric
*   Transformers (Hugging Face)
*   BitsAndBytes (用于量化)

### 运行训练
1.  **配置**: 修改 `configs/training.yaml` 以适应你的硬件环境（如设置 `device: "cuda"`）。
2.  **启动**:
    ```bash
    python main.py train --config configs/training.yaml
    ```
    *注：可以在配置文件中开启 `quick_mode: true` 进行快速功能验证。*

## 6. 原理解析 (Principles)
PGPR 的工作流程可概括为三步：
1.  **训练图构建**：从训练集交互构建训练图 $G_{train}$，节点为训练可见蛋白，边为训练交互（可带边置信度特征）。
2.  **代理端点 + 路径生成**：对查询 $(A,B)$ 先映射到 $(A',B')=(m(A),m(B))$，再用策略网络在训练图上采样多条证据路径。
3.  **路径 verbalization + LLM 预测**：将路径转成结构化文本（hop 级 relation），与蛋白功能文本一起构造 prompt，输出 7 类多标签预测。

    *提示词示例*:
    > "分析以下蛋白质相互作用路径，预测 {target_protein} 与 {source_protein} 之间的相互作用关系..."

## 7. 许可证 (License)
MIT License
