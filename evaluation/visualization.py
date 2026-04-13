import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
import time
import json
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PPIResultVisualizer:
    """
    PPI结果可视化工具类
    用于可视化PPI预测结果、图结构和评估指标
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化工具
        
        Args:
            config (Dict[str, Any]): 配置字典
        """
        self.config = config
        self.relation_type_mapping = {
            0: "激活作用",
            1: "抑制作用",
            2: "结合作用",
            3: "催化作用",
            4: "表达调控",
            5: "物理相互作用",
            6: "遗传相互作用",
            7: "具有序列相似性"
        }
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info("PPIResultVisualizer初始化完成")
    
    def plot_evaluation_metrics(self, 
                              results: Dict[str, float],
                              save_path: Optional[str] = None,
                              show: bool = True
                              ):
        """
        绘制评估指标图
        
        Args:
            results (Dict[str, float]): 评估结果
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制评估指标图...")
        
        # 准备数据
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        values = [results[metric] for metric in metrics]
        
        # 设置颜色
        colors = sns.color_palette('pastel')[0:len(metrics)]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 设置图表属性
        plt.title('PPI预测评估指标', fontsize=14, fontweight='bold')
        plt.xlabel('指标名称', fontsize=12)
        plt.ylabel('指标值', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"评估指标图已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             save_path: Optional[str] = None,
                             show: bool = True
                             ):
        """
        绘制混淆矩阵
        
        Args:
            cm (np.ndarray): 混淆矩阵
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制混淆矩阵...")
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.relation_type_mapping[i] for i in range(cm.shape[1])],
            yticklabels=[self.relation_type_mapping[i] for i in range(cm.shape[0])],
            cbar=True
        )
        
        # 设置图表属性
        plt.title('PPI预测混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def plot_class_metrics(self, 
                         class_report: Dict[str, Any],
                         save_path: Optional[str] = None,
                         show: bool = True
                         ):
        """
        绘制各类别指标
        
        Args:
            class_report (Dict[str, Any]): 类别报告
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制各类别指标...")
        
        # 准备数据
        classes = []
        precision = []
        recall = []
        f1 = []
        support = []
        
        for class_name, metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(self.relation_type_mapping.get(int(class_name), class_name))
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1.append(metrics['f1-score'])
                support.append(metrics['support'])
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 设置x轴位置
        x = np.arange(len(classes))
        width = 0.25
        
        # 绘制柱状图
        plt.bar(x - width, precision, width, label='精确率', color='skyblue')
        plt.bar(x, recall, width, label='召回率', color='lightgreen')
        plt.bar(x + width, f1, width, label='F1分数', color='salmon')
        
        # 添加数值标签
        for i, v in enumerate(precision):
            plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(recall):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(f1):
            plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 设置图表属性
        plt.title('各类别预测指标', fontsize=14, fontweight='bold')
        plt.xlabel('关系类型', fontsize=12)
        plt.ylabel('指标值', fontsize=12)
        plt.xticks(x, classes, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"类别指标图已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def plot_ppi_graph(self, 
                     graph_data: Any,
                     source_node: Optional[int] = None,
                     target_node: Optional[int] = None,
                     paths: Optional[List[List[int]]] = None,
                     save_path: Optional[str] = None,
                     show: bool = True
                     ):
        """
        绘制PPI网络图
        
        Args:
            graph_data (Any): 图数据
            source_node (Optional[int]): 源节点
            target_node (Optional[int]): 目标节点
            paths (Optional[List[List[int]]]): 路径列表
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制PPI网络图...")
        
        # 将PyTorch Geometric图转换为NetworkX图
        if hasattr(graph_data, 'to_networkx'):
            nx_graph = graph_data.to_networkx()
        elif hasattr(graph_data, 'edge_index'):
            nx_graph = self._convert_to_networkx(graph_data)
        else:
            raise ValueError("不支持的图数据格式")
        
        # 创建图表
        plt.figure(figsize=(15, 12))
        
        # 使用Spring布局
        pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            nx_graph, pos, 
            node_size=300, 
            node_color='lightblue',
            alpha=0.8,
            edgecolors='gray'
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            nx_graph, pos,
            width=1.0,
            alpha=0.5,
            edge_color='gray'
        )
        
        # 高亮源节点和目标节点
        if source_node is not None and source_node in nx_graph.nodes():
            nx.draw_networkx_nodes(
                nx_graph, pos, 
                nodelist=[source_node],
                node_size=500,
                node_color='red',
                edgecolors='black'
            )
            plt.text(pos[source_node][0], pos[source_node][1] + 0.05, 
                    f'Source: {source_node}', fontsize=12, fontweight='bold',
                    ha='center', va='center', color='red')
        
        if target_node is not None and target_node in nx_graph.nodes():
            nx.draw_networkx_nodes(
                nx_graph, pos, 
                nodelist=[target_node],
                node_size=500,
                node_color='green',
                edgecolors='black'
            )
            plt.text(pos[target_node][0], pos[target_node][1] + 0.05, 
                    f'Target: {target_node}', fontsize=12, fontweight='bold',
                    ha='center', va='center', color='green')
        
        # 高亮路径
        if paths and len(paths) > 0:
            for i, path in enumerate(paths):
                # 检查路径是否存在
                if all(node in nx_graph.nodes() for node in path):
                    # 创建边列表
                    edges = list(zip(path[:-1], path[1:]))
                    
                    # 高亮路径边
                    nx.draw_networkx_edges(
                        nx_graph, pos,
                        edgelist=edges,
                        width=3.0,
                        alpha=0.7,
                        edge_color=sns.color_palette('Set2')[i % len(sns.color_palette('Set2'))],
                        style='solid'
                    )
                    
                    # 标注路径
                    plt.text(pos[path[0]][0], pos[path[0]][1] + 0.1, 
                            f'Path {i+1}', fontsize=10, fontweight='bold',
                            ha='center', va='center', color='purple')
        
        # 设置图表属性
        plt.title('PPI网络图', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PPI网络图已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def _convert_to_networkx(self, graph_data: Any) -> nx.Graph:
        """
        将图数据转换为NetworkX图
        
        Args:
            graph_data (Any): 图数据
            
        Returns:
            nx.Graph: NetworkX图
        """
        # 从edge_index创建边列表
        edge_list = graph_data.edge_index.t().tolist()
        
        # 创建图
        graph = nx.Graph()
        
        # 添加边
        graph.add_edges_from(edge_list)
        
        return graph
    
    def plot_path_length_distribution(self, 
                                   path_lengths: List[int],
                                   save_path: Optional[str] = None,
                                   show: bool = True
                                   ):
        """
        绘制路径长度分布
        
        Args:
            path_lengths (List[int]): 路径长度列表
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制路径长度分布图...")
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制直方图
        plt.hist(path_lengths, bins=max(path_lengths) - min(path_lengths) + 1,
                alpha=0.7, color='skyblue', edgecolor='black')
        
        # 设置图表属性
        plt.title('COT路径长度分布', fontsize=14, fontweight='bold')
        plt.xlabel('路径长度', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        mean_length = np.mean(path_lengths)
        median_length = np.median(path_lengths)
        plt.text(0.95, 0.95, f'均值: {mean_length:.2f}\n中位数: {median_length:.2f}',
                horizontalalignment='right', verticalalignment='top',
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"路径长度分布图已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def plot_embedding_2d(self, 
                        embeddings: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None,
                        show: bool = True
                        ):
        """
        绘制2D嵌入图
        
        Args:
            embeddings (np.ndarray): 嵌入向量 [N, D]
            labels (Optional[np.ndarray]): 标签 [N]
            save_path (Optional[str]): 保存路径
            show (bool): 是否显示
        """
        logger.info("开始绘制2D嵌入图...")
        
        # 使用TSNE降维
        if embeddings.shape[1] > 2:
            logger.info(f"使用TSNE将嵌入从{embeddings.shape[1]}维降至2维...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 绘制散点图
        if labels is not None:
            # 有标签时使用不同颜色
            unique_labels = np.unique(labels)
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                           label=self.relation_type_mapping.get(int(label), f'Label {label}'),
                           alpha=0.6, s=50)
            plt.legend(fontsize=12)
        else:
            # 无标签时使用单一颜色
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       alpha=0.6, s=50, color='blue')
        
        # 设置图表属性
        plt.title('蛋白质嵌入2D可视化', fontsize=16, fontweight='bold')
        plt.xlabel('维度1', fontsize=12)
        plt.ylabel('维度2', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D嵌入图已保存到：{save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
    
    def create_evaluation_report(self, 
                               results: Dict[str, float],
                               report_path: str,
                               include_plots: bool = True
                               ):
        """
        创建评估报告
        
        Args:
            results (Dict[str, float]): 评估结果
            report_path (str): 报告保存路径
            include_plots (bool): 是否包含图表
        """
        logger.info(f"开始创建评估报告：{report_path}...")
        
        # 创建报告目录
        report_dir = os.path.dirname(report_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建报告内容
        report_content = f"""# PPI预测评估报告

**生成时间**: {timestamp}

## 1. 基本信息
- 评估模式: {results.get('evaluation_mode', 'standard')}
- 评估时间: {results.get('evaluation_time', 0):.2f}秒

## 2. 评估指标
| 指标 | 值 |
|------|-----|
| 准确率 | {results.get('accuracy', 0):.4f} |
| 精确率 | {results.get('precision', 0):.4f} |
| 召回率 | {results.get('recall', 0):.4f} |
| F1分数 | {results.get('f1', 0):.4f} |
| AUC | {results.get('auc', 0):.4f} |
| 损失 | {results.get('loss', 0):.4f} |

## 3. 类别详细指标

"""
        
        # 添加类别报告
        class_report = results.get('class_report', {})
        for class_name, metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                relation_name = self.relation_type_mapping.get(int(class_name), f'未知类别 {class_name}')
                report_content += f"### {relation_name}\n"
                report_content += f"- 精确率: {metrics.get('precision', 0):.4f}\n"
                report_content += f"- 召回率: {metrics.get('recall', 0):.4f}\n"
                report_content += f"- F1分数: {metrics.get('f1-score', 0):.4f}\n"
                report_content += f"- 支持样本数: {metrics.get('support', 0)}\n\n"
        
        # 添加平均值
        if 'macro avg' in class_report:
            report_content += "### 宏平均\n"
            report_content += f"- 精确率: {class_report['macro avg'].get('precision', 0):.4f}\n"
            report_content += f"- 召回率: {class_report['macro avg'].get('recall', 0):.4f}\n"
            report_content += f"- F1分数: {class_report['macro avg'].get('f1-score', 0):.4f}\n\n"
        
        if 'weighted avg' in class_report:
            report_content += "### 加权平均\n"
            report_content += f"- 精确率: {class_report['weighted avg'].get('precision', 0):.4f}\n"
            report_content += f"- 召回率: {class_report['weighted avg'].get('recall', 0):.4f}\n"
            report_content += f"- F1分数: {class_report['weighted avg'].get('f1-score', 0):.4f}\n\n"
        
        # 如果包含图表，生成并保存
        if include_plots:
            plots_dir = os.path.join(report_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            report_content += "## 4. 可视化图表\n"
            
            # 绘制评估指标图
            metrics_plot_path = os.path.join(plots_dir, "evaluation_metrics.png")
            self.plot_evaluation_metrics(results, save_path=metrics_plot_path, show=False)
            report_content += f"![评估指标图](plots/evaluation_metrics.png)\n\n"
            
            # 绘制混淆矩阵
            cm = np.array(results.get('confusion_matrix', []))
            if cm.shape[0] > 0:
                cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
                self.plot_confusion_matrix(cm, save_path=cm_plot_path, show=False)
                report_content += f"![混淆矩阵](plots/confusion_matrix.png)\n\n"
            
            # 绘制类别指标
            class_plot_path = os.path.join(plots_dir, "class_metrics.png")
            self.plot_class_metrics(class_report, save_path=class_plot_path, show=False)
            report_content += f"![类别指标](plots/class_metrics.png)\n\n"
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"评估报告已保存到：{report_path}")
    
    def visualize_all_results(self, 
                            results: Dict[str, float],
                            path_lengths: Optional[List[int]] = None,
                            embeddings: Optional[np.ndarray] = None,
                            labels: Optional[np.ndarray] = None,
                            graph_data: Optional[Any] = None,
                            source_node: Optional[int] = None,
                            target_node: Optional[int] = None,
                            paths: Optional[List[List[int]]] = None,
                            report_path: str = "evaluation_report.md"
                            ):
        """
        可视化所有结果
        
        Args:
            results (Dict[str, float]): 评估结果
            path_lengths (Optional[List[int]]): 路径长度列表
            embeddings (Optional[np.ndarray]): 嵌入向量
            labels (Optional[np.ndarray]): 标签
            graph_data (Optional[Any]): 图数据
            source_node (Optional[int]): 源节点
            target_node (Optional[int]): 目标节点
            paths (Optional[List[List[int]]]): 路径列表
            report_path (str): 报告保存路径
        """
        logger.info("开始可视化所有结果...")
        
        # 创建报告目录
        report_dir = os.path.dirname(report_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # 创建评估报告
        self.create_evaluation_report(results, report_path, include_plots=True)
        
        # 如果有路径长度数据，绘制路径长度分布
        if path_lengths is not None and len(path_lengths) > 0:
            plots_dir = os.path.join(report_dir, "plots")
            path_plot_path = os.path.join(plots_dir, "path_length_distribution.png")
            self.plot_path_length_distribution(path_lengths, save_path=path_plot_path, show=False)
        
        # 如果有嵌入数据，绘制2D嵌入图
        if embeddings is not None:
            plots_dir = os.path.join(report_dir, "plots")
            emb_plot_path = os.path.join(plots_dir, "embedding_2d.png")
            self.plot_embedding_2d(embeddings, labels=labels, save_path=emb_plot_path, show=False)
        
        # 如果有图数据，绘制PPI网络图
        if graph_data is not None:
            plots_dir = os.path.join(report_dir, "plots")
            graph_plot_path = os.path.join(plots_dir, "ppi_graph.png")
            self.plot_ppi_graph(
                graph_data, 
                source_node=source_node, 
                target_node=target_node,
                paths=paths,
                save_path=graph_plot_path,
                show=False
            )
        
        logger.info(f"所有结果可视化完成，报告已保存到：{report_path}")
