import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from typing import Dict, List, Any, Tuple, Optional
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from models.gnn_ppi import GNN_PPI
from models.esm_encoder import ESMEncoder
from evaluation.evaluator import Evaluator
from evaluation.visualization import Visualization


class ModelAnalyzer:
    """
    模型分析工具类，用于分析和可视化模型结构、性能和特征重要性
    """
    
    def __init__(self, output_dir: str = 'analysis_results'):
        """
        初始化模型分析器
        
        Args:
            output_dir: 分析结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualization = Visualization()
    
    def analyze_model_structure(self, model: Module, save_path: str = None) -> Dict[str, Any]:
        """
        分析模型结构
        
        Args:
            model: 要分析的PyTorch模型
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            Dict: 模型结构分析结果
        """
        print("Analyzing model structure...")
        
        results = {
            'model_name': model.__class__.__name__,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layers': [],
            'layer_types': {}
        }
        
        # 分析每一层
        for name, layer in model.named_modules():
            if name == '':  # 跳过模型本身
                continue
                
            layer_info = {
                'name': name,
                'type': layer.__class__.__name__,
                'parameters': 0,
                'trainable': 0
            }
            
            # 计算参数数量
            for param_name, param in layer.named_parameters():
                num_params = param.numel()
                layer_info['parameters'] += num_params
                results['total_parameters'] += num_params
                
                if param.requires_grad:
                    layer_info['trainable'] += num_params
                    results['trainable_parameters'] += num_params
            
            # 收集层类型统计
            if layer_info['type'] not in results['layer_types']:
                results['layer_types'][layer_info['type']] = 0
            results['layer_types'][layer_info['type']] += 1
            
            results['layers'].append(layer_info)
        
        # 打印分析结果
        print(f"Model: {results['model_name']}")
        print(f"Total parameters: {results['total_parameters']:,}")
        print(f"Trainable parameters: {results['trainable_parameters']:,}")
        print(f"Non-trainable parameters: {results['total_parameters'] - results['trainable_parameters']:,}")
        print(f"Number of layers: {len(results['layers'])}")
        print("\nLayer types:")
        for layer_type, count in results['layer_types'].items():
            print(f"  {layer_type}: {count}")
        
        print("\nDetailed layer information:")
        for layer in results['layers']:
            print(f"  {layer['name']} ({layer['type']}): {layer['parameters']:,} parameters ({layer['trainable']:,} trainable)")
        
        # 保存分析结果
        if save_path:
            with open(save_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            print(f"\nModel structure analysis saved to {save_path}")
        
        return results
    
    def visualize_model_structure(self, model: Module, save_path: str = None):
        """
        可视化模型结构
        
        Args:
            model: 要可视化的PyTorch模型
            save_path: 保存路径，如果为None则显示图表
        """
        print("\nVisualizing model structure...")
        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点和边
        for name, layer in model.named_modules():
            if name == '':  # 跳过模型本身
                continue
                
            # 获取层的基本信息
            layer_type = layer.__class__.__name__
            num_params = sum(p.numel() for p in layer.parameters())
            
            # 添加节点
            G.add_node(name, layer_type=layer_type, num_params=num_params)
            
            # 添加边
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name in G.nodes:
                    G.add_edge(parent_name, name)
            else:
                # 根节点
                G.add_edge('root', name)
        
        # 绘制图
        plt.figure(figsize=(15, 10))
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 节点颜色基于层类型
        layer_types = list(set(nx.get_node_attributes(G, 'layer_type').values()))
        color_map = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))
        color_dict = {lt: color_map[i] for i, lt in enumerate(layer_types)}
        
        node_colors = []
        for node in G.nodes():
            if node == 'root':
                node_colors.append('gray')
            else:
                node_colors.append(color_dict[G.nodes[node]['layer_type']])
        
        # 绘制节点
        nx.draw(G, pos, node_size=300, node_color=node_colors, 
               with_labels=True, labels={n: n.split('.')[-1] if n != 'root' else 'Model' for n in G.nodes()}, 
               font_size=8, font_weight='bold', edge_color='gray', alpha=0.8)
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=10, label=lt) for lt, color in color_dict.items()]
        plt.legend(handles=legend_elements, loc='upper right', title='Layer Types', 
                  bbox_to_anchor=(1.3, 1))
        
        plt.title('Model Structure Visualization')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model structure visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_feature_importance(self, model: Module, 
                                 data_loader: DataLoader,
                                 feature_names: List[str] = None,
                                 save_path: str = None) -> Dict[int, float]:
        """
        分析特征重要性
        
        Args:
            model: PyTorch模型
            data_loader: 数据加载器
            feature_names: 特征名称列表
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            Dict: 特征重要性字典
        """
        print("\nAnalyzing feature importance...")
        
        model.eval()
        device = next(model.parameters()).device
        
        # 获取输入特征维度
        sample_data = next(iter(data_loader))
        if isinstance(sample_data, (tuple, list)):
            x = sample_data[0]
        else:
            x = sample_data
            
        if isinstance(x, (tuple, list)):
            # 如果是图数据，使用节点特征
            x = x[0]
            
        feature_dim = x.size(-1)
        
        # 初始化特征重要性
        feature_importance = {i: 0.0 for i in range(feature_dim)}
        
        # 使用集成梯度方法计算特征重要性
        for batch_idx, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (tuple, list)):
                inputs, targets = batch_data
            else:
                inputs = batch_data
                targets = None
                
            if isinstance(inputs, (tuple, list)):
                x, edge_index, batch = inputs
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch = batch.to(device)
            else:
                x = inputs.to(device)
                
            # 前向传播获取基线输出
            with torch.no_grad():
                if hasattr(model, 'forward') and callable(model.forward):
                    if isinstance(inputs, (tuple, list)):
                        baseline_output = model(x, edge_index, batch)
                    else:
                        baseline_output = model(x)
                else:
                    raise ValueError("Model does not have a forward method")
            
            # 计算特征重要性
            for feature_idx in range(feature_dim):
                # 创建扰动输入
                perturbed_x = x.clone()
                perturbed_x[..., feature_idx] = torch.rand_like(perturbed_x[..., feature_idx]) * 0.1
                
                # 前向传播获取扰动输出
                with torch.no_grad():
                    if isinstance(inputs, (tuple, list)):
                        perturbed_output = model(perturbed_x, edge_index, batch)
                    else:
                        perturbed_output = model(perturbed_x)
                
                # 计算输出差异
                output_diff = torch.mean(torch.abs(perturbed_output - baseline_output)).item()
                feature_importance[feature_idx] += output_diff
            
            # 进度更新
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")
        
        # 归一化特征重要性
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature_idx in feature_importance:
                feature_importance[feature_idx] /= total_importance
        
        # 排序特征重要性
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 打印特征重要性
        print("\nFeature importance (top 20):")
        for i, (feature_idx, importance) in enumerate(sorted_features[:20]):
            feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f"Feature {feature_idx}"
            print(f"  {i+1}. {feature_name}: {importance:.6f}")
        
        # 保存特征重要性
        if save_path:
            with open(save_path, 'w') as f:
                import json
                json.dump(feature_importance, f, indent=2)
            print(f"Feature importance saved to {save_path}")
        
        return feature_importance
    
    def visualize_feature_importance(self, feature_importance: Dict[int, float],
                                   feature_names: List[str] = None,
                                   top_n: int = 20,
                                   save_path: str = None):
        """
        可视化特征重要性
        
        Args:
            feature_importance: 特征重要性字典
            feature_names: 特征名称列表
            top_n: 显示前N个重要特征
            save_path: 保存路径，如果为None则显示图表
        """
        print("\nVisualizing feature importance...")
        
        # 排序特征重要性
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # 准备数据
        indices = [f[0] for f in top_features]
        importance = [f[1] for f in top_features]
        
        if feature_names:
            labels = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices]
        else:
            labels = [f"Feature {i}" for i in indices]
        
        # 绘制条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance, y=labels, palette='viridis')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_model_performance(self, model: Module,
                                data_loaders: Dict[str, DataLoader],
                                metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                save_path: str = None) -> Dict[str, Dict[str, float]]:
        """
        分析模型在不同数据集上的性能
        
        Args:
            model: PyTorch模型
            data_loaders: 数据集加载器字典，键为数据集名称，值为DataLoader
            metrics: 要评估的指标列表
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            Dict: 模型性能分析结果
        """
        print("\nAnalyzing model performance...")
        
        evaluator = Evaluator(metrics=metrics)
        results = {}
        
        for dataset_name, data_loader in data_loaders.items():
            print(f"\nEvaluating on {dataset_name} dataset...")
            eval_results = evaluator.evaluate(model, data_loader)
            results[dataset_name] = eval_results
            
            print(f"  Results for {dataset_name}:")
            for metric, value in eval_results.items():
                print(f"    {metric}: {value:.6f}")
        
        # 保存性能分析结果
        if save_path:
            with open(save_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            print(f"\nModel performance analysis saved to {save_path}")
        
        return results
    
    def compare_models(self, models: Dict[str, Module],
                      data_loaders: Dict[str, DataLoader],
                      metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                      save_path: str = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        比较多个模型的性能
        
        Args:
            models: 模型字典，键为模型名称，值为Module
            data_loaders: 数据集加载器字典
            metrics: 要评估的指标列表
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            Dict: 模型比较结果
        """
        print("\nComparing models...")
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nAnalyzing {model_name}...")
            model_results = self.analyze_model_performance(model, data_loaders, metrics)
            results[model_name] = model_results
        
        # 打印比较结果
        print("\nModel Comparison Results:")
        for dataset_name in data_loaders.keys():
            print(f"\nDataset: {dataset_name}")
            print("  Model Performance:")
            for model_name, model_results in results.items():
                print(f"    {model_name}:")
                for metric, value in model_results[dataset_name].items():
                    print(f"      {metric}: {value:.6f}")
        
        # 保存比较结果
        if save_path:
            with open(save_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            print(f"\nModel comparison results saved to {save_path}")
        
        return results
    
    def visualize_model_comparison(self, comparison_results: Dict[str, Dict[str, Dict[str, float]]],
                                  dataset_name: str,
                                  save_path: str = None):
        """
        可视化模型比较结果
        
        Args:
            comparison_results: 模型比较结果
            dataset_name: 要比较的数据集名称
            save_path: 保存路径，如果为None则显示图表
        """
        print(f"\nVisualizing model comparison for {dataset_name}...")
        
        # 准备数据
        models = list(comparison_results.keys())
        metrics = list(next(iter(next(iter(comparison_results.values())).values())).keys())
        
        data = []
        for model_name in models:
            for metric in metrics:
                value = comparison_results[model_name][dataset_name][metric]
                data.append({'model': model_name, 'metric': metric, 'value': value})
        
        # 绘制分组条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='metric', y='value', hue='model', data=data, palette='Set2')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Model Comparison on {dataset_name} Dataset')
        plt.legend(title='Models')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_parameter_distribution(self, model: Module, save_path: str = None):
        """
        分析模型参数分布
        
        Args:
            model: PyTorch模型
            save_path: 保存路径，如果为None则显示图表
        """
        print("\nAnalyzing parameter distribution...")
        
        # 收集所有参数
        all_params = []
        param_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_np = param.detach().cpu().numpy().flatten()
                all_params.append(params_np)
                param_names.append(name)
        
        # 绘制参数分布直方图
        plt.figure(figsize=(15, 10))
        
        num_params = len(all_params)
        rows = (num_params + 1) // 2
        
        for i, (params, name) in enumerate(zip(all_params, param_names)):
            ax = plt.subplot(rows, 2, i + 1)
            sns.histplot(params, bins=50, ax=ax)
            ax.set_title(f"{name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter distribution visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_full_analysis(self, model: Module,
                         data_loaders: Dict[str, DataLoader],
                         feature_names: List[str] = None,
                         metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                         analysis_name: str = 'full_analysis'):
        """
        运行完整的模型分析
        
        Args:
            model: PyTorch模型
            data_loaders: 数据集加载器字典
            feature_names: 特征名称列表
            metrics: 要评估的指标列表
            analysis_name: 分析名称
        """
        print(f"\nRunning full model analysis: {analysis_name}")
        
        # 创建分析目录
        analysis_dir = os.path.join(self.output_dir, analysis_name)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. 模型结构分析
        structure_results = self.analyze_model_structure(
            model, 
            save_path=os.path.join(analysis_dir, 'model_structure.json')
        )
        
        # 2. 模型结构可视化
        self.visualize_model_structure(
            model, 
            save_path=os.path.join(analysis_dir, 'model_structure.png')
        )
        
        # 3. 模型性能分析
        performance_results = self.analyze_model_performance(
            model, 
            data_loaders, 
            metrics, 
            save_path=os.path.join(analysis_dir, 'model_performance.json')
        )
        
        # 4. 特征重要性分析（仅在训练集上）
        if 'train' in data_loaders:
            feature_importance = self.analyze_feature_importance(
                model, 
                data_loaders['train'], 
                feature_names, 
                save_path=os.path.join(analysis_dir, 'feature_importance.json')
            )
            
            # 特征重要性可视化
            self.visualize_feature_importance(
                feature_importance, 
                feature_names, 
                save_path=os.path.join(analysis_dir, 'feature_importance.png')
            )
        
        # 5. 参数分布分析
        self.analyze_parameter_distribution(
            model, 
            save_path=os.path.join(analysis_dir, 'parameter_distribution.png')
        )
        
        print(f"\nFull analysis completed! Results saved to {analysis_dir}")
        
        return {
            'structure': structure_results,
            'performance': performance_results,
            'feature_importance': feature_importance if 'train' in data_loaders else None
        }


# 示例用法
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ModelAnalyzer()
    
    # 初始化模型
    esm_encoder = ESMEncoder(
        model_name="facebook/esm2_t6_8M_UR50D",
        pooling_strategy="mean"
    )
    
    model = GNN_PPI(
        num_features=320,
        hidden_dim=128,
        num_classes=1,
        num_layers=2
    )
    
    # 加载模型到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 运行模型结构分析
    print("=== Model Structure Analysis ===")
    analyzer.analyze_model_structure(model)
    
    # 可视化模型结构
    analyzer.visualize_model_structure(model, save_path="artifacts/analysis/model_structure.png")
    
    # 分析参数分布
    analyzer.analyze_parameter_distribution(model, save_path="artifacts/analysis/parameter_distribution.png")
    
    print("\nModel analysis completed successfully!")
