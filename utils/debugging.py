import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from typing import Dict, List, Any, Tuple
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from models.gnn_ppi import GNN_PPI
from models.esm_encoder import ESMEncoder


class GradientChecker:
    """
    梯度检查工具，用于检测梯度消失/爆炸问题
    """
    def __init__(self):
        self.gradient_history = {}
        self.layer_names = []
        
    def register_hooks(self, model: Module):
        """
        为模型注册梯度钩子
        
        Args:
            model: 要检查的PyTorch模型
        """
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Embedding, nn.GRU, nn.LSTM)):
                self.layer_names.append(name)
                self.gradient_history[name] = []
                
                def hook_fn(layer_name, module, grad_input, grad_output):
                    if grad_input[0] is not None:
                        grad_norm = torch.norm(grad_input[0]).item()
                        self.gradient_history[layer_name].append(grad_norm)
                        print(f"Layer {layer_name} gradient norm: {grad_norm:.6f}")
                
                layer.register_backward_hook(hook_fn.__get__(name, self))
    
    def analyze_gradient_flow(self):
        """
        分析梯度流动情况
        
        Returns:
            Dict: 梯度分析结果，包括均值、标准差、最小值、最大值
        """
        results = {}
        for layer_name, grads in self.gradient_history.items():
            if grads:
                grads_np = np.array(grads)
                results[layer_name] = {
                    'mean': np.mean(grads_np),
                    'std': np.std(grads_np),
                    'min': np.min(grads_np),
                    'max': np.max(grads_np),
                    'exploding': any(g > 1e3 for g in grads),
                    'vanishing': any(g < 1e-6 for g in grads)
                }
        return results
    
    def plot_gradient_flow(self, save_path: str = None):
        """
        绘制梯度流动图
        
        Args:
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(12, 8))
        
        for layer_name, grads in self.gradient_history.items():
            if grads:
                plt.plot(grads, label=layer_name)
        
        plt.title('Gradient Flow During Training')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gradient flow plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class ActivationStats:
    """
    激活统计工具，用于监控激活值分布
    """
    def __init__(self):
        self.activation_history = {}
        self.layer_names = []
        
    def register_hooks(self, model: Module):
        """
        为模型注册激活钩子
        
        Args:
            model: 要检查的PyTorch模型
        """
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.GELU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                self.layer_names.append(name)
                self.activation_history[name] = []
                
                def hook_fn(layer_name, module, input, output):
                    if isinstance(output, tuple):
                        act = output[0]
                    else:
                        act = output
                        
                    # 计算激活统计
                    mean = act.mean().item()
                    std = act.std().item()
                    min_val = act.min().item()
                    max_val = act.max().item()
                    zero_ratio = (act == 0).float().mean().item()
                    
                    self.activation_history[layer_name].append({
                        'mean': mean,
                        'std': std,
                        'min': min_val,
                        'max': max_val,
                        'zero_ratio': zero_ratio
                    })
                    
                    print(f"Layer {layer_name} activation stats - "
                          f"mean: {mean:.4f}, std: {std:.4f}, "
                          f"min: {min_val:.4f}, max: {max_val:.4f}, "
                          f"zero_ratio: {zero_ratio:.4f}")
                
                layer.register_forward_hook(hook_fn.__get__(name, self))
    
    def analyze_activations(self):
        """
        分析激活值分布
        
        Returns:
            Dict: 激活分析结果
        """
        results = {}
        for layer_name, activations in self.activation_history.items():
            if activations:
                means = [a['mean'] for a in activations]
                stds = [a['std'] for a in activations]
                min_vals = [a['min'] for a in activations]
                max_vals = [a['max'] for a in activations]
                zero_ratios = [a['zero_ratio'] for a in activations]
                
                results[layer_name] = {
                    'mean': np.mean(means),
                    'std': np.mean(stds),
                    'min': np.min(min_vals),
                    'max': np.max(max_vals),
                    'zero_ratio': np.mean(zero_ratios),
                    'saturated': any(m > 0.9 for m in max_vals) if 'ReLU' in layer_name else False
                }
        return results
    
    def plot_activation_distribution(self, layer_name: str, step: int = -1, save_path: str = None):
        """
        绘制指定层的激活值分布
        
        Args:
            layer_name: 层名称
            step: 训练步骤，-1表示最后一步
            save_path: 保存路径，如果为None则显示图表
        """
        if layer_name not in self.activation_history:
            print(f"Layer {layer_name} not found in activation history")
            return
            
        activations = self.activation_history[layer_name]
        if not activations:
            print(f"No activation data for layer {layer_name}")
            return
            
        plt.figure(figsize=(10, 6))
        
        # 绘制激活统计随时间变化
        means = [a['mean'] for a in activations]
        stds = [a['std'] for a in activations]
        zero_ratios = [a['zero_ratio'] for a in activations]
        
        plt.subplot(3, 1, 1)
        plt.plot(means)
        plt.title(f"{layer_name} Activation Mean over Time")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(stds)
        plt.title(f"{layer_name} Activation Std over Time")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(zero_ratios)
        plt.title(f"{layer_name} Activation Zero Ratio over Time")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activation distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class VisualizationTools:
    """
    可视化工具类
    """
    
    @staticmethod
    def visualize_attention_weights(attention_weights: torch.Tensor, 
                                  token_labels: List[str] = None, 
                                  save_path: str = None):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重张量，形状为 [head, seq_len, seq_len]
            token_labels: 序列标签列表
            save_path: 保存路径，如果为None则显示图表
        """
        if attention_weights.dim() == 2:
            # 只有一层注意力
            attention_weights = attention_weights.unsqueeze(0)
        
        num_heads = attention_weights.size(0)
        seq_len = attention_weights.size(1)
        
        plt.figure(figsize=(12, 10))
        
        for head_idx in range(num_heads):
            ax = plt.subplot(2, (num_heads + 1) // 2, head_idx + 1)
            head_weights = attention_weights[head_idx].detach().cpu().numpy()
            
            sns.heatmap(head_weights, cmap='viridis', ax=ax)
            ax.set_title(f"Head {head_idx + 1}")
            
            if token_labels:
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
                ax.set_yticklabels(token_labels, rotation=0, fontsize=8)
            else:
                ax.set_xlabel("Target Positions")
                ax.set_ylabel("Source Positions")
        
        plt.suptitle('Attention Weights Visualization')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention weights visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_graph_structure(edge_index: torch.Tensor, 
                                 node_features: torch.Tensor = None,
                                 node_labels: List[str] = None,
                                 save_path: str = None):
        """
        可视化图结构
        
        Args:
            edge_index: 边索引张量，形状为 [2, num_edges]
            node_features: 节点特征张量
            node_labels: 节点标签列表
            save_path: 保存路径，如果为None则显示图表
        """
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加边
        edges = edge_index.detach().cpu().numpy().T
        G.add_edges_from(edges)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制图
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 节点颜色基于特征（如果提供）
        if node_features is not None:
            node_color = node_features.detach().cpu().numpy().mean(axis=1)
            nx.draw(G, pos, node_size=500, node_color=node_color, 
                   cmap='coolwarm', with_labels=node_labels is not None,
                   labels={i: node_labels[i] for i in range(len(node_labels))} if node_labels else None)
            plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=plt.gca())
        else:
            nx.draw(G, pos, node_size=500, with_labels=node_labels is not None,
                   labels={i: node_labels[i] for i in range(len(node_labels))} if node_labels else None)
        
        plt.title('Graph Structure Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph structure visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_path_exploration(paths: List[List[int]], 
                                  edge_index: torch.Tensor,
                                  node_labels: List[str] = None,
                                  save_path: str = None):
        """
        可视化路径探索
        
        Args:
            paths: 路径列表，每个路径是节点索引列表
            edge_index: 边索引张量
            node_labels: 节点标签列表
            save_path: 保存路径，如果为None则显示图表
        """
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加所有边
        edges = edge_index.detach().cpu().numpy().T
        G.add_edges_from(edges)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制图
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制基础图
        nx.draw(G, pos, node_size=300, node_color='lightgray', 
               edge_color='gray', alpha=0.5, with_labels=node_labels is not None,
               labels={i: node_labels[i] for i in range(len(node_labels))} if node_labels else None)
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, path in enumerate(paths[:len(colors)]):
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color=colors[i], width=3, 
                                 alpha=0.8, style='dashed')
            
            # 标记路径节点
            nx.draw_networkx_nodes(G, pos, nodelist=path, 
                                  node_size=500, node_color=colors[i],
                                  alpha=0.8)
        
        plt.title('Path Exploration Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Path exploration visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PredictionInterpreter:
    """
    预测解释器，用于解释模型预测的决策过程
    """
    
    @staticmethod
    def generate_grad_cam(model: Module, input_data: Any, 
                         target_layer: Module, target_class: int = 0):
        """
        生成Grad-CAM可视化
        
        Args:
            model: PyTorch模型
            input_data: 输入数据
            target_layer: 目标层
            target_class: 目标类别
            
        Returns:
            np.ndarray: Grad-CAM热力图
        """
        # 注册钩子获取目标层输出和梯度
        activation_map = None
        grad_output = None
        
        def forward_hook(module, input, output):
            nonlocal activation_map
            activation_map = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            nonlocal grad_output
            grad_output = grad_out[0].detach()
        
        # 注册钩子
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # 前向传播
        model.eval()
        output = model(*input_data)
        
        # 计算目标类别损失
        loss = output[:, target_class].sum()
        
        # 反向传播
        model.zero_grad()
        loss.backward(retain_graph=True)
        
        # 移除钩子
        forward_handle.remove()
        backward_handle.remove()
        
        # 计算Grad-CAM
        if activation_map is not None and grad_output is not None:
            # 全局平均池化梯度
            weights = torch.mean(grad_output, dim=[2])
            
            # 加权组合激活图
            grad_cam = torch.sum(weights.unsqueeze(-1) * activation_map, dim=1)
            
            # ReLU
            grad_cam = torch.nn.functional.relu(grad_cam)
            
            # 归一化
            grad_cam = grad_cam / (torch.max(grad_cam) + 1e-8)
            
            return grad_cam.detach().cpu().numpy()
        
        return None
    
    @staticmethod
    def feature_perturbation_analysis(model: Module, 
                                     input_data: Any, 
                                     feature_indices: List[int] = None,
                                     num_samples: int = 10):
        """
        特征扰动分析，评估特征重要性
        
        Args:
            model: PyTorch模型
            input_data: 输入数据
            feature_indices: 要扰动的特征索引列表
            num_samples: 每个特征的扰动样本数
            
        Returns:
            Dict: 特征重要性分析结果
        """
        model.eval()
        
        # 原始预测
        with torch.no_grad():
            original_output = model(*input_data)
            original_pred = torch.sigmoid(original_output).item()
        
        # 获取输入特征
        if isinstance(input_data, (tuple, list)):
            x = input_data[0]
        else:
            x = input_data
        
        if feature_indices is None:
            feature_indices = list(range(x.size(1)))
        
        results = {}
        
        for idx in feature_indices:
            # 特征扰动
            perturbations = []
            for _ in range(num_samples):
                # 随机扰动特征
                perturbed_x = x.clone()
                if x.dim() == 2:
                    perturbed_x[:, idx] += torch.randn_like(perturbed_x[:, idx]) * 0.1
                elif x.dim() == 3:
                    perturbed_x[:, :, idx] += torch.randn_like(perturbed_x[:, :, idx]) * 0.1
                
                # 预测
                with torch.no_grad():
                    if isinstance(input_data, (tuple, list)):
                        perturbed_input = list(input_data)
                        perturbed_input[0] = perturbed_x
                        perturbed_output = model(*perturbed_input)
                    else:
                        perturbed_output = model(perturbed_x)
                    
                    perturbed_pred = torch.sigmoid(perturbed_output).item()
                    perturbations.append(perturbed_pred)
            
            # 计算特征重要性
            perturbation_std = np.std(perturbations)
            results[idx] = {
                'original_prediction': original_pred,
                'perturbed_predictions': perturbations,
                'importance': perturbation_std,
                'mean_change': np.mean(perturbations) - original_pred
            }
        
        return results
    
    @staticmethod
    def visualize_feature_importance(feature_importance: Dict[int, Dict],
                                   feature_names: List[str] = None,
                                   save_path: str = None):
        """
        可视化特征重要性
        
        Args:
            feature_importance: 特征重要性字典
            feature_names: 特征名称列表
            save_path: 保存路径，如果为None则显示图表
        """
        indices = sorted(feature_importance.keys())
        importance = [feature_importance[idx]['importance'] for idx in indices]
        
        plt.figure(figsize=(12, 8))
        
        if feature_names:
            x_labels = [feature_names[idx] for idx in indices]
        else:
            x_labels = [f"Feature {idx}" for idx in indices]
        
        plt.barh(range(len(indices)), importance)
        plt.yticks(range(len(indices)), x_labels)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance Analysis')
        plt.grid(axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class Debugger:
    """
    调试器主类，整合所有调试工具
    """
    def __init__(self, output_dir: str = 'debug_output'):
        """
        初始化调试器
        
        Args:
            output_dir: 调试输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.gradient_checker = GradientChecker()
        self.activation_stats = ActivationStats()
        self.visualization = VisualizationTools()
        self.interpreter = PredictionInterpreter()
    
    def attach_to_model(self, model: Module):
        """
        将调试器附加到模型
        
        Args:
            model: 要调试的PyTorch模型
        """
        self.gradient_checker.register_hooks(model)
        self.activation_stats.register_hooks(model)
    
    def run_debug_session(self, model: Module, 
                        input_data: Any, 
                        target: Any,
                        optimizer: Optimizer = None,
                        session_name: str = 'debug_session'):
        """
        运行完整的调试会话
        
        Args:
            model: PyTorch模型
            input_data: 输入数据
            target: 目标数据
            optimizer: 优化器
            session_name: 会话名称
        """
        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        print(f"Running debug session: {session_name}")
        
        # 1. 检查梯度
        print("\n1. Checking gradients...")
        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            output = model(*input_data)
            
            # 计算损失
            loss = nn.BCEWithLogitsLoss()(output, target)
            
            # 反向传播
            loss.backward()
            
            # 分析梯度
            grad_results = self.gradient_checker.analyze_gradient_flow()
            self.gradient_checker.plot_gradient_flow(os.path.join(session_dir, 'gradient_flow.png'))
        
        # 2. 分析激活值
        print("\n2. Analyzing activations...")
        model.eval()
        with torch.no_grad():
            output = model(*input_data)
        
        activation_results = self.activation_stats.analyze_activations()
        
        # 保存分析结果
        with open(os.path.join(session_dir, 'gradient_analysis.json'), 'w') as f:
            import json
            json.dump(grad_results, f, indent=2)
        
        with open(os.path.join(session_dir, 'activation_analysis.json'), 'w') as f:
            import json
            json.dump(activation_results, f, indent=2)
        
        print(f"Debug session completed. Results saved to {session_dir}")


# 示例用法
if __name__ == "__main__":
    # 创建调试器实例
    debugger = Debugger()
    
    # 初始化模型
    model = GNN_PPI(num_features=320, hidden_dim=128, num_classes=1, num_layers=2)
    
    # 附加调试器到模型
    debugger.attach_to_model(model)
    
    # 创建示例输入数据
    x = torch.randn(4, 100, 320)
    batch = torch.zeros(4 * 100, dtype=torch.long)
    for i in range(4):
        batch[i*100:(i+1)*100] = i
    
    target = torch.randint(0, 2, (4, 1)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 运行调试会话
    debugger.run_debug_session(model, (x, batch), target, optimizer, session_name='test_session')
    
    print("Debug session completed successfully!")
