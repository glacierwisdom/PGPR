import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from graph.utils import get_shortest_path_length
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class MultiScaleRewardCalculator:
    """
    多尺度奖励计算器
    实现多种奖励机制：
    - 基础奖励：LLM预测准确性
    - 路径长度奖励：鼓励简洁路径
    - 探索多样性奖励：鼓励访问新节点
    - 目标相关性奖励：基于图距离改进
    - 边特征合理性奖励：基于边特征置信度
    """
    
    def __init__(self, 
                 accuracy_reward: float = 2.0,
                 accuracy_penalty: float = -1.0,
                 length_penalty: float = -0.1,
                 diversity_bonus: float = 0.05,
                 distance_factor: float = 0.5,
                 edge_feature_weight: float = 0.2
                 ):
        """
        初始化奖励计算器
        
        Args:
            accuracy_reward (float): 预测正确的奖励
            accuracy_penalty (float): 预测错误的惩罚
            length_penalty (float): 路径长度惩罚因子
            diversity_bonus (float): 探索多样性奖励因子
            distance_factor (float): 距离奖励因子
            edge_feature_weight (float): 边特征奖励权重
        """
        self.accuracy_reward = accuracy_reward
        self.accuracy_penalty = accuracy_penalty
        self.length_penalty = length_penalty
        self.diversity_bonus = diversity_bonus
        self.distance_factor = distance_factor
        self.edge_feature_weight = edge_feature_weight
        
        # 缓存 NetworkX 图以提高效率
        self._nx_graph_cache = None
        self._last_graph_id = None
        
        logger.info(f"MultiScaleRewardCalculator初始化完成")
    
    def compute_accuracy_reward(self, 
                               predictions: torch.Tensor,
                               labels: torch.Tensor
                               ) -> torch.Tensor:
        """
        计算基础奖励：预测准确性
        
        Args:
            predictions (torch.Tensor): 预测结果 [batch_size]
            labels (torch.Tensor): 真实标签 [batch_size]
            
        Returns:
            torch.Tensor: 准确性奖励 [batch_size]
        """
        # 确保标签在同一设备上
        if labels.device != predictions.device:
            labels = labels.to(predictions.device)
            
        # 多标签准确性奖励
        # 1. Exact Match: 完全一致
        exact_match = (predictions == labels).all(dim=1).float()
        
        # 2. Hamming Accuracy: 每个标签的准确度
        label_accuracy = (predictions == labels).float().mean(dim=1)
        
        # 综合奖励：完全一致给满分，不完全一致则按 label_accuracy 比例给分
        accuracy_reward = exact_match * self.accuracy_reward + (1 - exact_match) * label_accuracy * self.accuracy_reward
        
        # 如果完全不一致（全错），给惩罚
        all_wrong = (predictions != labels).all(dim=1).float()
        accuracy_reward = accuracy_reward + all_wrong * self.accuracy_penalty
        
        return accuracy_reward
    
    def compute_path_length_reward(self, paths: List[List[int]], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        计算路径长度奖励：鼓励简洁路径
        
        Args:
            paths (List[List[int]]): 路径列表
            device (Optional[torch.device]): 设备
            
        Returns:
            torch.Tensor: 路径长度奖励 [batch_size]
        """
        rewards = []
        
        for path in paths:
            # 路径长度定义为步骤数
            length = len(path) - 1
            reward = self.length_penalty * length
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def compute_diversity_reward(self, paths: List[List[int]], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        计算探索多样性奖励：鼓励访问新节点
        
        Args:
            paths (List[List[int]]): 路径列表
            device (Optional[torch.device]): 设备
            
        Returns:
            torch.Tensor: 探索多样性奖励 [batch_size]
        """
        rewards = []
        
        for path in paths:
            # 计算唯一节点数量
            unique_nodes = len(set(path))
            reward = self.diversity_bonus * unique_nodes
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def _get_nx_graph(self, graph_data: Data):
        """
        获取 NetworkX 图（使用缓存）
        """
        # 使用 graph_data 的内存地址或特征哈希作为标识
        # 在 PyG 中，Data 对象通常是单例或在训练期间保持稳定
        graph_id = id(graph_data)
        
        if self._nx_graph_cache is None or self._last_graph_id != graph_id:
            from graph.utils import pyg_data_to_networkx
            self._nx_graph_cache = pyg_data_to_networkx(graph_data)
            self._last_graph_id = graph_id
            logger.debug(f"已更新 NetworkX 图缓存，ID: {graph_id}")
            
        return self._nx_graph_cache

    def compute_distance_reward(self, 
                              source_nodes: List[int],
                              target_nodes: List[int],
                              paths: List[List[int]],
                              graph_data: Data
                              ) -> torch.Tensor:
        """
        计算目标相关性奖励：基于图距离改进
        
        Args:
            source_nodes (List[int]): 源节点列表
            target_nodes (List[int]): 目标节点列表
            paths (List[List[int]]): 路径列表
            graph_data (Data): 图数据
            
        Returns:
            torch.Tensor: 距离奖励 [batch_size]
        """
        device = graph_data.x.device if hasattr(graph_data, "x") else torch.device('cpu')
        rewards = []
        
        # 使用缓存的 NetworkX 图
        import networkx as nx
        G = self._get_nx_graph(graph_data)
        
        for source, target, path in zip(source_nodes, target_nodes, paths):
            # 直接使用NetworkX计算最短路径长度
            try:
                shortest_length = nx.shortest_path_length(G, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                shortest_length = None
            
            if shortest_length is None:  # 没有路径
                reward = -1.0
            else:
                # 计算路径长度比
                path_length = len(path) - 1
                # 避免路径长度为 0 导致的除零错误
                if path_length <= 0:
                    reward = -0.5 # 给予轻微惩罚，因为没有进行任何探索
                else:
                    length_ratio = path_length / shortest_length if shortest_length > 0 else 1.0
                    # 距离奖励 = 因子 * (1 / length_ratio)
                    # 确保 length_ratio 不为 0
                    reward = self.distance_factor * (1.0 / max(length_ratio, 0.1))
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def compute_edge_feature_reward(self, 
                                  edge_features: Optional[torch.Tensor],
                                  edge_indices: Optional[List[List[Tuple[int, int]]]]
                                  ) -> torch.Tensor:
        """
        计算边特征合理性奖励：基于边特征置信度
        
        Args:
            edge_features (Optional[torch.Tensor]): 边特征张量 [num_edges, edge_dim]
            edge_indices (Optional[List[List[Tuple[int, int]]]]): 批次边索引
            
        Returns:
            torch.Tensor: 边特征奖励 [batch_size]
        """
        # 统一使用 edge_features 的设备
        device = edge_features.device if edge_features is not None else torch.device('cpu')
        
        if edge_features is None or edge_indices is None or edge_features.numel() == 0:
            batch_size = len(edge_indices) if edge_indices else 0
            return torch.zeros(batch_size, dtype=torch.float32, device=device)

        rewards = []
        
        # 计算每条边的置信度（边特征的L2范数）
        edge_confidence = torch.norm(edge_features, dim=1)
        
        # 归一化置信度到[0, 1]范围
        if edge_confidence.numel() > 1:
            edge_min = edge_confidence.min()
            edge_max = edge_confidence.max()
            if edge_max > edge_min:
                edge_confidence = (edge_confidence - edge_min) / (edge_max - edge_min)
            else:
                edge_confidence = torch.ones_like(edge_confidence)
        elif edge_confidence.numel() == 1:
            edge_confidence = torch.ones_like(edge_confidence)
        
        current_edge_idx = 0
        for i, path_edges in enumerate(edge_indices):
            if not path_edges:
                rewards.append(0.0)
                continue
            
            # 计算路径中所有边的平均置信度
            path_confidence = 0.0
            for _ in range(len(path_edges)):
                if current_edge_idx < len(edge_confidence):
                    path_confidence += edge_confidence[current_edge_idx].item()
                    current_edge_idx += 1
            
            avg_confidence = path_confidence / len(path_edges) if path_edges else 0.0
            reward = self.edge_feature_weight * avg_confidence
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def compute_rewards(self, 
                       predictions: torch.Tensor,
                       labels: torch.Tensor,
                       paths: List[List[int]],
                       graph_data: Data,
                       source_nodes: Optional[List[int]] = None,
                       target_nodes: Optional[List[int]] = None,
                       edge_features: Optional[torch.Tensor] = None,
                       edge_indices: Optional[List[List[Tuple[int, int]]]] = None
                       ) -> Dict[str, torch.Tensor]:
        """
        计算所有奖励
        
        Args:
            predictions (torch.Tensor): 预测结果 [batch_size]
            labels (torch.Tensor): 真实标签 [batch_size]
            paths (List[List[int]]): 路径列表
            graph_data (Data): 图数据
            source_nodes (Optional[List[int]]): 源节点列表
            target_nodes (Optional[List[int]]): 目标节点列表
            edge_features (Optional[torch.Tensor]): 边特征张量
            edge_indices (Optional[List[List[Tuple[int, int]]]]): 批次边索引
            
        Returns:
            Dict[str, torch.Tensor]: 奖励字典
        """
        batch_size = len(predictions)
        rewards = {}
        
        # 1. 计算基础奖励
        rewards['accuracy'] = self.compute_accuracy_reward(predictions, labels)
        
        # 2. 计算路径长度奖励
        rewards['length'] = self.compute_path_length_reward(paths, device=predictions.device)
        
        # 3. 计算探索多样性奖励
        rewards['diversity'] = self.compute_diversity_reward(paths, device=predictions.device)
        
        # 4. 计算目标相关性奖励（如果提供了源节点和目标节点）
        if source_nodes and target_nodes:
            rewards['distance'] = self.compute_distance_reward(source_nodes, target_nodes, paths, graph_data)
        else:
            rewards['distance'] = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
        
        # 5. 计算边特征合理性奖励（如果提供了边特征和边索引）
        if edge_features is not None and edge_indices is not None:
            rewards['edge_feature'] = self.compute_edge_feature_reward(edge_features, edge_indices)
        else:
            rewards['edge_feature'] = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
        
        # 确保所有奖励都在正确的设备上
        for k, v in rewards.items():
            if v.device != predictions.device:
                rewards[k] = v.to(predictions.device)
                
        # 计算总奖励
        total_reward = sum(rewards.values())
        rewards['total'] = total_reward
        
        return rewards
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          next_values: torch.Tensor,
                          masks: torch.Tensor,
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95
                          ) -> torch.Tensor:
        """
        计算优势函数（GAE估计）
        
        Args:
            rewards (torch.Tensor): 奖励序列 [T, batch_size]
            values (torch.Tensor): 价值估计 [T, batch_size]
            next_values (torch.Tensor): 下一个状态的价值估计 [T, batch_size]
            masks (torch.Tensor): 掩码 [T, batch_size]
            gamma (float): 折扣因子
            gae_lambda (float): GAE参数
            
        Returns:
            torch.Tensor: 优势函数 [T, batch_size]
        """
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(T)):
            # 计算TD误差
            delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
            
            # GAE更新
            gae = delta + gamma * gae_lambda * masks[t] * gae
            advantages[t] = gae
        
        return advantages
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化奖励
        
        Args:
            rewards (torch.Tensor): 奖励 [batch_size] 或 [T, batch_size]
            
        Returns:
            torch.Tensor: 归一化后的奖励
        """
        if rewards.size(0) > 1:
            mean = rewards.mean()
            std = rewards.std() + 1e-8  # 避免除以零
            return (rewards - mean) / std
        else:
            return rewards
    
    def compute_reward_statistics(self, reward_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算奖励统计信息
        
        Args:
            reward_dict (Dict[str, torch.Tensor]): 奖励字典
            
        Returns:
            Dict[str, float]: 奖励统计信息
        """
        stats = {}
        
        for reward_name, reward_values in reward_dict.items():
            stats[f"{reward_name}_mean"] = reward_values.mean().item()
            stats[f"{reward_name}_std"] = reward_values.std().item()
            stats[f"{reward_name}_max"] = reward_values.max().item()
            stats[f"{reward_name}_min"] = reward_values.min().item()
        
        return stats

