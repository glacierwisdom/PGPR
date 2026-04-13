import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional
from torch_geometric.data import Data, Batch


class DynamicBatcher:
    """动态批处理器"""
    def __init__(self,
                 max_tokens: int = 10000,
                 max_batch_size: int = 32,
                 pad_token_id: int = 0):
        """
        初始化动态批处理器
        
        Args:
            max_tokens: 批次中最大token总数
            max_batch_size: 最大批次大小
            pad_token_id: 填充token的ID
        """
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id
    
    def batch_sequences(self, sequences: List[torch.Tensor]) -> List[List[int]]:
        """
        根据序列长度进行动态批处理
        
        Args:
            sequences: 序列列表，每个序列是一个1D张量
        
        Returns:
            批次索引列表，每个批次是一个索引列表
        """
        # 计算序列长度并按降序排序
        seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
        seq_lengths.sort(key=lambda x: x[1], reverse=True)
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, length in seq_lengths:
            # 检查是否可以将当前序列添加到当前批次
            if (current_tokens + length <= self.max_tokens and
                len(current_batch) + 1 <= self.max_batch_size):
                current_batch.append(idx)
                current_tokens += length
            else:
                # 开始新批次
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
        
        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def pad_batch(self, sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对批次中的序列进行填充
        
        Args:
            sequences: 序列列表
        
        Returns:
            填充后的序列张量和长度张量
        """
        max_length = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        
        padded_sequences = torch.full((batch_size, max_length), self.pad_token_id, dtype=sequences[0].dtype)
        lengths = torch.zeros(batch_size, dtype=torch.int64)
        
        for i, seq in enumerate(sequences):
            seq_length = len(seq)
            padded_sequences[i, :seq_length] = seq
            lengths[i] = seq_length
        
        return padded_sequences, lengths


class GraphNeighborSampler:
    """图邻居采样器"""
    def __init__(self,
                 num_layers: int,
                 num_neighbors: List[int],
                 replace: bool = False):
        """
        初始化图邻居采样器
        
        Args:
            num_layers: GNN层数
            num_neighbors: 每层采样的邻居数量列表
            replace: 是否替换采样
        """
        assert num_layers == len(num_neighbors), "num_layers must match len(num_neighbors)"
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.replace = replace
    
    def sample(self, data: Data, batch_nodes: torch.Tensor) -> Data:
        """
        对图进行邻居采样
        
        Args:
            data: 图数据
            batch_nodes: 批次节点索引
        
        Returns:
            采样后的子图数据
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        x = data.x
        
        # 构建邻接列表
        adj_list = self._build_adjacency_list(edge_index)
        
        # 多层采样
        sampled_nodes = [batch_nodes]
        sampled_edges = []
        
        for k in reversed(range(self.num_layers)):
            # 获取当前层的节点
            current_nodes = sampled_nodes[-1]
            
            # 采样邻居
            neighbors = self._sample_neighbors(adj_list, current_nodes, self.num_neighbors[k])
            
            # 添加到采样节点列表
            sampled_nodes.append(neighbors)
            
            # 构建当前层的边
            layer_edges = self._get_edges(adj_list, current_nodes, neighbors, edge_index)
            sampled_edges.append(layer_edges)
        
        # 反转采样节点列表（从外层到内层）
        sampled_nodes.reverse()
        
        # 合并所有采样节点
        all_nodes = torch.cat(sampled_nodes).unique()
        
        # 创建节点映射（原始节点ID -> 采样后ID）
        node_map = {node.item(): idx for idx, node in enumerate(all_nodes)}
        
        # 重新映射边
        remapped_edges = []
        for edge in sampled_edges:
            if edge.size(1) > 0:
                remapped_edge = torch.tensor([[node_map[n.item()] for n in edge[0]],
                                              [node_map[n.item()] for n in edge[1]]],
                                             device=edge.device)
                remapped_edges.append(remapped_edge)
        
        # 合并所有边
        if remapped_edges:
            sampled_edge_index = torch.cat(remapped_edges, dim=1)
        else:
            sampled_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        
        # 重新映射节点特征
        sampled_x = x[all_nodes]
        
        # 构建采样后的子图数据
        sampled_data = Data(
            x=sampled_x,
            edge_index=sampled_edge_index,
            batch=torch.tensor([node_map[n.item()] for n in batch_nodes], device=batch_nodes.device)
        )
        
        # 添加边特征（如果存在）
        if edge_attr is not None:
            # 获取采样边对应的边特征
            edge_mask = self._get_edge_mask(edge_index, sampled_edge_index, all_nodes, node_map)
            sampled_edge_attr = edge_attr[edge_mask]
            sampled_data.edge_attr = sampled_edge_attr
        
        return sampled_data
    
    def _build_adjacency_list(self, edge_index: torch.Tensor) -> Dict[int, List[int]]:
        """构建邻接列表"""
        adj_list = {}
        for u, v in zip(edge_index[0], edge_index[1]):
            u = u.item()
            v = v.item()
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)
        return adj_list
    
    def _sample_neighbors(self, adj_list: Dict[int, List[int]],
                         nodes: torch.Tensor, num_neighbors: int) -> torch.Tensor:
        """采样邻居"""
        all_neighbors = []
        
        for node in nodes:
            node = node.item()
            if node not in adj_list:
                continue
                
            neighbors = adj_list[node]
            if len(neighbors) <= num_neighbors:
                sampled = neighbors
            else:
                sampled = np.random.choice(neighbors, num_neighbors, replace=self.replace).tolist()
            
            all_neighbors.extend(sampled)
        
        return torch.tensor(all_neighbors, dtype=torch.long, device=nodes.device).unique()
    
    def _get_edges(self, adj_list: Dict[int, List[int]],
                  src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
        """获取源节点和目标节点之间的边"""
        src_set = set(src_nodes.tolist())
        dst_set = set(dst_nodes.tolist())
        
        edges = []
        for u, v in zip(edge_index[0], edge_index[1]):
            u = u.item()
            v = v.item()
            if u in src_set and v in dst_set:
                edges.append([u, v])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long, device=edge_index.device).t()
        else:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    
    def _get_edge_mask(self, original_edge_index: torch.Tensor,
                      sampled_edge_index: torch.Tensor,
                      sampled_nodes: torch.Tensor,
                      node_map: Dict[int, int]) -> torch.Tensor:
        """获取原始边索引中采样边的掩码"""
        mask = torch.zeros(original_edge_index.size(1), dtype=torch.bool, device=original_edge_index.device)
        
        # 构建采样边的集合
        sampled_edge_set = set()
        for u, v in zip(sampled_edge_index[0], sampled_edge_index[1]):
            u_orig = sampled_nodes[u.item()].item()
            v_orig = sampled_nodes[v.item()].item()
            sampled_edge_set.add((u_orig, v_orig))
        
        # 查找原始边索引中的采样边
        for i, (u, v) in enumerate(zip(original_edge_index[0], original_edge_index[1])):
            u = u.item()
            v = v.item()
            if (u, v) in sampled_edge_set:
                mask[i] = True
        
        return mask


def collate_graph_data(batch: List[Data]) -> Batch:
    """
    图数据批处理函数
    
    Args:
        batch: 图数据列表
    
    Returns:
        批处理后的图数据
    """
    return Batch.from_data_list(batch)


def batch_to_device(batch: Union[torch.Tensor, Data, Dict[str, Any]],
                   device: torch.device) -> Union[torch.Tensor, Data, Dict[str, Any]]:
    """
    将批次数据移动到指定设备
    
    Args:
        batch: 批次数据
        device: 目标设备
    
    Returns:
        移动到目标设备的数据
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, Data):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    else:
        return batch


def get_batch_size(batch: Any) -> int:
    """
    获取批次大小
    
    Args:
        batch: 批次数据
    
    Returns:
        批次大小
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    elif isinstance(batch, Data):
        return batch.batch.size(0)
    elif isinstance(batch, dict) and 'batch' in batch:
        return batch['batch'].size(0)
    elif isinstance(batch, list):
        return len(batch)
    else:
        raise ValueError(f"Cannot determine batch size for type {type(batch)}")


def pad_sequences(sequences: List[torch.Tensor],
                 max_length: Optional[int] = None,
                 padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    序列填充函数
    
    Args:
        sequences: 序列列表
        max_length: 最大长度，None表示使用最长序列的长度
        padding_value: 填充值
    
    Returns:
        填充后的序列张量和长度张量
    """
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    if max_length is None:
        max_length = lengths.max().item()
    
    # 创建填充后的张量
    padded = torch.full((len(sequences), max_length), padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        end = min(len(seq), max_length)
        padded[i, :end] = seq[:end]
    
    return padded, lengths
