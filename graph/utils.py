import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def pyg_data_to_networkx(graph_data: Data) -> nx.Graph:
    """
    将PyG的Data对象转换为NetworkX图
    
    Args:
        graph_data (Data): PyG图数据对象
        
    Returns:
        nx.Graph: NetworkX图对象
    """
    # 创建空图
    G = nx.Graph()
    
    # 添加节点
    num_nodes = graph_data.num_nodes
    G.add_nodes_from(range(num_nodes))
    
    # 添加节点属性
    if hasattr(graph_data, 'protein_ids'):
        for i, pid in enumerate(graph_data.protein_ids):
            G.nodes[i]['protein_id'] = pid
    
    # 添加节点特征
    if hasattr(graph_data, 'x'):
        for i in range(num_nodes):
            G.nodes[i]['feature'] = graph_data.x[i].detach().cpu().numpy()
    
    # 添加边
    edge_index = graph_data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    
    for i in range(0, num_edges, 2):  # 处理无向图，跳过反向边
        u = edge_index[0, i]
        v = edge_index[1, i]
        
        # 添加边属性
        edge_attr = {}
        if hasattr(graph_data, 'edge_attr'):
            edge_attr['feature'] = graph_data.edge_attr[i].detach().cpu().numpy()
        
        G.add_edge(u, v, **edge_attr)
    
    return G


def networkx_to_pyg_data(G: nx.Graph, edge_dim: int) -> Data:
    """
    将NetworkX图转换为PyG的Data对象
    
    Args:
        G (nx.Graph): NetworkX图对象
        edge_dim (int): 边特征维度
        
    Returns:
        Data: PyG图数据对象
    """
    # 获取节点列表
    num_nodes = G.number_of_nodes()
    nodes = sorted(G.nodes())
    
    # 构建节点特征矩阵
    if 'feature' in G.nodes[0]:
        feature_dim = G.nodes[0]['feature'].shape[0]
        x = torch.zeros((num_nodes, feature_dim))
        for i, node in enumerate(nodes):
            x[i] = torch.tensor(G.nodes[node]['feature'])
    else:
        # 如果没有节点特征，使用单位矩阵作为初始特征
        x = torch.eye(num_nodes)
    
    # 构建边索引和边特征
    edges = []
    edge_attrs = []
    
    for u, v, data in G.edges(data=True):
        # 添加正向边
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)
        edges.append([u_idx, v_idx])
        
        # 添加边特征
        if 'feature' in data:
            edge_attr = torch.tensor(data['feature'])
        else:
            # 如果没有边特征，使用零向量
            edge_attr = torch.zeros(edge_dim)
        edge_attrs.append(edge_attr)
        
        # 添加反向边（无向图）
        edges.append([v_idx, u_idx])
        edge_attrs.append(edge_attr)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs, dim=0) if edge_attrs else None
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_shortest_path(G: nx.Graph, start_node: int, end_node: int) -> Optional[List[int]]:
    """
    获取两个节点之间的最短路径
    
    Args:
        G (nx.Graph): NetworkX图对象
        start_node (int): 起始节点
        end_node (int): 目标节点
        
    Returns:
        Optional[List[int]]: 最短路径节点列表，如果不可达则返回None
    """
    try:
        path = nx.shortest_path(G, start_node, end_node)
        return path
    except nx.NetworkXNoPath:
        logger.warning(f"节点 {start_node} 和 {end_node} 之间没有路径")
        return None
    except nx.NodeNotFound as e:
        logger.warning(f"节点不存在: {e}")
        return None


def get_shortest_path_length(graph: nx.Graph | Data, start_node: int, end_node: int) -> Optional[int]:
    """
    获取两个节点之间的最短路径长度
    
    Args:
        graph (nx.Graph | Data): NetworkX图对象或PyG Data对象
        start_node (int): 起始节点
        end_node (int): 目标节点
        
    Returns:
        Optional[int]: 最短路径长度，如果不可达则返回None
    """
    # 如果是PyG Data对象，转换为NetworkX图
    if isinstance(graph, Data):
        G = pyg_data_to_networkx(graph)
    else:
        G = graph
    
    try:
        path_length = nx.shortest_path_length(G, start_node, end_node)
        return path_length
    except nx.NetworkXNoPath:
        logger.warning(f"节点 {start_node} 和 {end_node} 之间没有路径")
        return None
    except nx.NodeNotFound as e:
        logger.warning(f"节点不存在: {e}")
        return None


def get_node_neighbors(graph_data: Data, node_idx: int) -> List[int]:
    """
    获取节点的所有邻居
    
    Args:
        graph_data (Data): PyG图数据对象
        node_idx (int): 节点索引
        
    Returns:
        List[int]: 邻居节点索引列表
    """
    edge_index = graph_data.edge_index.cpu().numpy()
    
    # 查找所有以node_idx为源节点的边
    neighbor_indices = edge_index[1, edge_index[0] == node_idx]
    
    # 去重（因为无向图会有重复边）
    unique_neighbors = list(np.unique(neighbor_indices))
    
    return unique_neighbors


def get_edge_between(graph_data: Data, node_i: int, node_j: int) -> Optional[Tuple[int, int]]:
    """
    检查两个节点之间是否存在边，并返回边索引
    
    Args:
        graph_data (Data): PyG图数据对象
        node_i (int): 第一个节点
        node_j (int): 第二个节点
        
    Returns:
        Optional[Tuple[int, int]]: 边索引元组 (forward_edge_idx, backward_edge_idx)，如果不存在边则返回None
    """
    edge_index = graph_data.edge_index.cpu().numpy()
    
    # 查找正向边
    forward_mask = (edge_index[0] == node_i) & (edge_index[1] == node_j)
    forward_edges = np.where(forward_mask)[0]
    
    if len(forward_edges) == 0:
        return None
    
    # 查找反向边
    backward_mask = (edge_index[0] == node_j) & (edge_index[1] == node_i)
    backward_edges = np.where(backward_mask)[0]
    
    return (forward_edges[0], backward_edges[0]) if len(backward_edges) > 0 else (forward_edges[0], None)


def compute_graph_statistics(graph_data: Data) -> Dict:
    """
    计算图的统计信息
    
    Args:
        graph_data (Data): PyG图数据对象
        
    Returns:
        Dict: 图统计信息字典
    """
    num_nodes = graph_data.num_nodes
    num_edges = graph_data.edge_index.shape[1] // 2  # 无向图
    
    # 计算度分布
    edge_index = graph_data.edge_index.cpu().numpy()
    degrees = np.bincount(edge_index[0])
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
        'avg_degree': np.mean(degrees),
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees),
        'median_degree': np.median(degrees)
    }
    
    return stats


def normalize_node_features(x: torch.Tensor) -> torch.Tensor:
    """
    标准化节点特征
    
    Args:
        x (torch.Tensor): 节点特征矩阵
        
    Returns:
        torch.Tensor: 标准化后的节点特征矩阵
    """
    # 沿节点维度标准化
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1  # 避免除以零
    
    return (x - mean) / std


def normalize_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    标准化边特征
    
    Args:
        edge_attr (torch.Tensor): 边特征矩阵
        
    Returns:
        torch.Tensor: 标准化后的边特征矩阵
    """
    if edge_attr is None:
        return None
    
    # 沿边维度标准化
    mean = edge_attr.mean(dim=0, keepdim=True)
    std = edge_attr.std(dim=0, keepdim=True)
    std[std == 0] = 1  # 避免除以零
    
    return (edge_attr - mean) / std


def has_edge(graph_data: Data, node_i: int, node_j: int) -> bool:
    """
    检查两个节点之间是否存在边
    
    Args:
        graph_data (Data): PyG图数据对象
        node_i (int): 第一个节点
        node_j (int): 第二个节点
        
    Returns:
        bool: 如果存在边则返回True，否则返回False
    """
    edge_index = graph_data.edge_index.cpu().numpy()
    
    # 检查正向或反向边是否存在
    forward_mask = (edge_index[0] == node_i) & (edge_index[1] == node_j)
    backward_mask = (edge_index[0] == node_j) & (edge_index[1] == node_i)
    
    return np.any(forward_mask) or np.any(backward_mask)
