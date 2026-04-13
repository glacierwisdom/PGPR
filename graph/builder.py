import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
# 移除了RandomNodeSampler的导入
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import logging
from pathlib import Path
from data.dataset import PPIDataset
from data.blastp_utils import BlastpSimilarityFinder

logger = logging.getLogger(__name__)


class PPIGraphBuilder:
    """蛋白质相互作用图构建器"""
    
    def __init__(self, 
                 esm_dim: int = 320, 
                 edge_dim: int = 32, 
                 device: str = "cuda",
                 data_dir: Optional[str] = None,
                 use_blast: bool = False,
                 num_neighbors: int = 8,
                 max_path_length: int = 10,
                 **kwargs
                 ):
        """
        初始化PPIGraphBuilder
        
        Args:
            esm_dim (int): ESM特征维度
            edge_dim (int): 边特征维度
            device (str or torch.device): 设备
            data_dir (str, optional): 数据目录路径
            use_blast (bool, optional): 是否使用BLAST结果
            num_neighbors (int, optional): 邻居数量
            max_path_length (int, optional): 最大路径长度
        """
        self.esm_dim = esm_dim
        self.edge_dim = edge_dim
        # 检查设备是否可用
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.use_blast = use_blast
        self.num_neighbors = num_neighbors
        self.max_path_length = max_path_length
        
        # 序列到ID的映射
        self.sequence_to_id = {}
        self.id_to_sequence = {}
        
        # 关系类型嵌入 (SHS27k 有 7 种 PPI 关系类型, 0-6)
        self.num_relations = kwargs.get('num_relations', 7)
        self.relation_embedding = torch.nn.Embedding(self.num_relations, edge_dim).to(self.device)
        self.similarity_relation_type = "__similarity__"
        
        # 初始化BLAST工具
        if self.use_blast:
            blast_dir = self.data_dir / "blast"
            blast_dir.mkdir(parents=True, exist_ok=True)
            self.similarity_finder = BlastpSimilarityFinder(
                blast_db_path=blast_dir / "protein_db",
                blast_output_dir=blast_dir / "output",
                evalue=kwargs.get('blast_evalue', 1e-5),
                max_target_seqs=kwargs.get('blast_num_alignments', 10)
            )

    def load_sequence_dict(self, dict_path: str):
        """
        加载序列到ID的映射字典
        
        Args:
            dict_path (str): 字典文件路径
        """
        try:
            import pandas as pd
            df = pd.read_csv(dict_path, sep='\t', header=None, names=['id', 'sequence'])
            for _, row in df.iterrows():
                self.sequence_to_id[row['sequence']] = row['id']
                self.id_to_sequence[row['id']] = row['sequence']
            logger.info(f"从 {dict_path} 加载了 {len(self.sequence_to_id)} 条序列映射信息")
        except Exception as e:
            logger.error(f"加载序列映射字典失败: {e}")

    def load_background_edges(self, background_file: str) -> List[Tuple[str, str, Any]]:
        """
        加载背景图的边
        
        Args:
            background_file (str): 背景数据文件路径
            
        Returns:
            List[Tuple[str, str, Any]]: 边列表
        """
        try:
            import pandas as pd
            import ast
            
            logger.info(f"正在从 {background_file} 加载背景边...")
            df = pd.read_csv(background_file)
            
            background_edges = []
            
            # 处理 SHS27k_ml.csv 格式
            if 'id' in df.columns and 'mode' in df.columns:
                for _, row in df.iterrows():
                    # 解析 ID (例如 9606.ENSP00000000233-9606.ENSP00000250971)
                    ids = row['id'].split('-')
                    if len(ids) == 2:
                        p_a, p_b = ids[0], ids[1]
                        
                        # 解析多标签关系
                        try:
                            # 可能是字符串形式的列表 "['binding', 'catalysis']"
                            mode = row['mode']
                            if isinstance(mode, str) and mode.startswith('['):
                                mode_list = ast.literal_eval(mode)
                            else:
                                mode_list = [mode]
                            
                            # 转换为 7 维独热编码 (SHS27k)
                            # 关系类型映射
                            rel_map = {
                                'activation': 0,
                                'inhibition': 1,
                                'inhibitory': 1,
                                'binding': 2,
                                'catalysis': 3,
                                'expression': 4,
                                'ptm': 5,
                                'physical': 5,
                                'reaction': 6,
                                'genetic': 6,
                            }
                            
                            label = [0] * 7
                            for m in mode_list:
                                if m in rel_map:
                                    label[rel_map[m]] = 1
                            
                            background_edges.append((p_a, p_b, label))
                        except Exception as e:
                            logger.warning(f"解析边 {p_a}-{p_b} 的关系失败: {e}")
            
            logger.info(f"从背景文件加载了 {len(background_edges)} 条边")
            return background_edges
            
        except Exception as e:
            logger.error(f"加载背景边失败: {e}")
            return []

    def get_protein_info_by_sequence(self, sequence: str, prompt_designer) -> Dict:
        """
        通过序列获取蛋白质信息
        """
        protein_id = self.sequence_to_id.get(sequence)
        if protein_id and protein_id in prompt_designer.protein_id_to_info:
            return prompt_designer.protein_id_to_info[protein_id]
        
        # 如果直接找不到，尝试清理序列后匹配
        for sid, seq in self.id_to_sequence.items():
            if seq == sequence:
                if sid in prompt_designer.protein_id_to_info:
                    return prompt_designer.protein_id_to_info[sid]
        
        return {'name': 'Unknown', 'function': '暂无功能描述'}

    def build_graph_and_load_data(self, 
                                 split: str = 'train', 
                                 data_file: Optional[Union[str, Path]] = None,
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 num_workers: int = 4
                                 ) -> Tuple[Data, PPIDataset]:
        """
        构建图并加载数据
        
        Args:
            split (str): 数据集划分 ('train', 'val', 'test')
            data_file (Optional[Union[str, Path]]): 指定数据文件路径，如果为None则自动查找
            
        Returns:
            Tuple[Data, PPIDataset]: 图数据和数据集
        """
        # 加载训练数据
        logger.info(f"从 {self.data_dir} 加载 {split} 数据")
        
        if data_file is None:
            # 确定数据文件路径
            # 优先查找 shs27k_{split}.csv, 然后是 {split}.csv
            possible_files = [
                f"shs27k_{split}.csv",
                f"shs27k_{split}.tsv",
                f"{split}.csv",
                f"{split}.tsv"
            ]
            
            for fname in possible_files:
                fpath = Path(self.data_dir) / fname
                if fpath.exists():
                    data_file = fpath
                    break
        else:
            data_file = Path(data_file)
            if not data_file.is_absolute() and not data_file.exists():
                data_file = Path(self.data_dir) / data_file
                
        if data_file is None or not data_file.exists():
             raise FileNotFoundError(f"未找到 {split} 数据文件. Path: {data_file}")

        logger.info(f"使用数据文件: {data_file}")
        
        # 1. 自动加载序列字典
        candidate_dict_paths = []
        for root in [Path(self.data_dir), Path(self.data_dir).parent, Path(self.data_dir).parent.parent]:
            candidate_dict_paths.extend([
                root / "protein_id_seq.tsv",
                root / "processed" / "protein_id_seq.tsv",
                root / "raw" / "shs27k" / "extracted" / "raw_data" / "protein.SHS27k.sequences.dictionary.tsv",
                root / "raw" / "shs27k" / "extracted" / "raw_data" / "protein.STRING.sequences.dictionary.tsv",
            ])
        dict_path = next((p for p in candidate_dict_paths if p.exists()), None)
        
        if dict_path and dict_path.exists():
            self.load_sequence_dict(str(dict_path))
        
        # 2. 加载背景图 (仅在训练时加载)
        background_edges = []
        if split == 'train':
            # 强制使用 7 种关系类型，以匹配 SHS27k
            self.num_relations = 7
            self.relation_embedding = torch.nn.Embedding(7, self.edge_dim).to(self.device)
            
            candidate_bg_paths = []
            for root in [Path(self.data_dir), Path(self.data_dir).parent, Path(self.data_dir).parent.parent]:
                candidate_bg_paths.extend([
                    root / "background_edges.tsv",
                    root / "processed" / "background_edges.tsv",
                    root / "SHS27k_ml.csv",
                ])
            bg_path = next((p for p in candidate_bg_paths if p.exists()), None)
                
            if bg_path and bg_path.exists():
                background_edges = self.load_background_edges(str(bg_path))
        
        # 3. 加载数据集
        dataset = PPIDataset(data_path=data_file, seq_dict_path=str(dict_path) if dict_path and dict_path.exists() else None)
        
        # 4. 获取所有蛋白质序列并映射为ID
        protein_sequences = dataset.get_protein_sequences()
        for pid, seq in protein_sequences.items():
            if pid not in self.id_to_sequence:
                self.id_to_sequence[pid] = seq
            if seq and seq not in self.sequence_to_id:
                self.sequence_to_id[seq] = pid
        
        # 构建节点ID列表
        protein_ids_set = set()
        
        # 首先添加背景图中的所有节点
        for p_a, p_b, _ in background_edges:
            protein_ids_set.add(p_a)
            protein_ids_set.add(p_b)
            
        # 然后添加当前数据集中的节点
        mapped_dataset_pairs = []
        for id_a, id_b, label in dataset.get_ppi_pairs():
            protein_ids_set.add(id_a)
            protein_ids_set.add(id_b)
            mapped_dataset_pairs.append((id_a, id_b, label))
            
        protein_ids = list(protein_ids_set)
        
        # 5. 构建总边列表
        edge_list = []
        # 添加背景边
        edge_list.extend(background_edges)
        # 添加当前数据集的边
        edge_list.extend(mapped_dataset_pairs)
        
        # 去重：以 (p_a, p_b) 为键，保留标签
        unique_edges = {}
        for p_a, p_b, label in edge_list:
            # 排序以处理无向图
            pair = tuple(sorted([p_a, p_b]))
            if pair not in unique_edges:
                unique_edges[pair] = label
            else:
                # 如果重复，可以考虑合并标签（如果是多标签）
                if isinstance(label, list) and isinstance(unique_edges[pair], list):
                    unique_edges[pair] = [max(l1, l2) for l1, l2 in zip(label, unique_edges[pair])]
        
        final_edge_list = [(p[0], p[1], l) for p, l in unique_edges.items()]
        
        # 6. 如果使用BLAST，添加相似性边
        if self.use_blast:
            logger.info("正在计算蛋白质相似性并添加相似性边...")
            # 创建临时FASTA文件以构建数据库
            fasta_file = self.data_dir / f"proteins_{split}.fasta"
            from data.preprocessing import ProteinPreprocessor
            preprocessor = ProteinPreprocessor()
            preprocessor.create_blastp_index(protein_sequences, fasta_file)
            
            # 创建BLAST数据库
            self.similarity_finder.create_blast_db_from_fasta(fasta_file)
            
            # 批量查找相似蛋白质
            similarity_results = self.similarity_finder.batch_find_similar_proteins(protein_sequences)
            
            # 添加相似性边
            sim_edge_count = 0
            for protein_id, similar_list in similarity_results.items():
                for sim_id, score in similar_list:
                    # 避免自环和重复边（因为后面会添加反向边）
                    if protein_id != sim_id:
                        final_edge_list.append((protein_id, sim_id, self.similarity_relation_type))
                        sim_edge_count += 1
            
            logger.info(f"添加了 {sim_edge_count} 条相似性边")
        
        # 初始化蛋白质特征（使用零向量，后续会被ESM嵌入替换）
        protein_features = {pid: torch.zeros(self.esm_dim, device=self.device) for pid in protein_ids}
        
        # 构建图
        graph_data = self.build_graph(
            protein_ids=protein_ids,
            protein_features=protein_features,
            edge_list=final_edge_list
        )
        
        return graph_data, dataset

    def build_graph(self, 
                   protein_ids: List[str],
                   protein_features: Dict[str, torch.Tensor],
                   edge_list: List[Tuple[str, str, int]],
                   node_attrs: Optional[Dict[str, Dict]] = None
                   ) -> Data:
        """
        构建PPI图
        
        Args:
            protein_ids (List[str]): 蛋白质ID列表
            protein_features (Dict[str, torch.Tensor]): 蛋白质ID到ESM特征的映射
            edge_list (List[Tuple[str, str, int]]): 边列表，包含(protein_A, protein_B, relation_type)
            node_attrs (Dict[str, Dict], optional): 节点属性字典
            
        Returns:
            torch_geometric.data.Data: 图数据对象
        """
        logger.info(f"开始构建PPI图，包含 {len(protein_ids)} 个节点和 {len(edge_list)} 条边")
        
        # 创建节点ID映射
        protein_id_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
        num_nodes = len(protein_ids)
        
        # 构建节点特征矩阵
        node_features = torch.zeros((num_nodes, self.esm_dim), device=self.device)
        for i, pid in enumerate(protein_ids):
            if pid in protein_features:
                node_features[i] = protein_features[pid]
            else:
                logger.warning(f"蛋白质 {pid} 没有ESM特征，使用零向量代替")
        
        # 构建边索引和边特征
        edge_index = []
        edge_attrs = []
        
        for protein_a, protein_b, relation_type in edge_list:
            if protein_a not in protein_id_to_idx or protein_b not in protein_id_to_idx:
                logger.warning(f"边 ({protein_a}, {protein_b}) 包含不存在的节点，跳过")
                continue
            
            # 添加正向边
            a_idx = protein_id_to_idx[protein_a]
            b_idx = protein_id_to_idx[protein_b]
            edge_index.append([a_idx, b_idx])
            
            if isinstance(relation_type, str) and relation_type == self.similarity_relation_type:
                rel_emb = torch.zeros(self.edge_dim, device=self.device)
            elif isinstance(relation_type, (list, np.ndarray, torch.Tensor)):
                rel_tensor = torch.as_tensor(relation_type, device=self.device)
                if rel_tensor.dim() > 0 and rel_tensor.numel() > 1:
                    # 多标签处理：获取所有激活关系的嵌入并取平均
                    active_indices = torch.nonzero(rel_tensor).flatten()
                    if active_indices.numel() > 0:
                        rel_emb = self.relation_embedding(active_indices).mean(dim=0)
                    else:
                        # 无激活关系，使用默认索引 0
                        rel_emb = self.relation_embedding(torch.tensor(0, device=self.device))
                else:
                    # 单个元素的张量/列表
                    rel_emb = self.relation_embedding(rel_tensor.long())
            else:
                # 整数或其他单标签类型
                rel_emb = self.relation_embedding(torch.tensor(relation_type, dtype=torch.long, device=self.device))
            
            # 确保 rel_emb 是 [edge_dim] 而不是 [1, edge_dim] 或 [7, edge_dim]
            if rel_emb.dim() > 1:
                rel_emb = rel_emb.view(-1, self.edge_dim).mean(dim=0)
            
            edge_attrs.append(rel_emb)
            
            # 添加反向边（无向图）
            edge_index.append([b_idx, a_idx])
            edge_attrs.append(rel_emb)
        
        # 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0) if edge_attrs else torch.empty((0, self.edge_dim), device=self.device)
        
        # 创建图数据对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            protein_ids=protein_ids,
            protein_id_to_idx=protein_id_to_idx
        )
        
        logger.info(f"PPI图构建完成，实际包含 {num_nodes} 个节点和 {edge_index.shape[1]} 条边")
        return graph_data

    def build_graph_for_pair(self, sequence_a: str, sequence_b: str, encoder=None) -> Data:
        """
        构建用于推理的蛋白质对图
        
        Args:
            sequence_a (str): 蛋白质A的序列
            sequence_b (str): 蛋白质B的序列
            encoder (Optional[ESMEncoder]): 预训练的ESM编码器，用于生成特征
            
        Returns:
            Data: 图数据对象
        """
        protein_ids = ["protein_a", "protein_b"]
        
        # 生成特征
        if encoder:
            embeddings = encoder.get_batch_embeddings([sequence_a, sequence_b])
            protein_features = {
                "protein_a": embeddings[0],
                "protein_b": embeddings[1]
            }
        else:
            logger.warning("未提供编码器，使用零向量作为特征")
            protein_features = {
                "protein_a": torch.zeros(self.esm_dim, device=self.device),
                "protein_b": torch.zeros(self.esm_dim, device=self.device)
            }
            
        # 构建边列表 (默认关系类型0)
        edge_list = [
            ("protein_a", "protein_b", 0)
        ]
        
        return self.build_graph(protein_ids, protein_features, edge_list)

    def sample_subgraph(self, graph_data: Data, sample_size: int = 100, num_layers: int = 2) -> Data:
        """
        采样子图用于批量训练
        
        Args:
            graph_data (Data): 完整图数据
            sample_size (int): 采样的节点数量
            num_layers (int): 采样的层数
            
        Returns:
            Data: 子图数据对象
        """
        # 实现简单的随机节点采样
        num_nodes = graph_data.x.size(0)
        if num_nodes <= sample_size:
            # 如果节点数量小于等于采样大小，直接返回原图
            return graph_data
        
        # 随机选择节点
        sampled_node_indices = torch.randperm(num_nodes, device=self.device)[:sample_size]
        sampled_node_indices = sampled_node_indices.sort()[0]
        
        # 创建节点索引映射
        node_idx_to_subgraph_idx = {idx.item(): i for i, idx in enumerate(sampled_node_indices)}
        
        # 找到与采样节点相关的边
        edge_index = graph_data.edge_index
        
        # 优化边过滤：使用向量化操作
        # 创建一个布尔张量，标记采样节点
        is_sampled = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        is_sampled[sampled_node_indices] = True
        
        # 检查边的两个端点是否都在采样集中
        mask = is_sampled[edge_index[0]] & is_sampled[edge_index[1]]
        
        # 过滤边
        sampled_edge_index = edge_index[:, mask]
        
        # 更新边索引到子图中的新索引
        # 使用 torch.bucketize 快速查找新索引
        # 首先需要确保 sampled_node_indices 是排序的（前面已经排过序了）
        sampled_edge_index = torch.bucketize(sampled_edge_index, sampled_node_indices)
        
        # 采样节点特征
        sampled_x = graph_data.x[sampled_node_indices]
        
        # 采样边特征
        sampled_edge_attr = graph_data.edge_attr[mask] if graph_data.edge_attr is not None else None
        
        # 采样蛋白质ID
        sampled_protein_ids = [graph_data.protein_ids[idx.item()] for idx in sampled_node_indices]
        
        # 采样蛋白质ID到索引的映射
        sampled_protein_id_to_idx = {pid: i for i, pid in enumerate(sampled_protein_ids)}
        
        # 创建子图数据对象
        subgraph = Data(
            x=sampled_x,
            edge_index=sampled_edge_index,
            edge_attr=sampled_edge_attr,
            protein_ids=sampled_protein_ids,
            protein_id_to_idx=sampled_protein_id_to_idx
        )
        
        return subgraph
    
    def batch_sample_subgraphs(self, graph_data: Data, batch_size: int = 4, sample_size: int = 100, num_layers: int = 2) -> Batch:
        """
        批量采样子图
        
        Args:
            graph_data (Data): 完整图数据
            batch_size (int): 批次大小
            sample_size (int): 每个子图的节点数量
            num_layers (int): 采样的层数
            
        Returns:
            Batch: 子图批次
        """
        subgraphs = []
        
        for _ in range(batch_size):
            subgraph = self.sample_subgraph(graph_data, sample_size, num_layers)
            subgraphs.append(subgraph)
        
        # 合并为批次
        return Batch.from_data_list(subgraphs)
    
    def get_edge_feature(self, graph_data: Data, node_i_idx: int, node_j_idx: int) -> Optional[torch.Tensor]:
        """
        获取特定边的特征
        
        Args:
            graph_data (Data): 图数据
            node_i_idx (int): 源节点索引
            node_j_idx (int): 目标节点索引
            
        Returns:
            Optional[torch.Tensor]: 边特征，如果边不存在则返回None
        """
        # 查找边索引
        edge_index = graph_data.edge_index.cpu().numpy()
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # 查找匹配的边
        matches = np.where((source_nodes == node_i_idx) & (target_nodes == node_j_idx))[0]
        
        if len(matches) > 0:
            edge_idx = matches[0]
            return graph_data.edge_attr[edge_idx]
        else:
            return None
    
    def add_node_feature(self, graph_data: Data, feature_name: str, feature_data: torch.Tensor) -> Data:
        """
        向图中添加节点特征
        
        Args:
            graph_data (Data): 图数据
            feature_name (str): 特征名称
            feature_data (torch.Tensor): 特征数据，形状为[num_nodes, feature_dim]
            
        Returns:
            Data: 更新后的图数据
        """
        if feature_data.shape[0] != graph_data.x.shape[0]:
            raise ValueError(f"特征数据的节点数 ({feature_data.shape[0]}) 与图的节点数 ({graph_data.x.shape[0]}) 不匹配")
        
        # 将新特征添加到图中
        setattr(graph_data, feature_name, feature_data)
        return graph_data
    
    def update_edge_features(self, graph_data: Data, new_edge_attr: torch.Tensor) -> Data:
        """
        更新图的边特征
        
        Args:
            graph_data (Data): 图数据
            new_edge_attr (torch.Tensor): 新的边特征，形状为[num_edges, edge_dim]
            
        Returns:
            Data: 更新后的图数据
        """
        if new_edge_attr.shape[0] != graph_data.edge_attr.shape[0]:
            raise ValueError(f"新边特征的数量 ({new_edge_attr.shape[0]}) 与图的边数 ({graph_data.edge_attr.shape[0]}) 不匹配")
        
        graph_data.edge_attr = new_edge_attr
        return graph_data
