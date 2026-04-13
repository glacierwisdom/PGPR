import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LLMUtils:
    """
    LLM相关工具函数类
    """
    
    @staticmethod
    def extract_relation_from_text(text: str, relation_map: Dict[int, str]) -> Optional[int]:
        """
        从生成的文本中提取关系类型
        
        Args:
            text (str): 生成的文本
            relation_map (Dict[int, str]): 关系类型映射
            
        Returns:
            Optional[int]: 关系类型索引，如果未找到则返回None
        """
        text_lower = text.lower()
        
        # 反向映射
        reverse_map = {v.lower(): k for k, v in relation_map.items()}
        
        # 查找关系类型
        for relation_text, relation_idx in reverse_map.items():
            if relation_text in text_lower:
                return relation_idx
        
        return None
    
    @staticmethod
    def batch_extract_relation(texts: List[str], relation_map: Dict[int, str]) -> List[Optional[int]]:
        """
        批量从文本中提取关系类型
        
        Args:
            texts (List[str]): 生成的文本列表
            relation_map (Dict[int, str]): 关系类型映射
            
        Returns:
            List[Optional[int]]: 关系类型索引列表
        """
        return [LLMUtils.extract_relation_from_text(text, relation_map) for text in texts]
    
    @staticmethod
    def compute_confidence(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        计算预测置信度
        
        Args:
            logits (torch.Tensor): 模型输出logits [batch_size, num_classes]
            temperature (float): 温度参数
            
        Returns:
            torch.Tensor: 置信度 [batch_size]
        """
        # 应用softmax
        probabilities = F.softmax(logits / temperature, dim=-1)
        
        # 置信度定义为最大概率
        confidence = torch.max(probabilities, dim=-1)[0]
        
        return confidence
    
    @staticmethod
    def normalize_embeddings(embeddings: torch.Tensor, norm_type: float = 2.0) -> torch.Tensor:
        """
        归一化嵌入向量
        
        Args:
            embeddings (torch.Tensor): 嵌入向量 [batch_size, embedding_dim]
            norm_type (float): 归一化类型
            
        Returns:
            torch.Tensor: 归一化后的嵌入向量
        """
        return F.normalize(embeddings, p=norm_type, dim=-1)
    
    @staticmethod
    def compute_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        计算嵌入向量之间的余弦相似度
        
        Args:
            embeddings1 (torch.Tensor): 第一个嵌入向量集合 [batch_size1, embedding_dim]
            embeddings2 (torch.Tensor): 第二个嵌入向量集合 [batch_size2, embedding_dim]
            
        Returns:
            torch.Tensor: 相似度矩阵 [batch_size1, batch_size2]
        """
        # 归一化
        norm_emb1 = LLMUtils.normalize_embeddings(embeddings1)
        norm_emb2 = LLMUtils.normalize_embeddings(embeddings2)
        
        # 计算余弦相似度
        similarity = torch.mm(norm_emb1, norm_emb2.t())
        
        return similarity
    
    @staticmethod
    def filter_low_confidence_predictions(predictions: torch.Tensor, 
                                        confidence: torch.Tensor, 
                                        threshold: float = 0.5
                                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        过滤低置信度预测
        
        Args:
            predictions (torch.Tensor): 预测结果 [batch_size]
            confidence (torch.Tensor): 置信度 [batch_size]
            threshold (float): 置信度阈值
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 过滤后的预测结果和置信度
        """
        mask = confidence >= threshold
        
        return predictions[mask], confidence[mask]
    
    @staticmethod
    def generate_relation_prompt(protein1: str, protein2: str, relation: str) -> str:
        """
        生成关系提示
        
        Args:
            protein1 (str): 蛋白质1名称
            protein2 (str): 蛋白质2名称
            relation (str): 关系类型
            
        Returns:
            str: 关系提示
        """
        return f"{protein1}与{protein2}之间的相互作用关系是{relation}。"
    
    @staticmethod
    def compute_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """
        计算概率分布的熵
        
        Args:
            probabilities (torch.Tensor): 概率分布 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: 熵 [batch_size]
        """
        # 避免log(0)
        epsilon = 1e-10
        probabilities = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
        
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
        
        return entropy
    
    @staticmethod
    def top_k_predictions(probabilities: torch.Tensor, k: int = 3) -> List[List[Tuple[int, float]]]:
        """
        获取top-k预测结果
        
        Args:
            probabilities (torch.Tensor): 概率分布 [batch_size, num_classes]
            k (int): top-k数量
            
        Returns:
            List[List[Tuple[int, float]]]: top-k预测结果列表
        """
        batch_size, num_classes = probabilities.shape
        
        # 获取top-k索引
        top_k_indices = torch.topk(probabilities, k, dim=-1)[1]
        
        # 获取top-k概率
        top_k_probs = torch.gather(probabilities, 1, top_k_indices)
        
        # 转换为列表
        results = []
        for i in range(batch_size):
            sample_results = []
            for j in range(k):
                sample_results.append((top_k_indices[i][j].item(), top_k_probs[i][j].item()))
            results.append(sample_results)
        
        return results
    
    @staticmethod
    def compute_relation_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算关系预测准确率
        
        Args:
            predictions (torch.Tensor): 预测结果 [batch_size]
            labels (torch.Tensor): 真实标签 [batch_size]
            
        Returns:
            float: 准确率
        """
        correct = (predictions == labels).sum().item()
        total = len(labels)
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def generate_prompt_with_examples(source_protein: str, 
                                    target_protein: str,
                                    path: List[int],
                                    path_text: str,
                                    examples: List[Dict[str, str]]
                                    ) -> str:
        """
        生成带有示例的提示
        
        Args:
            source_protein (str): 源蛋白质名称
            target_protein (str): 目标蛋白质名称
            path (List[int]): 探索路径
            path_text (str): 路径文本描述
            examples (List[Dict[str, str]]): 示例列表
            
        Returns:
            str: 带有示例的提示
        """
        # 构建示例部分
        examples_text = "示例:\n"
        for i, example in enumerate(examples):
            examples_text += f"{i+1}. {example['input']}\n   输出: {example['output']}\n"
        
        # 构建完整提示
        prompt = f"""你是一个蛋白质相互作用关系预测专家。请基于探索路径预测两个蛋白质之间的相互作用关系。

{examples_text}

现在，请分析以下蛋白质相互作用路径，预测{target_protein}与{source_protein}之间的相互作用关系。

探索路径：
{path_text}

基于上述探索路径，预测{source_protein}与{target_protein}之间的相互作用关系。
请从以下选项中选择：激活作用、抑制作用、结合作用、催化作用、表达调控、物理相互作用、遗传相互作用。
"""
        
        return prompt
