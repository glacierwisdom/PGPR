import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


class PPIMetrics:
    """PPI预测模型评估指标类"""
    
    def __init__(self):
        """初始化PPIMetrics"""
        pass
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        计算准确率
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_pred (torch.Tensor or np.ndarray): 预测标签
            
        Returns:
            float: 准确率
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_pred_np = PPIMetrics._to_numpy(y_pred)
        
        return accuracy_score(y_true_np, y_pred_np)
    
    @staticmethod
    def calculate_precision(y_true, y_pred, average='binary'):
        """
        计算精确率
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_pred (torch.Tensor or np.ndarray): 预测标签
            average (str): 多分类平均策略 ('binary', 'macro', 'micro', 'weighted')
            
        Returns:
            float: 精确率
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_pred_np = PPIMetrics._to_numpy(y_pred)
        
        return precision_score(y_true_np, y_pred_np, average=average)
    
    @staticmethod
    def calculate_recall(y_true, y_pred, average='binary'):
        """
        计算召回率
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_pred (torch.Tensor or np.ndarray): 预测标签
            average (str): 多分类平均策略 ('binary', 'macro', 'micro', 'weighted')
            
        Returns:
            float: 召回率
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_pred_np = PPIMetrics._to_numpy(y_pred)
        
        return recall_score(y_true_np, y_pred_np, average=average)
    
    @staticmethod
    def calculate_f1(y_true, y_pred, average='micro'):
        """
        计算F1分数
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_pred (torch.Tensor or np.ndarray): 预测标签
            average (str): 多分类平均策略 ('binary', 'macro', 'micro', 'weighted')
            
        Returns:
            float: F1分数
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_pred_np = PPIMetrics._to_numpy(y_pred)
        
        return f1_score(y_true_np, y_pred_np, average=average, zero_division=0)
    
    @staticmethod
    def calculate_roc_auc(y_true, y_score, average='macro', multi_class='ovr'):
        """
        计算ROC-AUC分数
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_score (torch.Tensor or np.ndarray): 预测概率分数
            average (str): 多分类平均策略 ('macro', 'micro', 'samples', 'weighted')
            multi_class (str): 多分类策略 ('ovr', 'ovo')
            
        Returns:
            float: ROC-AUC分数
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_score_np = PPIMetrics._to_numpy(y_score)
        
        try:
            return roc_auc_score(y_true_np, y_score_np, average=average, multi_class=multi_class)
        except Exception as e:
            logger.warning(f"无法计算ROC-AUC: {e}")
            return 0.0

    @staticmethod
    def calculate_auprc(y_true, y_score, average='macro'):
        """
        计算AUPRC (Average Precision) 分数
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_score (torch.Tensor or np.ndarray): 预测概率分数
            average (str): 多分类平均策略 ('macro', 'micro', 'weighted', 'samples')
            
        Returns:
            float: AUPRC分数
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_score_np = PPIMetrics._to_numpy(y_score)
        
        # 处理多分类/多标签情况
        if len(y_score_np.shape) > 1 and y_score_np.shape[1] > 1:
            # 如果y_true已经是和y_score形状一致的二值矩阵 (多标签情况)
            if y_true_np.shape == y_score_np.shape:
                return average_precision_score(y_true_np, y_score_np, average=average)
                
            # 对于多分类，需要将y_true转换为one-hot编码
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_true_np)
            if len(classes) > 1:
                y_true_bin = label_binarize(y_true_np, classes=classes)
                # 如果y_score的列数不等于类别数（可能有些类别在当前batch没出现）
                if y_true_bin.shape[1] != y_score_np.shape[1]:
                    # 这里可能需要更复杂的对齐逻辑，简单起见先返回0
                    return average_precision_score(y_true_bin, y_score_np[:, :y_true_bin.shape[1]], average=average)
                return average_precision_score(y_true_bin, y_score_np, average=average)
            else:
                return 0.0
        else:
            return average_precision_score(y_true_np, y_score_np, average=average)
    
    @staticmethod
    def calculate_average_precision(y_true, y_score, average='macro'):
        """
        计算平均精确率（PR曲线下面积）
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_score (torch.Tensor or np.ndarray): 预测概率分数
            average (str): 多分类平均策略 ('macro', 'micro', 'samples', 'weighted')
            
        Returns:
            float: 平均精确率
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_score_np = PPIMetrics._to_numpy(y_score)
        
        # 对于二分类情况，确保y_score是概率向量
        if len(y_score_np.shape) == 2 and y_score_np.shape[1] > 1:
            y_score_np = y_score_np[:, 1]
        
        return average_precision_score(y_true_np, y_score_np, average=average)
    
    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred):
        """
        计算混淆矩阵
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_pred (torch.Tensor or np.ndarray): 预测标签
            
        Returns:
            np.ndarray: 混淆矩阵
        """
        y_true_np = PPIMetrics._to_numpy(y_true)
        y_pred_np = PPIMetrics._to_numpy(y_pred)
        
        return confusion_matrix(y_true_np, y_pred_np)
    
    @staticmethod
    def get_thresholded_predictions(y_score, threshold=0.5):
        """
        根据阈值将概率转换为预测标签
        
        Args:
            y_score (torch.Tensor or np.ndarray): 预测概率分数
            threshold (float): 分类阈值
            
        Returns:
            np.ndarray: 二值化的预测标签
        """
        y_score_np = PPIMetrics._to_numpy(y_score)
        
        # 对于二分类情况，确保y_score是概率向量
        if len(y_score_np.shape) == 2 and y_score_np.shape[1] > 1:
            y_score_np = y_score_np[:, 1]
        
        return (y_score_np >= threshold).astype(int)
    
    @staticmethod
    def _to_numpy(tensor_or_array):
        """
        将输入转换为numpy数组
        
        Args:
            tensor_or_array (torch.Tensor or np.ndarray): 输入张量或数组
            
        Returns:
            np.ndarray: 转换后的numpy数组
        """
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.cpu().detach().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        else:
            raise TypeError(f"输入类型不支持: {type(tensor_or_array)}")
    
    @classmethod
    def compute_all_metrics(cls, y_true, y_score, threshold=0.5):
        """
        计算所有评估指标
        
        Args:
            y_true (torch.Tensor or np.ndarray): 真实标签
            y_score (torch.Tensor or np.ndarray): 预测概率分数
            threshold (float): 分类阈值
            
        Returns:
            dict: 包含所有指标的字典
        """
        # 获取二值化预测
        y_pred = cls.get_thresholded_predictions(y_score, threshold)
        
        metrics = {
            "accuracy": cls.calculate_accuracy(y_true, y_pred),
            "precision": cls.calculate_precision(y_true, y_pred),
            "recall": cls.calculate_recall(y_true, y_pred),
            "f1": cls.calculate_f1(y_true, y_pred),
            "roc_auc": cls.calculate_roc_auc(y_true, y_score),
            "average_precision": cls.calculate_average_precision(y_true, y_score),
            "confusion_matrix": cls.calculate_confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    @classmethod
    def log_metrics(cls, metrics, phase="validation", epoch=None):
        """
        记录指标到日志
        
        Args:
            metrics (dict): 包含指标的字典
            phase (str): 阶段名称 ('training', 'validation', 'testing')
            epoch (int, optional):  epoch编号
        """
        if epoch is not None:
            logger.info(f"Epoch {epoch} {phase} 指标:")
        else:
            logger.info(f"{phase} 指标:")
        
        for key, value in metrics.items():
            if key != "confusion_matrix":
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  混淆矩阵: {value}")


def calculate_metrics(y_true, y_score, threshold=0.5):
    """
    计算所有评估指标的包装函数
    这是为了向后兼容，提供一个直接的函数接口
    
    Args:
        y_true (torch.Tensor or np.ndarray): 真实标签
        y_score (torch.Tensor or np.ndarray): 预测概率分数
        threshold (float): 分类阈值
        
    Returns:
        dict: 包含所有指标的字典
    """
    return PPIMetrics.compute_all_metrics(y_true, y_score, threshold)