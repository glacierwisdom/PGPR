import os
import sys
import time
import argparse
import logging
import json
import random
import numpy as np
import torch
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.config_manager import ConfigManager
from graph.builder import PPIGraphBuilder
from models.gnn_ppi import GNN_PPI
from llm.wrapper import LLMWrapper
from training.trainer import ExploratoryPPITrainer
from evaluation.evaluator import PPIEvaluator
from utils.logger import setup_logger

def setup_environment(config):
    """
    设置超参数调优环境
    
    Args:
        config (ConfigManager): 配置管理器
    """
    # 设置日志
    setup_logger(config['logging']['level'], config['logging']['log_file'])
    
    # 设置随机种子
    random.seed(config['seed']['random_seed'])
    np.random.seed(config['seed']['numpy_seed'])
    torch.manual_seed(config['seed']['torch_seed'])
    torch.cuda.manual_seed_all(config['seed']['torch_seed'])
    
    # 设置设备
    device = config['device']['device_type']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA不可用，将使用CPU")
        device = 'cpu'
        config['device']['device_type'] = device
    
    return device

def prepare_data(config):
    """
    准备超参数调优所需的数据
    
    Args:
        config (ConfigManager): 配置管理器
        
    Returns:
        tuple: 训练数据、验证数据
    """
    logging.info("开始准备数据...")
    
    # 初始化图构建器
    graph_builder = PPIGraphBuilder(
        data_dir=os.path.join(config['paths']['data_dir'], 'processed'),
        use_blast=config['preprocessing']['graph']['use_blast'],
        num_neighbors=config['preprocessing']['graph']['num_neighbors'],
        max_path_length=config['preprocessing']['graph']['max_path_length'],
        device=config['device']['device_type']
    )
    
    # 构建图和加载数据集
    train_data = graph_builder.build_graph_and_load_data(
        split='train',
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['data_loader']['shuffle'],
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    val_data = graph_builder.build_graph_and_load_data(
        split='val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['data_loader']['num_workers']
    )
    
    logging.info("数据准备完成")
    return train_data, val_data, graph_builder

def evaluate_hyperparameters(hyperparams: Dict, base_config: Dict, train_data, val_data, graph_builder, max_epochs: int = 20) -> float:
    """
    评估一组超参数的性能
    
    Args:
        hyperparams (Dict): 超参数配置
        base_config (Dict): 基础配置
        train_data: 训练数据
        val_data: 验证数据
        graph_builder: 图构建器
        max_epochs (int): 最大训练轮数
        
    Returns:
        float: 验证集上的F1分数
    """
    logging.info(f"评估超参数: {hyperparams}")
    
    # 合并基础配置和超参数
    config = base_config.copy()
    
    # 更新超参数
    for param_path, value in hyperparams.items():
        # 支持嵌套参数路径，如 'model.hidden_size'
        keys = param_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    # 初始化模型
    model = GNN_PPI(config)
    model = model.to(config['device']['device_type'])
    
    # 初始化LLM包装器
    llm_wrapper = LLMWrapper(
        model_name=config['llm']['model_name'],
        tokenizer_name=config['llm']['tokenizer_name'],
        use_lora=config['llm']['lora']['use_lora'],
        lora_r=config['llm']['lora']['r'],
        lora_alpha=config['llm']['lora']['lora_alpha'],
        lora_dropout=config['llm']['lora']['lora_dropout'],
        target_modules=config['llm']['lora']['target_modules'],
        use_quantization=config['llm']['quantization']['use_quantization'],
        bits=config['llm']['quantization']['bits'],
        device=config['device']['device_type']
    )
    
    # 初始化训练器
    trainer = ExploratoryPPITrainer(
        model=model,
        llm_wrapper=llm_wrapper,
        graph_builder=graph_builder,
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        config=config
    )
    
    # 训练模型（使用较少的轮数）
    logging.info(f"开始训练模型，最大轮数: {max_epochs}")
    start_time = time.time()
    
    try:
        best_model = trainer.train(
            num_epochs=max_epochs,
            early_stopping_patience=5,
            verbose=False
        )
        
        # 评估模型
        logging.info("评估模型...")
        val_metrics = trainer.evaluate(val_data, best_model)
        f1_score = val_metrics.get('f1_score', 0)
        
        logging.info(f"超参数评估完成，F1分数: {f1_score:.4f}，耗时: {time.time() - start_time:.2f}秒")
        
        return f1_score
        
    except Exception as e:
        logging.error(f"超参数评估失败: {str(e)}", exc_info=True)
        return 0

class BayesianOptimizer:
    """
    贝叶斯优化器，用于自动调优超参数
    """
    
    def __init__(self, param_space: Dict, base_config: Dict, n_initial_points: int = 10, n_iterations: int = 50):
        """
        初始化贝叶斯优化器
        
        Args:
            param_space (Dict): 超参数搜索空间
            base_config (Dict): 基础配置
            n_initial_points (int): 初始随机采样点数
            n_iterations (int): 优化迭代次数
        """
        self.param_space = param_space
        self.base_config = base_config
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        
        # 结果存储
        self.trials = []
        self.best_params = None
        self.best_score = -1
        
    def _sample_random_params(self) -> Dict:
        """
        从参数空间中随机采样一组超参数
        
        Returns:
            Dict: 采样的超参数
        """
        params = {}
        
        for param_name, space in self.param_space.items():
            param_type = space['type']
            
            if param_type == 'int':
                params[param_name] = random.randint(space['min'], space['max'])
            
            elif param_type == 'float':
                params[param_name] = random.uniform(space['min'], space['max'])
            
            elif param_type == 'logfloat':
                # 对数尺度采样
                min_log = np.log10(space['min'])
                max_log = np.log10(space['max'])
                value_log = random.uniform(min_log, max_log)
                params[param_name] = 10 ** value_log
            
            elif param_type == 'choice':
                params[param_name] = random.choice(space['choices'])
            
            elif param_type == 'bool':
                params[param_name] = random.choice([True, False])
        
        return params
    
    def _gp_predict(self, X: np.ndarray, y: np.ndarray, x_new: np.ndarray):
        """
        使用高斯过程模型预测新参数的性能
        
        Args:
            X (np.ndarray): 训练数据特征
            y (np.ndarray): 训练数据目标
            x_new (np.ndarray): 新参数向量
            
        Returns:
            tuple: 预测均值和方差
        """
        # 使用简单的径向基函数核
        def rbf_kernel(X1, X2, length_scale=1.0, signal_variance=1.0):
            X1 = np.array(X1)
            X2 = np.array(X2)
            if X1.ndim == 1:
                X1 = X1.reshape(-1, 1)
            if X2.ndim == 1:
                X2 = X2.reshape(-1, 1)
            
            dist_sq = (np.sum(X1 ** 2, axis=1).reshape(-1, 1) + 
                      np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T))
            
            return signal_variance * np.exp(-0.5 * dist_sq / (length_scale ** 2))
        
        # 训练GP模型
        K = rbf_kernel(X, X)
        K += np.eye(len(K)) * 1e-10  # 添加噪声
        K_inv = np.linalg.inv(K)
        
        # 预测
        k_new = rbf_kernel(X, x_new)
        k_new_new = rbf_kernel(x_new, x_new)
        
        mu = np.dot(np.dot(k_new.T, K_inv), y)
        sigma = k_new_new - np.dot(np.dot(k_new.T, K_inv), k_new)
        
        return mu[0], sigma[0, 0]
    
    def _acquisition_function(self, mu: float, sigma: float, y_max: float, acquisition_type: str = 'ei') -> float:
        """
        计算获取函数值
        
        Args:
            mu (float): 预测均值
            sigma (float): 预测方差
            y_max (float): 当前最佳分数
            acquisition_type (str): 获取函数类型 ('ei', 'ucb', 'poi')
            
        Returns:
            float: 获取函数值
        """
        if sigma <= 1e-9:
            return 0
        
        if acquisition_type == 'ei':
            # 期望改进 (Expected Improvement)
            z = (mu - y_max - 0.01) / sigma
            return (mu - y_max - 0.01) * norm.cdf(z) + sigma * norm.pdf(z)
        
        elif acquisition_type == 'ucb':
            # 置信上限 (Upper Confidence Bound)
            return mu + 2.0 * np.sqrt(sigma)
        
        elif acquisition_type == 'poi':
            # 改进概率 (Probability of Improvement)
            z = (mu - y_max - 0.01) / sigma
            return norm.cdf(z)
        
        return 0
    
    def _discretize_space(self, n_samples: int = 100) -> List[Dict]:
        """
        离散化参数空间用于获取函数评估
        
        Args:
            n_samples (int): 采样点数
            
        Returns:
            List[Dict]: 离散化的参数点列表
        """
        samples = []
        
        for _ in range(n_samples):
            samples.append(self._sample_random_params())
        
        return samples
    
    def _encode_params(self, params: Dict) -> np.ndarray:
        """
        将参数字典编码为数值向量
        
        Args:
            params (Dict): 参数字典
            
        Returns:
            np.ndarray: 编码后的参数向量
        """
        encoded = []
        
        for param_name, space in self.param_space.items():
            value = params[param_name]
            param_type = space['type']
            
            if param_type in ['int', 'float', 'logfloat']:
                # 归一化到[0, 1]范围
                min_val = space['min']
                max_val = space['max']
                if param_type == 'logfloat':
                    # 对数尺度归一化
                    value = np.log10(value)
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                
                normalized = (value - min_val) / (max_val - min_val)
                encoded.append(normalized)
            
            elif param_type == 'choice':
                # 独热编码
                choices = space['choices']
                one_hot = [0.0] * len(choices)
                idx = choices.index(value)
                one_hot[idx] = 1.0
                encoded.extend(one_hot)
            
            elif param_type == 'bool':
                # 布尔值编码为0或1
                encoded.append(1.0 if value else 0.0)
        
        return np.array(encoded)
    
    def optimize(self, train_data, val_data, graph_builder):
        """
        运行贝叶斯优化
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            graph_builder: 图构建器
        """
        logging.info("开始贝叶斯优化...")
        logging.info(f"初始随机采样点: {self.n_initial_points}")
        logging.info(f"优化迭代次数: {self.n_iterations}")
        
        # 1. 初始随机采样
        for i in range(self.n_initial_points):
            logging.info(f"\n初始采样点 {i+1}/{self.n_initial_points}")
            
            # 随机采样参数
            params = self._sample_random_params()
            
            # 评估参数
            score = evaluate_hyperparameters(
                params, self.base_config, train_data, val_data, graph_builder
            )
            
            # 记录结果
            self.trials.append({
                'params': params,
                'score': score,
                'iteration': i,
                'type': 'random'
            })
            
            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logging.info(f"新的最佳参数: {params}")
                logging.info(f"新的最佳分数: {score:.4f}")
        
        # 2. 贝叶斯优化迭代
        for i in range(self.n_iterations):
            logging.info(f"\n贝叶斯优化迭代 {i+1}/{self.n_iterations}")
            
            if len(self.trials) < 2:
                # 数据点不足时继续随机采样
                params = self._sample_random_params()
            
            else:
                # 构建训练数据
                X = np.array([self._encode_params(trial['params']) for trial in self.trials])
                y = np.array([trial['score'] for trial in self.trials])
                
                # 离散化参数空间
                candidate_params = self._discretize_space(n_samples=100)
                
                # 计算每个候选点的获取函数值
                best_val = max(y)
                candidate_scores = []
                
                for candidate in candidate_params:
                    x_encoded = self._encode_params(candidate)
                    
                    # 使用GP预测
                    try:
                        mu, sigma = self._gp_predict(X, y, x_encoded)
                        
                        # 计算获取函数值
                        acq_score = self._acquisition_function(mu, sigma, best_val, acquisition_type='ei')
                        candidate_scores.append((candidate, acq_score))
                    except Exception as e:
                        logging.warning(f"获取函数计算失败: {e}")
                        candidate_scores.append((candidate, 0))
                
                # 选择获取函数值最大的参数
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                params = candidate_scores[0][0]
            
            # 评估选择的参数
            score = evaluate_hyperparameters(
                params, self.base_config, train_data, val_data, graph_builder
            )
            
            # 记录结果
            self.trials.append({
                'params': params,
                'score': score,
                'iteration': self.n_initial_points + i,
                'type': 'bayesian'
            })
            
            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logging.info(f"新的最佳参数: {params}")
                logging.info(f"新的最佳分数: {score:.4f}")
            
            # 定期保存中间结果
            if (i + 1) % 5 == 0:
                self.save_results(f"artifacts/hyperparameter_tuning/iter_{i+1}")
    
    def save_results(self, output_dir: str):
        """
        保存超参数调优结果
        
        Args:
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有试验结果
        trials_path = os.path.join(output_dir, 'trials.json')
        with open(trials_path, 'w', encoding='utf-8') as f:
            json.dump(self.trials, f, indent=4, ensure_ascii=False, default=str)
        
        # 保存最佳参数
        best_path = os.path.join(output_dir, 'best_params.json')
        with open(best_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score
            }, f, indent=4, ensure_ascii=False, default=str)
        
        logging.info(f"超参数调优结果已保存到: {output_dir}")

def get_default_param_space() -> Dict:
    """
    获取默认的超参数搜索空间
    
    Returns:
        Dict: 超参数搜索空间
    """
    return {
        # 优化器参数
        'optimizer.lr': {
            'type': 'logfloat',
            'min': 1e-6,
            'max': 1e-3
        },
        
        'optimizer.weight_decay': {
            'type': 'logfloat',
            'min': 1e-6,
            'max': 1e-3
        },
        
        # GNN_PPI模型参数
        'model.gnn_ppi.node_representation.hidden_dim': {
            'type': 'int',
            'min': 128,
            'max': 512
        },
        
        'model.gnn_ppi.target_attention.num_heads': {
            'type': 'int',
            'min': 4,
            'max': 16
        },
        
        'model.gnn_ppi.graph_attention.num_layers': {
            'type': 'int',
            'min': 2,
            'max': 6
        },
        
        'model.gnn_ppi.dropout': {
            'type': 'float',
            'min': 0.0,
            'max': 0.5
        },
        
        # 强化学习参数
        'model.rl.ppo.clip_ratio': {
            'type': 'float',
            'min': 0.1,
            'max': 0.3
        },
        
        'model.rl.ppo.ent_coef': {
            'type': 'logfloat',
            'min': 1e-5,
            'max': 1e-2
        },
        
        'model.rl.ppo.gamma': {
            'type': 'float',
            'min': 0.95,
            'max': 0.999
        },
        
        # 训练参数
        'training.batch_size': {
            'type': 'int',
            'min': 16,
            'max': 128
        },
        
        # 损失函数参数
        'loss.alpha': {
            'type': 'float',
            'min': 0.1,
            'max': 0.5
        },
        
        'loss.gamma': {
            'type': 'float',
            'min': 1.0,
            'max': 3.0
        },
        
        # 数据增强参数
        'preprocessing.augmentation.enable': {
            'type': 'bool'
        },
        
        # 图构建参数
        'preprocessing.graph.num_neighbors': {
            'type': 'int',
            'min': 5,
            'max': 20
        },
        
        'preprocessing.graph.max_path_length': {
            'type': 'int',
            'min': 3,
            'max': 10
        }
    }

def run_hyperparameter_tuning():
    """运行超参数调优
    
    这是hyperparameter_main函数的别名，用于保持与其他实验脚本的一致性。
    """
    hyperparameter_main()

if __name__ == "__main__":
    hyperparameter_main()
