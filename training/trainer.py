import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
import time
import sys
import faulthandler
import signal
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from graph.builder import PPIGraphBuilder
from utils.protein_mapper import ProteinSimilarityMapper, compute_non_isolated_ids
from models.cot_generator import ExploratoryCOTGenerator
from models.component_builder import ComponentBuilder
from models.node_representations import NodeRepresentation
from models.attention_mechanism import TargetConditionedAttention
from models.rnn_encoder import PathRNNEncoder
from models.gnn_ppi import GNN_PPI
from models.esm_encoder import ESMEncoder

from llm.prompt_designer import PromptDesigner
from llm.wrapper import LLMWrapper
from llm.utils import LLMUtils

from training.rl_framework import PPOTrainer
from training.reward_calculator import MultiScaleRewardCalculator
from training.callback import TrainingCallback, get_callbacks

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class ExploratoryPPITrainer:
    """
    探索式PPI训练器
    整合所有训练组件：图构建、COT生成、LLM、强化学习等
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config (Dict[str, Any]): 配置字典
        """
        self.config = config
        self.device = self._get_device()
        if str(getattr(self.device, "type", "")).lower() == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        amp_cfg = config.get('amp', {}) or {}
        self.use_amp = bool(amp_cfg.get('use_amp', config.get('training', {}).get('use_amp', False)))
        self.multi_gpu = config.get('distributed', {}).get('use_distributed', False)
        if not hasattr(self, "rank"):
            self.rank = 0
        if not hasattr(self, "world_size"):
            self.world_size = 1
        
        # 初始化组件
        self.graph_builder = None
        self.cot_generator = None
        self.llm_wrapper = None
        self.prompt_designer = None
        self.reward_calculator = None
        self.ppo_trainer = None
        
        # 初始化训练相关
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # 指标监控
        self.history = {
            'loss': [],
            'reward': [],
            'accuracy': []
        }
        
        # 初始化回调
        self.callbacks = get_callbacks(config)
        
        logger.info(f"ExploratoryPPITrainer初始化完成，使用设备：{self.device}")

    def _plot_metrics(self):
        """
        在终端绘制训练指标曲线
        """
        if not sys.stdout.isatty():
            return
        if getattr(self, "multi_gpu", False):
            return
        if not self.history['loss']:
            return
            
        # 只保留最近100个点，避免图表过长
        plot_history = {
            'loss': self.history['loss'][-100:],
            'reward': self.history['reward'][-100:],
            'accuracy': self.history['accuracy'][-100:]
        }
            
        print("\n" + "="*50)
        print("训练实时监控 (最近100步)")
        print("="*50)
        
        metrics = [
            ('Loss', plot_history['loss'], 'r'),
            ('Reward', plot_history['reward'], 'g'),
            ('Accuracy', plot_history['accuracy'], 'b')
        ]
        
        for name, data, color in metrics:
            if not data: continue
            print(f"\n{name}: {data[-1]:.4f}")
            self._ascii_chart(data)
        print("="*50 + "\n")

    def _ascii_chart(self, data, height=10, width=60):
        """
        简单的ASCII图表绘制
        """
        if not data:
            return

        import math
        data = [d for d in data if isinstance(d, (int, float)) and math.isfinite(d)]
        if not data:
            return
        
        # 采样数据以适应宽度
        if len(data) > width:
            indices = [int(i * (len(data)-1) / (width-1)) for i in range(width)]
            sampled_data = [data[i] for i in indices]
        else:
            sampled_data = data
            
        min_val = min(data)
        max_val = max(data)
        val_range = max_val - min_val if max_val != min_val else 1.0
        
        chart = [[' ' for _ in range(len(sampled_data))] for _ in range(height)]
        
        for x, val in enumerate(sampled_data):
            y = int((val - min_val) / val_range * (height - 1))
            chart[height - 1 - y][x] = '*'
            
        for row in chart:
            print(''.join(row))
        print(f"Min: {min_val:.4f} | Max: {max_val:.4f}")
    
    def _load_data(self, split: str) -> Any:
        """
        加载数据
        
        Args:
            split (str): 数据分割 ('train', 'val', 'test')
            
        Returns:
            Any: 数据
        """
        logger.info(f"正在为 {split} 加载数据...")
        
        # 优先使用配置中的 processed_data_dir，如果没有则使用 data_dir/processed
        data_dir = self.config['paths'].get('processed_data_dir')
        if not data_dir:
            data_dir = os.path.join(self.config['paths']['data_dir'], 'processed')
            
        # 统一维度设置
        esm_dim_value = self.config.get('gnn_ppi', {}).get('node_representation', {}).get('feature_dim', 320)
        edge_dim_value = self.config.get('model', {}).get('num_edge_features', 64)
            
        if not hasattr(self, 'graph_builder') or self.graph_builder is None:
            self.graph_builder = PPIGraphBuilder(
                esm_dim=esm_dim_value,
                edge_dim=edge_dim_value,
                device=self.device,
                data_dir=data_dir,
                use_blast=self.config['preprocessing']['graph']['use_blast'],
                num_neighbors=self.config['preprocessing']['graph']['num_neighbors'],
                max_path_length=self.config['preprocessing']['graph']['max_path_length']
            )
        
        data_file = self.config['dataset'].get(f'{split}_file')
        
        # 构建图和加载数据集
        # 注意：这里只返回数据集对象，DataLoader由 _create_dataloader 在 train() 中创建
        _, dataset = self.graph_builder.build_graph_and_load_data(
            split=split,
            data_file=data_file,
            batch_size=self.config['training']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=self.config['training'].get('data_loader', {}).get('num_workers', 0)
        )
        
        return dataset
     
    def _get_device(self) -> str:
        """
        获取设备
        
        Returns:
            str: 设备字符串
        """
        device_cfg = (self.config.get('device', {}) or {}).get('device_type', 'cuda')
        device_cfg = str(device_cfg).lower()

        if device_cfg == 'cpu' or device_cfg.startswith('cpu'):
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        if getattr(self, "multi_gpu", False) and hasattr(self, "rank"):
            return f"cuda:{int(self.rank)}"

        device_ids = (self.config.get('device', {}) or {}).get('device_ids')
        if isinstance(device_ids, (list, tuple)) and len(device_ids) > 0:
            try:
                return f"cuda:{int(device_ids[0])}"
            except Exception:
                return "cuda:0"

        return "cuda:0"

    def build_components(self):
        """
        构建所有组件
        """
        logger.info("开始构建训练组件...")
        
        # 1. 构建图构建器
        self._build_graph_builder()
        
        # 2. 构建模型组件
        self._build_model_components()
        
        # 3. 构建LLM组件
        self._build_llm_components()
        
        # 4. 构建奖励计算器
        self._build_reward_calculator()
        
        # 5. 构建PPO训练器
        self._build_ppo_trainer()
        
        # 6. 构建优化器和学习率调度器
        self._build_optimizer_and_scheduler()
        
        logger.info("所有训练组件构建完成")
    
    def _build_graph_builder(self):
        """
        构建图构建器
        """
        self.graph_builder = ComponentBuilder.build_graph_builder(self.config)
    
    def _build_model_components(self):
        """
        构建模型组件
        """
        # 构建ESM编码器
        esm_model_name = self.config.get('dataset', {}).get('preprocessing', {}).get('feature_extraction', {}).get('esm_model', 'facebook/esm2_t6_8M_UR50D')
        self.esm_encoder = ESMEncoder(model_name=esm_model_name, device=self.device)
        
        self.cot_generator = ComponentBuilder.build_cot_generator(self.config, self.device)
    
    def _build_llm_components(self):
        """
        构建LLM组件
        """
        # 传入当前设备，确保在分布式模式下加载到正确的 GPU
        device_str = str(self.device) if hasattr(self, 'device') else None
        self.prompt_designer, self.llm_wrapper = ComponentBuilder.build_llm_components(self.config, device=device_str)
        
        # 加载蛋白质信息（用于提示生成）
        data_dir = self.config.get('paths', {}).get('data_dir', 'data')
        info_file = self.config.get('dataset', {}).get('protein_info_file', 'protein_info.csv')
        # 补充：通常这些文件都在 data/processed 下
        info_path = os.path.join(data_dir, 'processed', info_file)
        
        if not os.path.exists(info_path):
            # 备选路径：直接在 data_dir 下
            info_path = os.path.join(data_dir, info_file)
        
        if os.path.exists(info_path):
            logger.info(f"正在从 {info_path} 加载蛋白质补充信息...")
            self.prompt_designer.load_protein_info(info_path)
        else:
            logger.warning(f"未找到蛋白质信息文件: {info_path}")
    
    def _build_reward_calculator(self):
        """
        构建奖励计算器
        """
        reward_config = self.config.get('reward', {})
        self.reward_calculator = MultiScaleRewardCalculator(
            accuracy_reward=reward_config.get('accuracy_reward', 2.0),
            accuracy_penalty=reward_config.get('accuracy_penalty', -1.0),
            length_penalty=reward_config.get('length_penalty', -0.1),
            diversity_bonus=reward_config.get('diversity_bonus', 0.05),
            distance_factor=reward_config.get('distance_factor', 0.5),
            edge_feature_weight=reward_config.get('edge_feature_weight', 0.2)
        )
        logger.info("奖励计算器已构建完成")
    
    def _build_ppo_trainer(self):
        """
        构建PPO训练器
        """
        rl_config = dict(self.config.get('reinforcement_learning', {}) or {})
        legacy_rl = self.config.get('rl', {}) or {}
        ppo_cfg = legacy_rl.get('ppo', {}) if isinstance(legacy_rl, dict) else {}
        if isinstance(ppo_cfg, dict) and ppo_cfg:
            if 'clip_param' not in rl_config:
                rl_config['clip_param'] = ppo_cfg.get('clip_ratio', ppo_cfg.get('clip_param', 0.2))
            if 'epochs' not in rl_config:
                rl_config['epochs'] = ppo_cfg.get('num_epochs', ppo_cfg.get('epochs', 10))
            if 'batch_size' not in rl_config:
                rl_config['batch_size'] = ppo_cfg.get('minibatch_size', ppo_cfg.get('batch_size', 64))
            if 'gamma' not in rl_config:
                rl_config['gamma'] = ppo_cfg.get('gamma', 0.99)
            if 'gae_lambda' not in rl_config:
                rl_config['gae_lambda'] = ppo_cfg.get('lam', ppo_cfg.get('gae_lambda', 0.95))
            if 'entropy_coef' not in rl_config:
                rl_config['entropy_coef'] = ppo_cfg.get('ent_coef', 0.01)
            if 'value_loss_coef' not in rl_config:
                rl_config['value_loss_coef'] = ppo_cfg.get('vf_coef', ppo_cfg.get('value_loss_coef', 0.5))
            if 'max_grad_norm' not in rl_config:
                rl_config['max_grad_norm'] = ppo_cfg.get('max_grad_norm', 0.5)
            if 'learning_rate' not in rl_config:
                rl_config['learning_rate'] = ppo_cfg.get('learning_rate', ppo_cfg.get('lr', 3e-4))
        
        # 使用COT生成器作为策略网络
        policy_model = self.cot_generator
        
        # 使用一个简单的MLP作为价值网络
        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        # 价值网络输入维度应为 source_node_dim + target_esm_dim
        # 以便同时感知起点和终点
        model_config = self.config.get('model', {})
        node_rep_config = self.config.get('gnn_ppi', {}).get('node_representation', {})
        
        node_dim = model_config.get('embedding_dim', 256)
        esm_dim = node_rep_config.get('feature_dim', node_rep_config.get('esm_dim', model_config.get('num_node_features', 1280)))
        
        value_model = ValueNetwork(
            input_dim=node_dim + esm_dim,
            hidden_dim=model_config.get('hidden_dim', 128)
        )
        
        self.ppo_trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            learning_rate=rl_config.get('learning_rate', 3e-4),
            clip_param=rl_config.get('clip_param', 0.2),
            epochs=rl_config.get('epochs', 10),
            batch_size=rl_config.get('batch_size', 64),
            gamma=rl_config.get('gamma', 0.99),
            gae_lambda=rl_config.get('gae_lambda', 0.95),
            entropy_coef=rl_config.get('entropy_coef', 0.01),
            value_loss_coef=rl_config.get('value_loss_coef', 0.5),
            max_grad_norm=rl_config.get('max_grad_norm', 0.5),
            use_amp=bool(rl_config.get('use_amp', False)),
            device=self.device
        )
        
        logger.info("PPO训练器已构建完成")
    
    def _build_optimizer_and_scheduler(self):
        """
        构建优化器和学习率调度器
        """
        optim_config = self.config.get('optimizer', {})
        lr_config = self.config.get('lr_scheduler', {})
        
        # 获取要通过主优化器训练的参数
        # 注意：cot_generator 由 PPOTrainer 管理自己的优化器
        # 这里的主优化器主要用于 LLM 微调或其他模块
        params = []
        llm_cfg = self.config.get('llm', {}) or {}
        train_llm = bool(llm_cfg.get('train_llm', False))
        train_backbone = bool(llm_cfg.get('train_backbone', False))
        if train_llm and self.llm_wrapper:
            logger.info("将 LLM 关系头参数添加到主优化器")
            params += list(self.llm_wrapper.relation_classifier.parameters())
            if self.llm_wrapper.model and train_backbone:
                logger.info("将 LLM backbone 参数添加到主优化器")
                params += [p for p in self.llm_wrapper.model.parameters() if p.requires_grad]
            elif self.llm_wrapper.model and not train_backbone:
                if bool(getattr(self.llm_wrapper, "use_lora", False)):
                    logger.info("将 LLM LoRA 参数添加到主优化器")
                    params += [p for p in self.llm_wrapper.model.parameters() if p.requires_grad]
                else:
                    for p in self.llm_wrapper.model.parameters():
                        p.requires_grad = False
        
        # 如果没有要训练的参数，则创建一个空的优化器列表
        if not params:
            logger.info("没有额外的参数需要通过主优化器训练")
            # 即使没有参数也创建一个虚拟优化器以避免后续代码报错
            # 或者在训练循环中检查 self.optimizer 是否为 None
            self.optimizer = None
            self.scheduler = None
            return
        
        # 构建优化器
        optimizer_type = optim_config.get('type', 'adam')
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=float(optim_config.get('learning_rate', optim_config.get('lr', 1e-4))),
                betas=tuple(optim_config.get('betas', (0.9, 0.999))),
                weight_decay=float(optim_config.get('weight_decay', 0.01))
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=float(optim_config.get('learning_rate', optim_config.get('lr', 1e-4))),
                betas=tuple(optim_config.get('betas', (0.9, 0.999))),
                weight_decay=float(optim_config.get('weight_decay', 0.01))
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=float(optim_config.get('learning_rate', optim_config.get('lr', 1e-4))),
                momentum=float(optim_config.get('momentum', 0.9)),
                weight_decay=float(optim_config.get('weight_decay', 0.01))
            )
        
        # 构建学习率调度器
        def _to_int(v, default: int) -> int:
            if v is None:
                return default
            if isinstance(v, bool):
                return default
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(str(v)))
                except Exception:
                    return default

        def _to_float(v, default: float) -> float:
            if v is None:
                return default
            if isinstance(v, bool):
                return default
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v).strip())
                except Exception:
                    return default

        scheduler_type = lr_config.get('type', 'none')
        scheduler_type = str(scheduler_type or 'none').lower()
        if scheduler_type in ("cosine_annealing", "cosineannealing", "cosineannealinglr", "cosine_annealinglr"):
            scheduler_type = "cosine"
        if scheduler_type in ("reduceonplateau", "reduce_on_plateau_lr", "reduce_on_plateau"):
            scheduler_type = "reduce_on_plateau"
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=_to_int(lr_config.get('step_size', 10), 10),
                gamma=_to_float(lr_config.get('gamma', 0.1), 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=_to_int(lr_config.get('T_max', self.config.get('training', {}).get('epochs', 100)), 100),
                eta_min=_to_float(lr_config.get('eta_min', 0.0), 0.0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=lr_config.get('mode', 'min'),
                factor=_to_float(lr_config.get('factor', 0.1), 0.1),
                patience=_to_int(lr_config.get('patience', 10), 10),
                threshold=_to_float(lr_config.get('threshold', 1e-4), 1e-4)
            )
        
        logger.info("优化器和学习率调度器已构建完成")
    
    def train(self, train_data: Any, val_data: Optional[Any] = None, resume_checkpoint: Optional[str] = None):
        """
        开始训练
        
        Args:
            train_data (Any): 训练数据 (Data, Dataset) tuple or Dataset
            val_data (Optional[Any]): 验证数据
        """
        # 初始化
        self.build_components()

        if resume_checkpoint:
            ckpt_path = resume_checkpoint
            if os.path.exists(ckpt_path):
                logger.info(f"从检查点加载模型权重: {ckpt_path}")
                try:
                    try:
                        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(ckpt_path, map_location="cpu")

                    model_state = checkpoint.get('model_state_dict', checkpoint)
                    self.cot_generator.load_state_dict(model_state, strict=False)

                    llm_cls_state = checkpoint.get('llm_classifier_state_dict')
                    if llm_cls_state is not None and self.llm_wrapper is not None:
                        self.llm_wrapper.relation_classifier.load_state_dict(llm_cls_state, strict=False)

                    logger.info("检查点权重加载完成")
                except Exception as e:
                    logger.error(f"加载检查点失败: {ckpt_path} ({e})", exc_info=True)
                    raise
            else:
                logger.warning(f"检查点文件不存在，跳过加载: {ckpt_path}")
        
        # Unpack train_data if it is a tuple (graph, dataset)
        if isinstance(train_data, tuple):
            self.graph, train_dataset = train_data
            # Move graph to device
            self.graph = self.graph.to(self.device)
        else:
            train_dataset = train_data
        
        # Unpack val_data if it is a tuple
        val_dataset = None
        if val_data:
            if isinstance(val_data, tuple):
                _, val_dataset = val_data
            else:
                val_dataset = val_data
        
        # 获取配置
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        batch_size = training_config.get('batch_size', 32)
        
        # 构建全局图 (if not provided in tuple)
        if (not hasattr(self, 'graph') or self.graph is None) and train_dataset:
            logger.info("使用 GraphBuilder 构建全局图 (包含背景边和序列映射)...")
            # 优先从配置获取训练文件路径
            train_file = self.config.get('dataset', {}).get('train_file')
            # 如果没有，尝试使用默认值
            if not train_file:
                train_file = "data/shs27k_llapa/processed/bfs_train.tsv"
            
            try:
                self.graph, _ = self.graph_builder.build_graph_and_load_data(
                    split='train', 
                    data_file=train_file
                )
                self.graph = self.graph.to(self.device)
                logger.info(f"全局图构建成功: {self.graph.x.size(0)} 节点, {self.graph.edge_index.size(1)} 边")
            except Exception as e:
                logger.error(f"通过 build_graph_and_load_data 构建图失败: {e}")
                # 回退到原始逻辑
                logger.info("回退到原始图构建逻辑...")
                protein_sequences = train_dataset.get_protein_sequences()
                protein_ids = list(protein_sequences.keys())
                # ... (后续逻辑保持不变，但这里应该尽量使用上面的新方法)

        # 创建数据加载器
        train_loader = self._create_dataloader(train_dataset, batch_size)
        val_loader = None
        if val_dataset:
            val_loader = self._create_dataloader(val_dataset, batch_size)
        
        # 根据配置决定是否进行验证
        do_validate = self.config.get('training', {}).get('validate', False)
        if not do_validate:
            val_loader = None
        
        # 设置进度条的实际总批次数和总epoch数
        actual_total_batches = len(train_loader)
        quick_steps = self.config.get('training', {}).get('quick_run_steps')
        if isinstance(quick_steps, int) and quick_steps > 0:
            actual_total_batches = min(actual_total_batches, quick_steps)
        for cb in self.callbacks:
            if hasattr(cb, 'total_batches'):
                cb.total_batches = actual_total_batches
            if hasattr(cb, 'total_epochs'):
                cb.total_epochs = epochs
        
        # 训练开始前回调
        self._on_train_begin()

        self._train_protein_sequences = {}
        if train_dataset and hasattr(train_dataset, "get_protein_sequences"):
            self._train_protein_sequences = train_dataset.get_protein_sequences()

        self._non_isolated_protein_ids = set()
        if hasattr(self, "graph") and self.graph is not None:
            self._non_isolated_protein_ids = compute_non_isolated_ids(self.graph)

        sim_cfg = (self.config.get("preprocessing", {}) or {}).get("similarity_mapping", {}) or self.config.get("similarity_mapping", {}) or {}
        sim_enabled = bool(sim_cfg.get("enabled", True))
        sim_batch_size = int(sim_cfg.get("batch_size", 16))
        sim_method = str(sim_cfg.get("method", "esm"))
        sim_allow_fallback_to_esm = bool(sim_cfg.get("allow_fallback_to_esm", True))
        sim_blastp_evalue = float(sim_cfg.get("blastp_evalue", 1e-5))
        sim_blastp_num_threads = int(sim_cfg.get("blastp_num_threads", 4))
        sim_cache_dir = sim_cfg.get("cache_dir")
        if not sim_cache_dir:
            sim_cache_dir = os.path.join(self.config.get("paths", {}).get("data_dir", "."), "cache", "protein_similarity_mapper")

        self.protein_similarity_mapper = ProteinSimilarityMapper(
            esm_encoder=self.esm_encoder,
            cache_dir=sim_cache_dir,
            enabled=sim_enabled,
            batch_size=sim_batch_size,
            method=sim_method,
            allow_fallback_to_esm=sim_allow_fallback_to_esm,
            blastp_evalue=sim_blastp_evalue,
            blastp_num_threads=sim_blastp_num_threads,
        )
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            if self.rank == 0:
                self.protein_similarity_mapper.fit(
                    train_protein_sequences=self._train_protein_sequences,
                    non_isolated_ids=self._non_isolated_protein_ids,
                )
            dist.barrier()
            if self.rank != 0:
                self.protein_similarity_mapper.fit(
                    train_protein_sequences=self._train_protein_sequences,
                    non_isolated_ids=self._non_isolated_protein_ids,
                )
        else:
            self.protein_similarity_mapper.fit(
                train_protein_sequences=self._train_protein_sequences,
                non_isolated_ids=self._non_isolated_protein_ids,
            )

        self._ensure_graph_node_features()
        
        # 主训练循环
        quick_epochs = self.config.get('training', {}).get('quick_run_epochs')
        effective_epochs = epochs
        if isinstance(quick_epochs, int) and quick_epochs > 0:
            effective_epochs = min(effective_epochs, quick_epochs)
        for epoch in range(effective_epochs):
            # Epoch开始回调
            self._on_epoch_begin(epoch)
            if self.multi_gpu and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                try:
                    train_loader.sampler.set_epoch(epoch)
                except Exception:
                    pass
            
            # 训练一个epoch
            train_metrics = self._train_epoch(epoch, train_loader)
            
            # 验证
            val_metrics = None
            if val_loader:
                val_metrics = self._validate_epoch(epoch, val_loader)
            
            # 合并指标
            metrics = {
                **train_metrics,
                **(val_metrics if val_metrics else {})
            }
            
            # 更新学习率调度器
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get('val_loss', metrics.get('loss')))
            elif self.scheduler:
                self.scheduler.step()
            
            # Epoch结束回调
            self._on_epoch_end(epoch, metrics)
            
            # 检查早停
            if self._should_stop_training():
                logger.info("早停触发，训练提前结束")
                break
        
        # 训练结束回调
        self._on_train_end()
        
        logger.info("训练完成！")

    def _ensure_graph_node_features(self) -> None:
        if not hasattr(self, "graph") or self.graph is None:
            return
        if float(self.graph.x.abs().sum().detach().cpu().item()) > 0:
            return
        if not hasattr(self, "esm_encoder") or self.esm_encoder is None:
            return
        seqs = getattr(self, "_train_protein_sequences", {}) or {}
        protein_ids = getattr(self.graph, "protein_ids", None) or []
        if not protein_ids:
            return

        data_dir = (self.config.get("paths", {}) or {}).get("data_dir", "data")
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        safe_esm = str(getattr(self.esm_encoder, "model_name", "esm")).replace("/", "_").replace("\\", "_")
        cache_path = os.path.join(cache_dir, f"graph_x_{safe_esm}_{len(protein_ids)}.pt")

        dist_ready = bool(self.multi_gpu and dist.is_available() and dist.is_initialized())
        if dist_ready and self.rank != 0:
            dist.barrier()
            x = torch.load(cache_path, map_location="cpu")
            self.graph.x.copy_(x.to(self.graph.x.device))
            return
        if os.path.exists(cache_path):
            x = torch.load(cache_path, map_location="cpu")
            self.graph.x.copy_(x.to(self.graph.x.device))
            if dist_ready:
                dist.barrier()
            return

        indices = []
        sequences = []
        for pid in protein_ids:
            seq = seqs.get(pid)
            if not seq:
                continue
            idx = self.graph.protein_id_to_idx.get(pid)
            if idx is None:
                continue
            indices.append(int(idx))
            sequences.append(seq)

        if not sequences:
            return

        embeddings = self.esm_encoder.get_batch_embeddings(sequences, batch_size=16)
        x_cpu = torch.zeros((len(protein_ids), int(self.graph.x.size(-1))), dtype=torch.float32)
        for idx, emb in zip(indices, embeddings):
            if emb is None:
                continue
            x_cpu[idx] = emb.detach().to("cpu").float()
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(x_cpu, tmp_path)
        os.replace(tmp_path, cache_path)
        if dist_ready and self.rank == 0:
            dist.barrier()
        self.graph.x.copy_(x_cpu.to(self.graph.x.device))
    
    def _create_dataloader(self, data: Any, batch_size: int) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            data (Any): 数据
            batch_size (int): 批次大小
            
        Returns:
            DataLoader: 数据加载器
        """
        dl_cfg = (self.config.get('training', {}) or {}).get('data_loader', {}) or {}
        sampler = None
        shuffle = bool(dl_cfg.get('shuffle', True))
        num_workers = int(dl_cfg.get('num_workers', 0) or 0)
        pin_memory = bool(dl_cfg.get('pin_memory', False))
        drop_last = bool(dl_cfg.get('drop_last', False))

        kwargs = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(dl_cfg.get('persistent_workers', False))
            prefetch_factor = dl_cfg.get('prefetch_factor', None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)

        if self.multi_gpu:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                data,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )
    
    def _train_epoch(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch (int): 当前epoch
            dataloader (DataLoader): 数据加载器
            
        Returns:
            Dict[str, float]: 训练指标
        """
        self.cot_generator.train()
        if self.llm_wrapper and hasattr(self.llm_wrapper, 'model') and self.llm_wrapper.model:
            self.llm_wrapper.model.train()
        
        self._prompt_logged_this_epoch = False
        
        total_loss = 0
        total_reward = 0
        total_steps = 0
        
        start_time = time.time()
        
        from tqdm import tqdm
        show_pbar = sys.stdout.isatty() and not (getattr(self, "multi_gpu", False) and int(getattr(self, "rank", 0)) != 0)
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1} Train",
            disable=not show_pbar,
        )
        last_nonzero_loss = 0.0
        for batch_idx, batch in pbar:
            # Batch开始回调
            self._on_batch_begin(batch_idx)
            
            # 处理批次
            loss, reward = self._train_step(batch)
            
            if loss > 0:
                last_nonzero_loss = loss
                
            total_loss += loss
            total_reward += reward
            total_steps += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{last_nonzero_loss:.4f}" if last_nonzero_loss > 0 else "accumulating", 
                'reward': f"{reward:.4f}"
            })
            
            # Batch结束回调
            self._on_batch_end(batch_idx, {'loss': loss, 'reward': reward})
            
            # 每隔10个batch更新一次历史并绘图，以便更实时地观察变化
            if (batch_idx + 1) % 10 == 0 and show_pbar:
                self.history['loss'].append(loss)
                self.history['reward'].append(reward)
                self._plot_metrics()
            
            # 快速验证模式：限制每个epoch的批次数
            max_steps = self.config.get('training', {}).get('quick_run_steps')
            if isinstance(max_steps, int) and max_steps > 0 and (batch_idx + 1) >= max_steps:
                logger.info(f"达到快速运行步数限制 ({max_steps})，停止训练")
                break
        
        end_time = time.time()
        
        # 计算平均指标
        avg_loss = total_loss / total_steps
        avg_reward = total_reward / total_steps
        
        # 更新历史记录
        self.history['loss'].append(avg_loss)
        self.history['reward'].append(avg_reward)
        
        # 绘制曲线
        if show_pbar:
            self._plot_metrics()
        
        metrics = {
            'loss': avg_loss,
            'reward': avg_reward,
            'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
            'epoch_time': end_time - start_time
        }
        
        logger.info(f"Epoch {epoch+1} - 训练完成，损失：{avg_loss:.6f}，奖励：{avg_reward:.6f}")
        
        return metrics
    
    def _train_step(self, batch: Any) -> Tuple[float, float]:
        """
        训练一个步骤
        
        Args:
            batch (Any): 批次数据
            
        Returns:
            Tuple[float, float]: 损失和奖励
        """
        # 重置梯度
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # 移动数据到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # 1. 使用全局图
        graph_data = self.graph
        
        # 2. 获取节点索引
        # batch['protein_a_id'] 和 batch['protein_b_id'] 是列表 (collated string/int)
        source_ids = batch['protein_a_id']
        target_ids = batch['protein_b_id']

        source_seqs = batch.get("protein_a", None)
        target_seqs = batch.get("protein_b", None)
        if not isinstance(source_seqs, (list, tuple)):
            source_seqs = []
        if not isinstance(target_seqs, (list, tuple)):
            target_seqs = []
        if len(source_seqs) != len(source_ids):
            source_seqs = list(source_seqs) + [""] * max(0, len(source_ids) - len(source_seqs))
        if len(target_seqs) != len(target_ids):
            target_seqs = list(target_seqs) + [""] * max(0, len(target_ids) - len(target_seqs))

        mapped_source_ids = source_ids
        mapped_target_ids = target_ids
        if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
            mapped_source_ids = self.protein_similarity_mapper.map_batch(
                protein_ids=source_ids,
                protein_sequences=source_seqs,
                non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
            ).mapped_ids
            mapped_target_ids = self.protein_similarity_mapper.map_batch(
                protein_ids=target_ids,
                protein_sequences=target_seqs,
                non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
            ).mapped_ids

        protein_id_to_idx = self.graph.protein_id_to_idx
        valid_indices = []
        source_nodes = []
        target_nodes = []
        for i, (spid, tpid) in enumerate(zip(mapped_source_ids, mapped_target_ids)):
            sidx = protein_id_to_idx.get(spid)
            tidx = protein_id_to_idx.get(tpid)
            if sidx is None or tidx is None:
                continue
            valid_indices.append(i)
            source_nodes.append(sidx)
            target_nodes.append(tidx)

        if not valid_indices:
            return 0.0, 0.0

        def _subset(v):
            if isinstance(v, torch.Tensor):
                return v[valid_indices]
            if isinstance(v, (list, tuple)):
                return [v[i] for i in valid_indices]
            return v

        for k in ['protein_a_id', 'protein_b_id', 'protein_a', 'protein_b', 'label']:
            if k in batch:
                batch[k] = _subset(batch[k])

        source_ids = _subset(source_ids)
        target_ids = _subset(target_ids)
        source_seqs = _subset(source_seqs)
        target_seqs = _subset(target_seqs)
        mapped_source_ids = _subset(mapped_source_ids)
        mapped_target_ids = _subset(mapped_target_ids)
        
        # 3. 批量生成COT路径
        # 构造目标信息，包含 protein_id 用于在路径探索中识别目标节点
        mapped_target_seqs = [
            (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
            for i in range(len(target_nodes))
        ]
        target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
        target_infos = [
            {'esm_features': target_esms[i], 'protein_id': target_nodes[i]}
            for i in range(len(target_nodes))
        ]
        
        # 获取生成的链数量配置
        num_chains = self.config.get('model', {}).get('num_chains', 3)
        
        # 兼容分布式：如果是DDP，则通过.module访问自定义方法
        generator = getattr(self.cot_generator, 'module', self.cot_generator)
        paths_info_batch = generator.batch_generate_chains(
            start_protein_ids=source_nodes,
            target_protein_infos=target_infos,
            graph_data=graph_data,
            training=True,
            device=self.device,
            exclude_direct_edge=True,  # 强制多步探索
            num_chains=num_chains
        )
        # paths_info_batch 是 List[List[Dict]] (batch_size, num_chains)
        
        # 提取用于提示的路径和关系 (List[List[List[int]]])
        multi_paths = [[info['path'] for info in sample_paths] for sample_paths in paths_info_batch]
        multi_relations = [[info.get('relations', []) for info in sample_paths] for sample_paths in paths_info_batch]
        
        # 4. 生成提示
        # 构造批量提示所需数据
        # 优先使用 prompt_designer 中已加载的详细 protein_id_to_info
        protein_info = {}
        for i, idx in enumerate(source_nodes):
            # 通过序列获取蛋白质信息（解决 ID 不匹配问题）
            seq_a = source_seqs[i] if i < len(source_seqs) else ""
            if self.graph_builder:
                info_a = self.graph_builder.get_protein_info_by_sequence(seq_a, self.prompt_designer)
            else:
                info_a = {'name': str(source_ids[i]), 'function': '暂无功能描述'}
            protein_info[idx] = info_a
            
        for i, idx in enumerate(target_nodes):
            # 通过序列获取蛋白质信息（解决 ID 不匹配问题）
            seq_b = target_seqs[i] if i < len(target_seqs) else ""
            if self.graph_builder:
                info_b = self.graph_builder.get_protein_info_by_sequence(seq_b, self.prompt_designer)
            else:
                info_b = {'name': str(target_ids[i]), 'function': '暂无功能描述'}
            protein_info[idx] = info_b
            
        # 补充图中其他节点（如果需要）
        for pid, idx in graph_data.protein_id_to_idx.items():
            if idx not in protein_info:
                if pid in self.prompt_designer.protein_id_to_info:
                    protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
                else:
                    protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}
        
        batch_prompt_data = []
        for i in range(len(multi_paths)):
            batch_prompt_data.append({
                'source_protein': source_ids[i],
                'target_protein': target_ids[i],
                'path': multi_paths[i], # 传入 List[List[int]]
                'relations': multi_relations[i], # 传入 List[List[int]]
                'protein_info': protein_info
            })
        prompts = self.prompt_designer.generate_batch_prompts(
            template_type='exploratory_reasoning',
            batch_data=batch_prompt_data
        )
        
        # 5. LLM预测 / 训练
        llm_cfg = self.config.get('llm', {}) or {}
        train_llm = bool(llm_cfg.get('train_llm', False))
        llm_output_mode = str(llm_cfg.get('output_mode', 'relation_head') or 'relation_head').lower()
        probabilities = None
        llm_loss_value = 0.0
        if train_llm and self.optimizer:
            self.optimizer.zero_grad()
            if llm_output_mode == "text":
                labels = batch["label"]
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                label_names = [
                    "Activation",
                    "Inhibition",
                    "Binding",
                    "Catalysis",
                    "Expression Regulation",
                    "Physical Interaction",
                    "Genetic Interaction",
                ]
                target_texts = []
                for i in range(labels.size(0)):
                    active = []
                    for j in range(min(labels.size(1), len(label_names))):
                        if float(labels[i, j].item()) > 0.5:
                            active.append(label_names[j])
                    if not active:
                        active = [label_names[5] if len(label_names) > 5 else label_names[0]]
                    target_texts.append(",".join(active))
                self.llm_wrapper.train()
                llm_loss = self.llm_wrapper.compute_text_loss(prompts, target_texts)
                if llm_loss is not None:
                    llm_loss.backward()
                    max_grad_norm = float(self.config.get('optimizer', {}).get('max_grad_norm', 1.0))
                    torch.nn.utils.clip_grad_norm_(self.llm_wrapper.parameters(), max_grad_norm)
                    self.optimizer.step()
                    llm_loss_value = float(llm_loss.detach().item())
                predictions, logits = self.llm_wrapper.predict(prompts, return_type='logits')
                probabilities = logits.detach() if isinstance(logits, torch.Tensor) else logits
            else:
                self.llm_wrapper.train()
                llm_out = self.llm_wrapper(prompts, labels=batch['label'])
                logits = llm_out['logits']
                llm_loss = llm_out['loss']
                if llm_loss is not None:
                    llm_loss.backward()
                    max_grad_norm = float(self.config.get('optimizer', {}).get('max_grad_norm', 1.0))
                    torch.nn.utils.clip_grad_norm_(self.llm_wrapper.parameters(), max_grad_norm)
                    self.optimizer.step()
                    llm_loss_value = float(llm_loss.detach().item())
                predictions = (logits.detach() > 0).float()
                probabilities = logits.detach()
        else:
            predictions, probabilities = self.llm_wrapper.predict(prompts, return_type='logits')
        
        # 6. 计算奖励
        # 为了支持多条链，我们需要平铺数据进行奖励计算
        # 每个样本的预测和标签都要复制 num_chains 次
        flattened_paths_info = []
        flattened_source_nodes = []
        flattened_target_nodes = []
        for i, sample_paths in enumerate(paths_info_batch):
            for info in sample_paths:
                flattened_paths_info.append(info)
                flattened_source_nodes.append(source_nodes[i])
                flattened_target_nodes.append(target_nodes[i])
        
        flattened_paths = [info['path'] for info in flattened_paths_info]

        chain_counts = [len(sample_paths) for sample_paths in paths_info_batch]
        repeat_counts = torch.tensor(chain_counts, dtype=torch.long, device=self.device)

        if isinstance(predictions, torch.Tensor):
            rep_predictions = predictions.repeat_interleave(repeat_counts, dim=0)
        else:
            rep_predictions = np.repeat(predictions, chain_counts, axis=0)

        rep_labels = batch['label'].repeat_interleave(repeat_counts, dim=0)
        
        # 提取实际选择的边特征和边索引
        all_edge_features = []
        all_edge_indices = []
        for info in flattened_paths_info:
            if 'edge_features' in info and info['edge_features'] is not None:
                all_edge_features.append(info['edge_features'])
                p = info['path']
                edges = []
                for j in range(len(p) - 1):
                    edges.append((p[j], p[j+1]))
                all_edge_indices.append(edges)
            else:
                all_edge_features.append(torch.zeros((0, generator.edge_dim), device=self.device))
                all_edge_indices.append([])
        
        if all_edge_features:
            combined_edge_features = torch.cat(all_edge_features, dim=0)
        else:
            combined_edge_features = None

        reward_dict = self.reward_calculator.compute_rewards(
            predictions=rep_predictions,
            labels=rep_labels,
            paths=flattened_paths,
            graph_data=graph_data,
            source_nodes=flattened_source_nodes,
            target_nodes=flattened_target_nodes,
            edge_features=combined_edge_features,
            edge_indices=all_edge_indices
        )
        
        # 7. 构造RL训练数据
        # 状态应包含起始节点表征和目标蛋白质特征
        initial_hist = generator.initial_history.to(self.device)
        
        # 批量获取源节点表征
        source_node_feats = self.graph.x[source_nodes].to(self.device)
        source_node_reps = generator.node_rep(source_node_feats, initial_hist.unsqueeze(0).expand(len(source_nodes), -1))
        
        # 获取目标ESM特征
        target_esm_feats = torch.stack([target_infos[i]['esm_features'] for i in range(len(target_nodes))]).to(self.device)
        
        # 拼接作为状态 [batch_size, state_dim]
        states = torch.cat([source_node_reps, target_esm_feats], dim=-1)

        rep_states = states.repeat_interleave(repeat_counts, dim=0)
        
        # Actions: Pad paths to fixed length
        # Use -1 as padding so PPO policy can skip padded steps
        fixed_max_len = generator.max_steps + 1
        actions = torch.full((len(flattened_paths), fixed_max_len), -1, dtype=torch.long, device=self.device)
        for i, p in enumerate(flattened_paths):
            path_len = len(p)
            if path_len > 0:
                if path_len > fixed_max_len:
                    p = p[:fixed_max_len]
                    path_len = fixed_max_len
                actions[i, :path_len] = torch.tensor(p, device=self.device, dtype=torch.long)

        rep_logits = probabilities.repeat_interleave(repeat_counts, dim=0)
        per_chain_loss = F.binary_cross_entropy_with_logits(rep_logits.float(), rep_labels.float(), reduction='none').mean(dim=1)
            
        rl_data = {
            'states': rep_states.detach(),
            'actions': actions.detach(),
            'rewards': reward_dict['total'].detach(),
            'next_states': rep_states.detach(), # Dummy
            'masks': torch.zeros(len(flattened_paths), device=self.device) # Terminal state mask should be 0
        }
        
        # 8. PPO更新
        loss = 0.0
        rl_cfg = self.config.get('reinforcement_learning', {}) or {}
        if not rl_cfg:
            legacy_rl = self.config.get('rl', {}) or {}
            ppo_cfg = legacy_rl.get('ppo', {}) if isinstance(legacy_rl, dict) else {}
            if isinstance(ppo_cfg, dict) and ppo_cfg:
                rl_cfg = {
                    'enabled': True,
                    'learning_rate': ppo_cfg.get('learning_rate', ppo_cfg.get('lr', 3e-4)),
                    'clip_param': ppo_cfg.get('clip_ratio', ppo_cfg.get('clip_param', 0.2)),
                    'epochs': ppo_cfg.get('num_epochs', ppo_cfg.get('epochs', 10)),
                    'batch_size': ppo_cfg.get('minibatch_size', ppo_cfg.get('batch_size', 64)),
                    'gamma': ppo_cfg.get('gamma', 0.99),
                    'gae_lambda': ppo_cfg.get('lam', ppo_cfg.get('gae_lambda', 0.95)),
                    'entropy_coef': ppo_cfg.get('ent_coef', 0.01),
                    'value_loss_coef': ppo_cfg.get('vf_coef', ppo_cfg.get('value_loss_coef', 0.5)),
                    'max_grad_norm': ppo_cfg.get('max_grad_norm', 0.5),
                }
        rl_enabled = bool(rl_cfg.get('enabled', False))
        if not hasattr(self, "_ppo_status_logged"):
            self._ppo_status_logged = True
            if int(getattr(self, "rank", 0) or 0) == 0:
                logger.info(
                    f"RL(PPO) enabled={rl_enabled}, update_every_steps={int(rl_cfg.get('update_every_steps', 20))}, "
                    f"loss_reward_weight={float(rl_cfg.get('loss_reward_weight', 1.0))}"
                )
        if rl_enabled:
            try:
                ppo_generator = getattr(self.cot_generator, 'module', self.cot_generator)
                loss_reward_weight = float(rl_cfg.get('loss_reward_weight', 1.0))
                rl_rewards = rl_data['rewards'] + (-per_chain_loss.detach()) * loss_reward_weight
                rl_data['rewards'] = rl_rewards
                path_log_probs = torch.tensor([
                    sum(info['log_probs']) if info['log_probs'] else 0.0
                    for info in flattened_paths_info
                ], device=self.device)
                if int(getattr(self, "rank", 0) or 0) == 0:
                    if not hasattr(self, "_ppo_collect_steps"):
                        self._ppo_collect_steps = 0
                    self._ppo_collect_steps += 1
                    if (self._ppo_collect_steps % 10) == 1:
                        try:
                            buf_size = int(self.ppo_trainer.buffer.size())
                        except Exception:
                            buf_size = -1
                        logger.info(
                            f"PPO collecting experience: step={self._ppo_collect_steps}, "
                            f"paths={int(path_log_probs.numel())}, buffer_size={buf_size}"
                        )
                self.ppo_trainer.collect_experience_batch(rl_data, provided_log_probs=path_log_probs, model=ppo_generator)
                update_every_steps = int(rl_cfg.get('update_every_steps', 20))
                if not hasattr(self, '_rl_global_step'):
                    self._rl_global_step = 0
                self._rl_global_step += 1
                if update_every_steps > 0 and (self._rl_global_step % update_every_steps) == 0:
                    if int(getattr(self, "rank", 0) or 0) == 0:
                        try:
                            buf_size = int(self.ppo_trainer.buffer.size())
                        except Exception:
                            buf_size = -1
                        logger.info(f"PPO update triggered at global_step={self._rl_global_step}, buffer_size={buf_size}")
                    rl_metrics = self.ppo_trainer.update_policy(model=ppo_generator)
                    loss = float(rl_metrics.get('total_loss', 0.0) if rl_metrics else 0.0)
            except Exception as e:
                logger.warning(f"PPO step failed: {e}", exc_info=True)
                if hasattr(self.ppo_trainer, 'buffer'):
                    self.ppo_trainer.buffer.clear()
                loss = 0.0

        if train_llm and self.optimizer:
            loss = float(loss) + float(llm_loss_value)

        import math
        abort_on_nonfinite = bool((self.config.get('training', {}) or {}).get('abort_on_nonfinite', True))
        if not math.isfinite(float(loss)):
            if abort_on_nonfinite:
                raise RuntimeError(f"Non-finite loss detected: {loss}")
            loss = 0.0

        avg_reward = float(reward_dict['total'].mean().item())
        if not math.isfinite(avg_reward):
            if abort_on_nonfinite:
                raise RuntimeError(f"Non-finite reward detected: {avg_reward}")
            avg_reward = 0.0
        
        return float(loss), avg_reward
    
    def _validate_epoch(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            epoch (int): 当前epoch
            dataloader (DataLoader): 数据加载器
            
        Returns:
            Dict[str, float]: 验证指标
        """
        self.cot_generator.eval()
        if self.llm_wrapper and self.llm_wrapper.model:
            self.llm_wrapper.model.eval()
        
        total_loss = 0
        total_reward = 0
        total_steps = 0
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} Val")
            for batch_idx, batch in pbar:
                # 移动数据到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 1. 使用全局图
                graph_data = self.graph
                
                # 2. 获取节点索引
                source_ids = batch['protein_a_id']
                target_ids = batch['protein_b_id']

                source_seqs = batch.get("protein_a", [])
                target_seqs = batch.get("protein_b", [])

                mapped_source_ids = source_ids
                mapped_target_ids = target_ids
                if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
                    mapped_source_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=source_ids,
                        protein_sequences=source_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids
                    mapped_target_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=target_ids,
                        protein_sequences=target_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids

                protein_id_to_idx = self.graph.protein_id_to_idx
                valid_indices = []
                source_nodes = []
                target_nodes = []
                for i, (spid, tpid) in enumerate(zip(mapped_source_ids, mapped_target_ids)):
                    sidx = protein_id_to_idx.get(spid)
                    tidx = protein_id_to_idx.get(tpid)
                    if sidx is None or tidx is None:
                        continue
                    valid_indices.append(i)
                    source_nodes.append(sidx)
                    target_nodes.append(tidx)

                if not valid_indices:
                    continue

                def _subset(v):
                    if isinstance(v, torch.Tensor):
                        return v[valid_indices]
                    if isinstance(v, (list, tuple)):
                        return [v[i] for i in valid_indices]
                    return v

                for k in ['protein_a_id', 'protein_b_id', 'protein_a', 'protein_b', 'label']:
                    if k in batch:
                        batch[k] = _subset(batch[k])

                source_ids = _subset(source_ids)
                target_ids = _subset(target_ids)
                source_seqs = _subset(source_seqs)
                target_seqs = _subset(target_seqs)
                mapped_source_ids = _subset(mapped_source_ids)
                mapped_target_ids = _subset(mapped_target_ids)
                
                # 3. 批量生成COT路径
                mapped_target_seqs = [
                    (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
                    for i in range(len(target_nodes))
                ]
                target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
                target_infos = [
                    {'esm_features': target_esms[i], 'protein_id': target_nodes[i]}
                    for i in range(len(target_nodes))
                ]
                
                # 获取生成的链数量配置
                num_chains = self.config.get('model', {}).get('num_chains', 3)
                
                generator = getattr(self.cot_generator, "module", self.cot_generator)
                paths_info_batch = generator.batch_generate_chains(
                    start_protein_ids=source_nodes,
                    target_protein_infos=target_infos,
                    graph_data=graph_data,
                    training=False,
                    device=self.device,
                    num_chains=num_chains
                )
                # paths_info_batch 是 List[List[Dict]] (batch_size, num_chains)
                
                # 提取用于提示的路径和关系
                multi_paths = [[info['path'] for info in sample_paths] for sample_paths in paths_info_batch]
                multi_relations = [[info.get('relations', []) for info in sample_paths] for sample_paths in paths_info_batch]
                
                # 4. 生成提示
                protein_info = {}
                for i, idx in enumerate(source_nodes):
                    seq_a = batch['protein_a'][i]
                    info_a = self.graph_builder.get_protein_info_by_sequence(seq_a, self.prompt_designer)
                    protein_info[idx] = info_a
                    
                for i, idx in enumerate(target_nodes):
                    seq_b = batch['protein_b'][i]
                    info_b = self.graph_builder.get_protein_info_by_sequence(seq_b, self.prompt_designer)
                    protein_info[idx] = info_b
                    
                for pid, idx in graph_data.protein_id_to_idx.items():
                    if idx not in protein_info:
                        if pid in self.prompt_designer.protein_id_to_info:
                            protein_info[idx] = self.prompt_designer.protein_id_to_info[pid]
                        else:
                            protein_info[idx] = {'name': pid, 'function': '暂无功能描述'}
                
                batch_prompt_data = []
                for i in range(len(multi_paths)):
                    batch_prompt_data.append({
                        'source_protein': source_ids[i],
                        'target_protein': target_ids[i],
                        'path': multi_paths[i],
                        'relations': multi_relations[i],
                        'protein_info': protein_info
                    })
                prompts = self.prompt_designer.generate_batch_prompts(
                    template_type='exploratory_reasoning',
                    batch_data=batch_prompt_data
                )
                
                # 5. LLM预测
                predictions, probabilities = self.llm_wrapper.predict(prompts, return_type='logits')
                
                # 6. 计算奖励
                # 为了支持多条链，平铺数据进行奖励计算
                flattened_paths_info = []
                flattened_source_nodes = []
                flattened_target_nodes = []
                for i, sample_paths in enumerate(paths_info_batch):
                    for info in sample_paths:
                        flattened_paths_info.append(info)
                        flattened_source_nodes.append(source_nodes[i])
                        flattened_target_nodes.append(target_nodes[i])
                
                flattened_paths = [info['path'] for info in flattened_paths_info]

                chain_counts = [len(sample_paths) for sample_paths in paths_info_batch]
                repeat_counts = torch.tensor(chain_counts, dtype=torch.long, device=self.device)

                if isinstance(predictions, torch.Tensor):
                    rep_predictions = predictions.repeat_interleave(repeat_counts, dim=0)
                else:
                    rep_predictions = np.repeat(predictions, chain_counts, axis=0)

                rep_labels = batch['label'].repeat_interleave(repeat_counts, dim=0)
                
                all_edge_features = []
                all_edge_indices = []
                for info in flattened_paths_info:
                    if 'edge_features' in info and info['edge_features'] is not None:
                        all_edge_features.append(info['edge_features'])
                        p = info['path']
                        edges = []
                        for j in range(len(p) - 1):
                            edges.append((p[j], p[j+1]))
                        all_edge_indices.append(edges)
                    else:
                        all_edge_features.append(torch.zeros((0, self.cot_generator.edge_dim), device=self.device))
                        all_edge_indices.append([])
                
                if all_edge_features:
                    combined_edge_features = torch.cat(all_edge_features, dim=0)
                else:
                    combined_edge_features = None

                reward_dict = self.reward_calculator.compute_rewards(
                    predictions=rep_predictions,
                    labels=rep_labels,
                    paths=flattened_paths,
                    graph_data=graph_data,
                    source_nodes=flattened_source_nodes,
                    target_nodes=flattened_target_nodes,
                    edge_features=combined_edge_features,
                    edge_indices=all_edge_indices
                )
                
                if not torch.isfinite(probabilities).all():
                    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
                loss = F.binary_cross_entropy_with_logits(probabilities.float(), batch['label'].float())
                
                total_loss += loss.item()
                reward = reward_dict['total'].mean().item()
                total_reward += reward
                total_steps += 1

                # 更新进度条
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'reward': f"{reward:.4f}"})

                # 收集用于评估的预测和标签
                if isinstance(predictions, torch.Tensor):
                    all_preds.extend(predictions.detach().cpu().numpy())
                else:
                    all_preds.extend(predictions)
                    
                all_probs.extend(torch.sigmoid(probabilities.float()).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                
                # 限制验证步数以提高效率
                max_steps = self.config.get('training', {}).get('quick_run_steps')
                if isinstance(max_steps, int) and max_steps > 0 and (batch_idx + 1) >= max_steps:
                    break
        
        end_time = time.time()
        
        # 计算平均指标
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        
        # 计算分类指标
        metrics = {
            'val_loss': avg_loss,
            'val_reward': avg_reward,
            'val_epoch_time': end_time - start_time
        }

        import numpy as np
        if all_labels:
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)
            all_probs_np = np.array(all_probs)
            
            # 多标签评价指标
            from sklearn.metrics import hamming_loss
            metrics['val_accuracy'] = (all_preds_np == all_labels_np).all(axis=1).mean() # Exact Match Ratio
            metrics['val_f1_score'] = f1_score(all_labels_np, all_preds_np, average='micro')
            metrics['val_precision'] = precision_score(all_labels_np, all_preds_np, average='micro', zero_division=0)
            metrics['val_recall'] = recall_score(all_labels_np, all_preds_np, average='micro', zero_division=0)
            metrics['val_hamming_loss'] = hamming_loss(all_labels_np, all_preds_np)
            
            try:
                metrics['val_auc'] = roc_auc_score(all_labels_np, all_probs_np, average='micro')
            except Exception:
                pass
        
        logger.info(f"Epoch {epoch+1} - 验证完成，损失：{avg_loss:.6f}，奖励：{avg_reward:.6f}")
        
        return metrics

    def evaluate(self, test_data: Any, model: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Args:
            test_data (Any): 测试数据
            model (Optional[nn.Module]): 要评估的模型，如果为None则使用self.cot_generator
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("开始测试集评估...")
        
        # 准备模型
        original_model = self.cot_generator
        if model is not None:
            self.cot_generator = model
            
        self.cot_generator.eval()
        if self.llm_wrapper and self.llm_wrapper.model:
            self.llm_wrapper.model.eval()
            
        # 准备数据加载器
        if isinstance(test_data, tuple):
            _, test_dataset = test_data
        else:
            test_dataset = test_data
            
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        dataloader = self._create_dataloader(test_dataset, batch_size)
        
        total_loss = 0
        total_reward = 0
        total_steps = 0
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Evaluation")
            for batch_idx, batch in pbar:
                # 移动数据到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 1. 使用全局图
                graph_data = self.graph
                
                # 2. 获取节点索引
                source_ids = batch['protein_a_id']
                target_ids = batch['protein_b_id']

                source_seqs = batch.get("protein_a", [])
                target_seqs = batch.get("protein_b", [])

                mapped_source_ids = source_ids
                mapped_target_ids = target_ids
                if hasattr(self, "protein_similarity_mapper") and self.protein_similarity_mapper:
                    mapped_source_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=source_ids,
                        protein_sequences=source_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids
                    mapped_target_ids = self.protein_similarity_mapper.map_batch(
                        protein_ids=target_ids,
                        protein_sequences=target_seqs,
                        non_isolated_ids=getattr(self, "_non_isolated_protein_ids", set()),
                    ).mapped_ids

                protein_id_to_idx = self.graph.protein_id_to_idx
                valid_indices = []
                source_nodes = []
                target_nodes = []
                for i, (spid, tpid) in enumerate(zip(mapped_source_ids, mapped_target_ids)):
                    sidx = protein_id_to_idx.get(spid)
                    tidx = protein_id_to_idx.get(tpid)
                    if sidx is None or tidx is None:
                        continue
                    valid_indices.append(i)
                    source_nodes.append(sidx)
                    target_nodes.append(tidx)

                if not valid_indices:
                    continue

                def _subset(v):
                    if isinstance(v, torch.Tensor):
                        return v[valid_indices]
                    if isinstance(v, (list, tuple)):
                        return [v[i] for i in valid_indices]
                    return v

                for k in ['protein_a_id', 'protein_b_id', 'protein_a', 'protein_b', 'label']:
                    if k in batch:
                        batch[k] = _subset(batch[k])

                source_ids = _subset(source_ids)
                target_ids = _subset(target_ids)
                source_seqs = _subset(source_seqs)
                target_seqs = _subset(target_seqs)
                mapped_source_ids = _subset(mapped_source_ids)
                mapped_target_ids = _subset(mapped_target_ids)
                
                # 3. 批量生成COT路径
                mapped_target_seqs = [
                    (getattr(self, "_train_protein_sequences", {}) or {}).get(mapped_target_ids[i], target_seqs[i])
                    for i in range(len(target_nodes))
                ]
                target_esms = self.esm_encoder.get_batch_embeddings(mapped_target_seqs, batch_size=min(16, len(mapped_target_seqs)))
                target_infos = [
                    {'esm_features': target_esms[i], 'protein_id': target_nodes[i]}
                    for i in range(len(target_nodes))
                ]
                
                # 获取生成的链数量配置
                num_chains = self.config.get('model', {}).get('num_chains', 3)
                
                generator = getattr(self.cot_generator, "module", self.cot_generator)
                paths_info_batch = generator.batch_generate_chains(
                    start_protein_ids=source_nodes,
                    target_protein_infos=target_infos,
                    graph_data=graph_data,
                    training=False,
                    device=self.device,
                    num_chains=num_chains
                )
                # paths_info_batch 是 List[List[Dict]] (batch_size, num_chains)
                
                # 提取用于提示的路径和关系
                multi_paths = [[info['path'] for info in sample_paths] for sample_paths in paths_info_batch]
                multi_relations = [[info.get('relations', []) for info in sample_paths] for sample_paths in paths_info_batch]
                
                # 4. 生成提示
                protein_info = {idx: {'name': pid} for pid, idx in graph_data.protein_id_to_idx.items()}
                batch_prompt_data = []
                for i in range(len(multi_paths)):
                    batch_prompt_data.append({
                        'source_protein': source_ids[i],
                        'target_protein': target_ids[i],
                        'path': multi_paths[i],
                        'relations': multi_relations[i],
                        'protein_info': protein_info
                    })
                prompts = self.prompt_designer.generate_batch_prompts(
                    template_type='exploratory_reasoning',
                    batch_data=batch_prompt_data
                )
                
                # 5. LLM预测
                predictions, probabilities = self.llm_wrapper.predict(prompts, return_type='logits')
                
                # 6. 计算奖励
                flattened_paths_info = [info for sample_paths in paths_info_batch for info in sample_paths]
                paths = [info['path'] for info in flattened_paths_info]

                chain_counts = [len(sample_paths) for sample_paths in paths_info_batch]
                repeat_counts = torch.tensor(chain_counts, dtype=torch.long, device=self.device)

                if isinstance(predictions, torch.Tensor):
                    rep_predictions = predictions.repeat_interleave(repeat_counts, dim=0)
                else:
                    rep_predictions = np.repeat(predictions, chain_counts, axis=0)

                rep_labels = batch['label'].repeat_interleave(repeat_counts, dim=0)

                flattened_source_nodes = [sn for sn, c in zip(source_nodes, chain_counts) for _ in range(c)]
                flattened_target_nodes = [tn for tn, c in zip(target_nodes, chain_counts) for _ in range(c)]

                reward_dict = self.reward_calculator.compute_rewards(
                    predictions=rep_predictions,
                    labels=rep_labels,
                    paths=paths,
                    graph_data=graph_data,
                    source_nodes=flattened_source_nodes,
                    target_nodes=flattened_target_nodes
                )
                
                # 计算损失
                loss = F.binary_cross_entropy_with_logits(probabilities, batch['label'].float())
                
                total_loss += loss.item()
                reward = reward_dict['total'].mean().item()
                total_reward += reward
                total_steps += 1

                # 更新进度条
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'reward': f"{reward:.4f}"})

                # 收集用于评估的预测和标签
                if isinstance(predictions, torch.Tensor):
                    all_preds.extend(predictions.detach().cpu().numpy())
                else:
                    all_preds.extend(predictions)
                
                all_probs.extend(torch.sigmoid(probabilities).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                
                # 限制评估步数以提高效率 (主要用于调试/快速运行)
                max_steps = self.config.get('training', {}).get('quick_run_steps')
                if isinstance(max_steps, int) and max_steps > 0 and (batch_idx + 1) >= max_steps:
                    break
                
        end_time = time.time()
        
        # 计算平均指标
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        
        # 计算分类指标
        metrics = {
            'test_loss': avg_loss,
            'test_reward': avg_reward,
            'test_time': end_time - start_time
        }

        import numpy as np
        if all_labels:
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)
            all_probs_np = np.array(all_probs)
            
            metrics['test_accuracy'] = accuracy_score(all_labels_np, all_preds_np)
            metrics['test_f1_score'] = f1_score(all_labels_np, all_preds_np, average='macro')
            metrics['test_precision'] = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
            metrics['test_recall'] = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
            
            try:
                if len(np.unique(all_labels_np)) > 1:
                    metrics['test_auc'] = roc_auc_score(all_labels_np, all_probs_np)
            except:
                pass
        
        # 恢复模型
        if model is not None:
            self.cot_generator = original_model
            
        logger.info(f"测试集评估完成，损失：{avg_loss:.6f}，奖励：{avg_reward:.6f}")
        
        return metrics
    
    def _on_train_begin(self):
        """
        训练开始时调用
        """
        for callback in self.callbacks:
            callback.on_train_begin()
    
    def _on_train_end(self):
        """
        训练结束时调用
        """
        for callback in self.callbacks:
            callback.on_train_end()
    
    def _on_epoch_begin(self, epoch: int):
        """
        Epoch开始时调用
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)
    
    def _on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """
        Epoch结束时调用
        """
        # 更新历史指标用于绘图
        if 'loss' in metrics:
            self.history['loss'].append(metrics['loss'])
        if 'total_reward' in metrics:
            self.history['reward'].append(metrics['total_reward'])
        elif 'reward' in metrics:
            self.history['reward'].append(metrics['reward'])
        
        # 尝试从分类指标中获取准确率
        acc_key = 'val_accuracy' if 'val_accuracy' in metrics else 'accuracy'
        if acc_key in metrics:
            self.history['accuracy'].append(metrics[acc_key])
        
        # 仅在主进程中绘图
        if not self.multi_gpu or self.rank == 0:
            self._plot_metrics()

        # 添加模型等信息到指标
        metrics['model'] = self.cot_generator
        metrics['rank'] = getattr(self, "rank", 0)
        metrics['world_size'] = getattr(self, "world_size", 1)
        if self.llm_wrapper is not None:
            metrics['llm_classifier_state_dict'] = {
                k: v.detach().to("cpu")
                for k, v in self.llm_wrapper.relation_classifier.state_dict().items()
            }
            llm_cfg = self.config.get('llm', {}) or {}
            if getattr(self.llm_wrapper, "model", None) is not None:
                if bool(llm_cfg.get('train_backbone', False)):
                    metrics['llm_state_dict'] = {
                        k: v.detach().to("cpu")
                        for k, v in self.llm_wrapper.model.state_dict().items()
                    }
                else:
                    use_lora = bool((llm_cfg.get('lora', {}) or {}).get('use_lora', False)) and bool(getattr(self.llm_wrapper, "use_lora", False))
                    if use_lora:
                        try:
                            from peft import get_peft_model_state_dict
                            peft_sd = get_peft_model_state_dict(self.llm_wrapper.model)
                            metrics['llm_peft_state_dict'] = {k: v.detach().to("cpu") for k, v in peft_sd.items()}
                        except Exception:
                            try:
                                metrics['llm_peft_state_dict'] = {
                                    k: v.detach().to("cpu")
                                    for k, v in self.llm_wrapper.model.state_dict().items()
                                    if "lora_" in k
                                }
                            except Exception:
                                pass
        if self.optimizer:
            metrics['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            metrics['scheduler_state_dict'] = self.scheduler.state_dict()
        
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)
    
    def _on_batch_begin(self, batch: int):
        """
        Batch开始时调用
        """
        for callback in self.callbacks:
            callback.on_batch_begin(batch)
    
    def _on_batch_end(self, batch: int, metrics: Dict[str, float]):
        """
        Batch结束时调用
        """
        for callback in self.callbacks:
            callback.on_batch_end(batch, metrics)
    
    def _should_stop_training(self) -> bool:
        """
        检查是否需要停止训练
        
        Returns:
            bool: 是否需要停止训练
        """
        for callback in self.callbacks:
            if hasattr(callback, 'early_stop') and callback.early_stop:
                return True
        
        return False
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, Any]):
        """
        保存检查点
        
        Args:
            path (str): 保存路径
            epoch (int): 当前epoch
            metrics (Dict[str, Any]): 指标
        """
        try:
            # 创建保存目录
            os.makedirs(path, exist_ok=True)
            
            # 保存训练器状态
            checkpoint = {
                'epoch': epoch,
                'cot_generator_state_dict': self.cot_generator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': metrics,
                'config': self.config
            }
            
            if self.llm_wrapper and self.llm_wrapper.model:
                checkpoint['llm_state_dict'] = self.llm_wrapper.model.state_dict()
            
            checkpoint_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"检查点已保存到：{checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败：{e}", exc_info=True)
            # 尝试保存一个简化版本的检查点
            try:
                simple_checkpoint = {
                    'epoch': epoch,
                    'cot_generator_state_dict': self.cot_generator.state_dict(),
                    'metrics': metrics,
                    'config': self.config
                }
                simple_path = os.path.join(path, f"simple_checkpoint_epoch_{epoch}.pth")
                torch.save(simple_checkpoint, simple_path)
                logger.warning(f"已保存简化版检查点到：{simple_path}")
            except Exception as e2:
                logger.error(f"保存简化版检查点也失败：{e2}", exc_info=True)

    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path (str): 检查点路径
            
        Returns:
            Tuple[int, Dict[str, Any]]: 加载的epoch和指标
        """
        if not os.path.exists(path):
            logger.error(f"检查点文件不存在：{path}")
            return 0, {}
        
        try:
            # 加载检查点
            checkpoint = torch.load(path, map_location=self.device)
            logger.info(f"检查点文件已加载：{path}")
            
            # 构建组件
            try:
                self.build_components()
            except Exception as e:
                logger.error(f"构建组件失败：{e}", exc_info=True)
                return 0, {}
            
            # 加载状态
            try:
                self.cot_generator.load_state_dict(checkpoint['cot_generator_state_dict'], strict=False)
                logger.info("已加载cot_generator状态")
            except Exception as e:
                logger.error(f"加载cot_generator状态失败：{e}", exc_info=True)
                return 0, {}
            
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
                logger.info("已加载optimizer状态")
            except Exception as e:
                logger.error(f"加载optimizer状态失败：{e}", exc_info=True)
                # 尝试重新初始化优化器
                self._build_optimizer_and_scheduler()
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("已加载scheduler状态")
                except Exception as e:
                    logger.error(f"加载scheduler状态失败：{e}", exc_info=True)
            
            if self.llm_wrapper and self.llm_wrapper.model and checkpoint.get('llm_state_dict'):
                try:
                    self.llm_wrapper.model.load_state_dict(checkpoint['llm_state_dict'], strict=False)
                    logger.info("已加载llm状态")
                except Exception as e:
                    logger.error(f"加载llm状态失败：{e}", exc_info=True)
            
            logger.info(f"检查点加载完成：{path}")
            
            return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
            
        except Exception as e:
            logger.error(f"加载检查点失败：{e}", exc_info=True)
            return 0, {}


class DistributedExploratoryPPITrainer(ExploratoryPPITrainer):
    """
    分布式探索式PPI训练器
    支持多GPU分布式训练
    """
    
    def __init__(self, config: Dict[str, Any], rank: int, world_size: int):
        """
        初始化分布式训练器
        
        Args:
            config (Dict[str, Any]): 配置字典
            rank (int): 进程排名
            world_size (int): 进程数量
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # 初始化分布式环境
        self._init_distributed()
        
        super().__init__(config)
    
    def _init_distributed(self):
        """
        初始化分布式环境
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
        if not os.environ.get('MASTER_ADDR'):
            os.environ['MASTER_ADDR'] = self.config.get('distributed', {}).get('master_addr', '127.0.0.1')
        if not os.environ.get('MASTER_PORT'):
            os.environ['MASTER_PORT'] = str(self.config.get('distributed', {}).get('master_port', '12355'))
        
        dist.init_process_group(
            backend=self.config.get('distributed', {}).get('backend', 'nccl'),
            rank=self.rank,
            world_size=self.world_size
        )
        
        logger.info(f"分布式环境初始化完成，rank: {self.rank}, world_size: {self.world_size}")
    
    def _get_device(self) -> str:
        """
        获取设备
        
        Returns:
            str: 设备字符串
        """
        return f"cuda:{self.rank}"
    
    def _build_model_components(self):
        """
        构建模型组件（分布式）
        """
        super()._build_model_components()
        
        # 将模型包装为DDP
        if self.world_size > 1:
            base_model = getattr(self.cot_generator, "module", self.cot_generator)
            if hasattr(base_model, "to"):
                base_model.to(self.device)
            self.cot_generator = DDP(
                base_model,
                device_ids=[self.rank],
                output_device=self.rank
            )
    
    def _on_train_end(self):
        """
        训练结束时调用（分布式）
        """
        super()._on_train_end()
        
        # 销毁分布式进程组
        dist.destroy_process_group()
        logger.info("分布式进程组已销毁")
    
    @classmethod
    def launch(cls, config: Dict[str, Any]):
        """
        启动分布式训练
        
        Args:
            config (Dict[str, Any]): 配置字典
        """
        # 如果显式禁用了分布式，则只在 rank 0 运行一个进程
        if not config.get('distributed', {}).get('use_distributed', True):
            cls._train_process(0, 1, config)
            return

        cfg_world_size = config.get('distributed', {}).get('world_size', torch.cuda.device_count())
        try:
            cfg_world_size = int(cfg_world_size)
        except Exception:
            cfg_world_size = torch.cuda.device_count()
        available = torch.cuda.device_count()
        world_size = min(cfg_world_size, available) if available > 0 else cfg_world_size
        if world_size < 1:
            raise RuntimeError("No CUDA devices available for distributed training.")
        logger.info(f"启动分布式训练，world_size: {world_size}")
        
        mp.spawn(
            cls._train_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
            daemon=False
        )
    
    @staticmethod
    def _train_process(rank: int, world_size: int, config: Dict[str, Any]):
        """
        训练进程
        
        Args:
            rank (int): 进程排名
            world_size (int): 进程数量
            config (Dict[str, Any]): 配置字典
        """
        try:
            faulthandler.enable()
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except Exception:
            pass

        trainer = DistributedExploratoryPPITrainer(config, rank, world_size)
        try:
            train_data = trainer._load_data('train')
            val_data = trainer._load_data('val') if config.get('training', {}).get('validate', True) else None
            trainer.train(train_data, val_data)
        finally:
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
