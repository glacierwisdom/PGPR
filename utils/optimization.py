import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Union, List, Callable

class Optimizer:
    """
    性能优化器类，提供各种训练优化技术
    
    功能包括：
    - 梯度累积
    - 混合精度训练
    - 模型并行和数据并行
    - 激活检查点
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 gradient_accumulation_steps: int = 1,
                 mixed_precision: bool = False,
                 parallel_mode: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None):
        """
        初始化优化器
        
        Args:
            model: 要优化的模型
            optimizer: PyTorch优化器
            gradient_accumulation_steps: 梯度累积步数
            mixed_precision: 是否使用混合精度训练
            parallel_mode: 并行模式 ('dp' for DataParallel, 'ddp' for DistributedDataParallel, None for no parallel)
            device: 设备
        """
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.parallel_mode = parallel_mode
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 初始化混合精度训练
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # 初始化并行模式
        if self.parallel_mode == 'dp':
            self.model = nn.DataParallel(self.model)
        elif self.parallel_mode == 'ddp':
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        # 梯度累积计数器
        self.accumulation_counter = 0
    
    def zero_grad(self) -> None:
        """
        清空梯度
        """
        if self.accumulation_counter % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        反向传播，支持梯度累积和混合精度训练
        
        Args:
            loss: 损失值
        """
        self.accumulation_counter += 1
        
        # 混合精度训练的反向传播
        if self.mixed_precision:
            # 缩放损失以防止下溢
            self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        else:
            (loss / self.gradient_accumulation_steps).backward()
    
    def step(self) -> None:
        """
        更新模型参数，支持梯度累积和混合精度训练
        """
        if self.accumulation_counter % self.gradient_accumulation_steps == 0:
            if self.mixed_precision:
                # 解缩放梯度并更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
    
    def train_step(self,
                   batch: Dict[str, Any],
                   loss_fn: Callable,
                   model_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        完整的训练步骤
        
        Args:
            batch: 输入批次数据
            loss_fn: 损失函数
            model_output: 可选的模型输出
        
        Returns:
            包含损失和其他指标的字典
        """
        # 设置模型为训练模式
        self.model.train()
        
        # 清空梯度
        self.zero_grad()
        
        # 混合精度训练的前向传播
        if self.mixed_precision:
            with autocast():
                if model_output is None:
                    model_output = self.model(**batch)
                
                # 计算损失
                loss = loss_fn(model_output, batch)
        else:
            if model_output is None:
                model_output = self.model(**batch)
            
            # 计算损失
            loss = loss_fn(model_output, batch)
        
        # 反向传播
        self.backward(loss)
        
        # 更新参数
        self.step()
        
        return {
            "loss": loss.item(),
            "accumulation_step": self.accumulation_counter % self.gradient_accumulation_steps
        }
    
    def save_checkpoint(self,
                       checkpoint_path: str,
                       epoch: int,
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型检查点
        
        Args:
            checkpoint_path: 检查点保存路径
            epoch: 当前epoch
            additional_info: 额外的信息保存到检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accumulation_counter': self.accumulation_counter,
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, map_location: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
            map_location: 设备映射
        
        Returns:
            检查点中的信息
        """
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载累积计数器
        self.accumulation_counter = checkpoint['accumulation_counter']
        
        # 加载混合精度训练状态
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint

def apply_activation_checkpointing(model: nn.Module, layers_to_checkpoint: List[str] = None) -> nn.Module:
    """
    应用激活检查点，减少内存使用
    
    Args:
        model: 模型
        layers_to_checkpoint: 要应用检查点的层名称列表，如果为None则应用到所有可能的层
    
    Returns:
        应用了激活检查点的模型
    """
    # 导入检查点模块
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential
    
    if layers_to_checkpoint is None:
        # 自动查找可能的层类型
        target_layer_types = (nn.ModuleList, nn.Sequential, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        
        # 递归应用检查点
        def _apply_checkpoint(module):
            for name, child in module.named_children():
                if isinstance(child, target_layer_types):
                    # 替换为检查点版本
                    setattr(module, name, checkpoint(child))
                else:
                    _apply_checkpoint(child)
            return module
        
        return _apply_checkpoint(model)
    else:
        # 仅对指定层应用检查点
        for layer_name in layers_to_checkpoint:
            layer = getattr(model, layer_name, None)
            if layer is not None:
                setattr(model, layer_name, checkpoint(layer))
        
        return model

def get_optimizer(optimizer_name: str,
                  model_params: Union[List[nn.Parameter], nn.ParameterDict],
                  lr: float = 1e-4,
                  weight_decay: float = 0.0,
                  betas: tuple = (0.9, 0.999),
                  **kwargs) -> optim.Optimizer:
    """
    获取优化器
    
    Args:
        optimizer_name: 优化器名称 ('adam', 'adamw', 'sgd', 'rmsprop')
        model_params: 模型参数
        lr: 学习率
        weight_decay: 权重衰减
        betas: Adam的betas参数
        kwargs: 其他参数
    
    Returns:
        优化器
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay, betas=betas, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, betas=betas, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(scheduler_name: str,
                  optimizer: optim.Optimizer,
                  step_size: int = 10,
                  gamma: float = 0.1,
                  T_max: int = 100,
                  eta_min: float = 0,
                  lr: float = 1e-4,
                  **kwargs) -> optim.lr_scheduler._LRScheduler:
    """
    获取学习率调度器
    
    Args:
        scheduler_name: 调度器名称 ('step', 'cosine', 'reduceonplateau')
        optimizer: 优化器
        step_size: StepLR的步长
        gamma: 学习率衰减因子
        T_max: CosineAnnealingLR的最大迭代次数
        eta_min: CosineAnnealingLR的最小学习率
        kwargs: 其他参数
    
    Returns:
        学习率调度器
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, **kwargs)
    elif scheduler_name == 'reduceonplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)
    elif scheduler_name == 'cyclic':
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=lr*10, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def dynamic_batch_size(sequence_lengths: torch.Tensor,
                       base_batch_size: int = 32,
                       max_tokens: int = 4096) -> int:
    """
    动态批量大小计算，根据序列长度调整批量大小
    
    Args:
        sequence_lengths: 序列长度张量
        base_batch_size: 基础批量大小
        max_tokens: 最大token数
    
    Returns:
        调整后的批量大小
    """
    avg_length = sequence_lengths.float().mean().item()
    max_length = sequence_lengths.max().item()
    
    # 根据平均长度调整批量大小
    batch_size_by_avg = max(1, int(max_tokens / avg_length))
    
    # 根据最大长度调整批量大小
    batch_size_by_max = max(1, int(max_tokens / max_length))
    
    # 取最小值，确保不超过基础批量大小
    new_batch_size = min(base_batch_size, batch_size_by_avg, batch_size_by_max)
    
    return new_batch_size

def graph_neighbor_sampling(adjacency_matrix: torch.Tensor,
                            node_idx: torch.Tensor,
                            sizes: List[int]) -> Dict[str, torch.Tensor]:
    """
    图邻居采样，减少内存使用
    
    Args:
        adjacency_matrix: 邻接矩阵 (N x N)
        node_idx: 节点索引
        sizes: 每层采样的邻居数量列表
    
    Returns:
        采样后的子图数据
    """
    # 转换为稀疏矩阵以提高效率
    adj_sparse = adjacency_matrix.to_sparse()
    
    sampled_nodes = [node_idx]
    sampled_edges = []
    
    # 进行多层采样
    for size in sizes:
        # 获取当前层节点的邻居
        current_nodes = sampled_nodes[-1]
        
        # 获取邻居索引
        neighbors = []
        for node in current_nodes:
            # 找到所有邻居
            node_neighbors = adj_sparse.indices()[1][adj_sparse.indices()[0] == node]
            
            # 采样邻居
            if len(node_neighbors) > size:
                # 随机采样
                sample_idx = torch.randperm(len(node_neighbors))[:size]
                sampled_neighbors = node_neighbors[sample_idx]
            else:
                # 使用所有邻居
                sampled_neighbors = node_neighbors
            
            neighbors.append(sampled_neighbors)
        
        # 合并邻居
        neighbors = torch.cat(neighbors)
        
        # 去重
        neighbors = torch.unique(neighbors)
        
        # 添加到采样节点列表
        sampled_nodes.append(neighbors)
    
    # 构建子图
    all_sampled_nodes = torch.unique(torch.cat(sampled_nodes))
    
    # 重新索引
    node_mapping = {node.item(): idx for idx, node in enumerate(all_sampled_nodes)}
    
    # 构建子图邻接矩阵
    subgraph_adj = adjacency_matrix[all_sampled_nodes][:, all_sampled_nodes]
    
    return {
        "sampled_nodes": all_sampled_nodes,
        "node_mapping": node_mapping,
        "subgraph_adjacency": subgraph_adj
    }


