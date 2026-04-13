import torch
import os
import time
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """
    训练回调基类
    定义训练过程中的回调接口
    """
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """训练开始时调用"""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """训练结束时调用"""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每个epoch开始时调用"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每个epoch结束时调用"""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每个batch开始时调用"""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每个batch结束时调用"""
        pass
    
    @abstractmethod
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """验证开始时调用"""
        pass
    
    @abstractmethod
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """验证结束时调用"""
        pass
    
    @abstractmethod
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """保存模型时调用"""
        pass


class ModelCheckpoint(TrainingCallback):
    """
    模型检查点回调
    保存最佳模型和定期模型
    """
    
    def __init__(self, 
                 checkpoint_dir: str, 
                 monitor: str = 'val_loss', 
                 save_best_only: bool = True,
                 mode: str = 'min',
                 save_freq: int = 1,
                 save_weights_only: bool = False,
                 max_save_files: int = 5
                 ):
        """
        初始化模型检查点
        
        Args:
            checkpoint_dir (str): 检查点保存目录
            monitor (str): 监控指标
            save_best_only (bool): 是否只保存最佳模型
            mode (str): 监控模式 ('min' 或 'max')
            save_freq (int): 保存频率（epoch数）
            save_weights_only (bool): 是否只保存权重
            max_save_files (int): 最大保存文件数
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.max_save_files = max_save_files
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化最佳指标
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # 保存的文件列表
        self.saved_files = []
        
        logger.info(f"ModelCheckpoint初始化完成，监控指标：{monitor}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("训练开始，准备保存模型检查点")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("训练结束，所有模型检查点保存完成")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            return
        
        # 检查是否需要保存模型
        if epoch % self.save_freq == 0 or self.save_best_only:
            current_metric = logs.get(self.monitor)
            
            if current_metric is not None:
                # 检查是否是最佳模型
                if self._is_better(current_metric):
                    self.best_metric = current_metric
                    self._save_checkpoint(epoch, logs, is_best=True)
                elif not self.save_best_only:
                    self._save_checkpoint(epoch, logs, is_best=False)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._save_checkpoint(epoch, logs, is_best=False, custom=True)
    
    def _is_better(self, current_metric: float) -> bool:
        """检查当前指标是否更好"""
        if self.mode == 'min':
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def _save_checkpoint(self, 
                       epoch: int, 
                       logs: Dict[str, Any],
                       is_best: bool = False,
                       custom: bool = False
                       ) -> None:
        """保存模型检查点"""
        if int(logs.get("rank", 0) or 0) != 0:
            return
        # 生成文件名
        if is_best:
            filename = "best_model.pth"
        elif custom:
            filename = f"model_epoch_{epoch}_custom.pth"
        else:
            filename = f"model_epoch_{epoch}.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 获取模型
        model = logs.get('model')
        if model is None:
            logger.warning("没有找到模型，无法保存检查点")
            return
        model_to_save = getattr(model, "module", model)
        
        # 保存模型
        tmp_path = f"{filepath}.tmp.{os.getpid()}"
        if self.save_weights_only:
            torch.save(model_to_save.state_dict(), tmp_path)
            os.replace(tmp_path, filepath)
        else:
            safe_logs = {k: v for k, v in logs.items() if k != 'model'}
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'llm_classifier_state_dict': logs.get('llm_classifier_state_dict'),
                'llm_peft_state_dict': logs.get('llm_peft_state_dict'),
                'llm_state_dict': logs.get('llm_state_dict'),
                'optimizer_state_dict': logs.get('optimizer_state_dict'),
                'scheduler_state_dict': logs.get('scheduler_state_dict'),
                'best_metric': self.best_metric,
                'logs': safe_logs
            }, tmp_path)
            os.replace(tmp_path, filepath)
        
        logger.info(f"模型检查点已保存：{filepath}")
        
        # 管理保存的文件数量
        if not is_best and not custom:
            self.saved_files.append(filepath)
            
            if len(self.saved_files) > self.max_save_files:
                # 删除最旧的文件
                oldest_file = self.saved_files.pop(0)
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
                    logger.info(f"已删除旧模型检查点：{oldest_file}")


class EarlyStopping(TrainingCallback):
    """
    早停回调
    当监控指标不再改善时停止训练
    """
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True
                 ):
        """
        初始化早停回调
        
        Args:
            monitor (str): 监控指标
            patience (int): 没有改善的epoch数阈值
            min_delta (float): 最小改善量
            mode (str): 监控模式 ('min' 或 'max')
            verbose (bool): 是否打印日志
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        # 初始化计数器
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        
        logger.info(f"EarlyStopping初始化完成，监控指标：{monitor}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.counter = 0
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.early_stop:
            logger.info(f"早停触发，共训练了{self.counter}个没有改善的epoch")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            return
        
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            return
        
        if self._is_better(current_metric):
            self.counter = 0
            self.best_metric = current_metric
            if self.verbose:
                logger.info(f"早停监控指标改善：{self.monitor} = {current_metric:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"早停监控指标没有改善：第{self.counter}个epoch，{self.monitor} = {current_metric:.6f}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"早停触发！监控指标{self.monitor}在{self.patience}个epoch内没有改善")
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def _is_better(self, current_metric: float) -> bool:
        """检查当前指标是否更好"""
        if self.mode == 'min':
            return current_metric < self.best_metric - self.min_delta
        else:
            return current_metric > self.best_metric + self.min_delta


class TensorBoardLogger(TrainingCallback):
    """
    TensorBoard日志记录器
    将训练指标记录到TensorBoard
    """
    
    def __init__(self, log_dir: str, comment: str = ''):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir (str): 日志目录
            comment (str): 日志注释
        """
        self.log_dir = log_dir
        self.comment = comment
        self.writer = None
        
        logger.info(f"TensorBoardLogger初始化完成，日志目录：{log_dir}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # 初始化TensorBoard写入器
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        full_log_dir = os.path.join(self.log_dir, f"run-{timestamp}{self.comment}")
        
        self.writer = SummaryWriter(log_dir=full_log_dir)
        logger.info(f"TensorBoard写入器已启动，日志目录：{full_log_dir}")
        
        # 记录训练开始信息
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, 0)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard写入器已关闭")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.writer is None or logs is None:
            return
        
        # 记录epoch级别的指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.writer is None or logs is None:
            return
        
        # 记录batch级别的指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"batch/{key}", value, batch)
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.writer is None or logs is None:
            return
        
        # 记录验证指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, logs.get('epoch', 0))
    
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class LearningRateScheduler(TrainingCallback):
    """
    学习率调度回调
    管理学习率的更新
    """
    
    def __init__(self, scheduler, monitor: str = 'val_loss', verbose: bool = True):
        """
        初始化学习率调度器
        
        Args:
            scheduler: PyTorch学习率调度器
            monitor (str): 监控指标
            verbose (bool): 是否打印日志
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        
        logger.info("LearningRateScheduler初始化完成")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # 记录初始学习率
        if self.verbose:
            lr = self.scheduler.get_last_lr()[0]
            logger.info(f"初始学习率：{lr}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("学习率调度完成")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # 根据监控指标更新学习率
        if hasattr(self.scheduler, 'step'):
            if self.monitor in logs:
                self.scheduler.step(logs[self.monitor])
            else:
                self.scheduler.step()
            
            if self.verbose:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"第{epoch}个epoch结束，学习率更新为：{lr}")
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # 某些调度器需要每个batch更新
        if hasattr(self.scheduler, 'step') and hasattr(self.scheduler, 'step_size'):
            if hasattr(self.scheduler, 'gamma'):  # StepLR, MultiStepLR等
                pass
            elif hasattr(self.scheduler, 'T_max'):  # CosineAnnealingLR等
                pass
            else:
                # 其他调度器可能需要batch级别的更新
                pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class ProgressBar(TrainingCallback):
    """
    进度条回调
    显示训练进度
    """
    
    def __init__(self, total_epochs: int, total_batches: int):
        """
        初始化进度条
        
        Args:
            total_epochs (int): 总epoch数
            total_batches (int): 每个epoch的总batch数
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        
        logger.info("ProgressBar初始化完成")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"开始训练，共{self.total_epochs}个epoch，每个epoch{self.total_batches}个batch")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("训练完成！")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"\nEpoch {epoch+1}/{self.total_epochs}")
        self.current_epoch = epoch
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs:
            # 打印epoch结束信息
            metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in logs.items() if isinstance(v, (int, float))])
            logger.info(f"Epoch {epoch+1}/{self.total_epochs} - {metrics_str}")
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.current_batch = batch
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # 打印batch进度
        if batch % 10 == 0 or batch == self.total_batches - 1:
            progress = (batch + 1) / self.total_batches * 100
            logger.info(f"Batch {batch+1}/{self.total_batches} ({progress:.1f}%)")
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logger.info("开始验证...")
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs:
            metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in logs.items() if isinstance(v, (int, float))])
            logger.info(f"验证结果 - {metrics_str}")
    
    def on_save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


def get_callbacks(config: Dict[str, Any]) -> List[TrainingCallback]:
    """
    获取回调列表
    
    Args:
        config (Dict[str, Any]): 配置字典
        
    Returns:
        List[TrainingCallback]: 回调列表
    """
    callbacks = []
    
    # 获取回调配置部分
    callbacks_config = config.get('callbacks', {})
    
    # 添加模型检查点回调
    # 检查 callbacks.model_checkpoint 开关或 root.checkpoint.enabled
    enable_checkpoint = callbacks_config.get('model_checkpoint', config.get('checkpoint', {}).get('enabled', True))
    
    if enable_checkpoint:
        # 优先使用 callbacks.checkpoint，其次使用 root.checkpoint
        checkpoint_config = callbacks_config.get('checkpoint', config.get('checkpoint', {}))
        training_cfg = config.get('training', {}) or {}
        paths_cfg = config.get('paths', {}) or {}
        training_checkpoint_dir = training_cfg.get('checkpoint_dir')
        checkpoint_dir = checkpoint_config.get('dir') or training_checkpoint_dir or paths_cfg.get('checkpoints_dir', 'checkpoints')
        save_top_k = checkpoint_config.get('save_top_k')
        max_save_files = checkpoint_config.get('max_save_files')
        if max_save_files is None and isinstance(save_top_k, int):
            max_save_files = save_top_k
        
        callbacks.append(ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            save_best_only=checkpoint_config.get('save_best_only', True),
            mode=checkpoint_config.get('mode', 'min'),
            save_freq=checkpoint_config.get('save_freq', 1),
            save_weights_only=checkpoint_config.get('save_weights_only', False),
            max_save_files=max_save_files if max_save_files is not None else 5
        ))
    
    # 添加早停回调
    enable_early_stopping = callbacks_config.get('early_stopping', config.get('early_stopping', {}).get('enabled', False))
    
    if enable_early_stopping:
        training_cfg = config.get('training', {}) or {}

        if isinstance(callbacks_config.get('early_stopping'), dict):
            es_config = callbacks_config.get('early_stopping')
        else:
            es_config = callbacks_config.get('early_stopping_config', config.get('early_stopping', {}))
            if not isinstance(es_config, dict):
                es_config = config.get('early_stopping', {})

        patience = es_config.get('patience')
        if patience is None:
            patience = training_cfg.get('early_stopping_patience', 10)

        min_delta = es_config.get('min_delta')
        if min_delta is None:
            min_delta = training_cfg.get('early_stopping_min_delta', 0.0)

        callbacks.append(EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=patience,
            min_delta=min_delta,
            mode=es_config.get('mode', 'min'),
            verbose=es_config.get('verbose', True)
        ))
    
    # 添加TensorBoard日志回调
    enable_tensorboard = callbacks_config.get('tensorboard_logger', config.get('tensorboard', {}).get('enabled', True))
    
    if enable_tensorboard:
        tensorboard_config = callbacks_config.get('tensorboard', config.get('tensorboard', {}))
        callbacks.append(TensorBoardLogger(
            log_dir=tensorboard_config.get('log_dir', config.get('paths', {}).get('logs_dir', 'logs')),
            comment=tensorboard_config.get('comment', '')
        ))
    
    # 添加进度条回调
    enable_progress_bar = callbacks_config.get('progress_bar', config.get('progress_bar', {}).get('enabled', True))
    
    if enable_progress_bar:
        total_epochs = config.get('training', {}).get('epochs', 100)
        # 这里假设我们知道总batch数，实际使用时可能需要调整
        callbacks.append(ProgressBar(total_epochs=total_epochs, total_batches=100))
    
    logger.info(f"已创建{len(callbacks)}个回调")
    
    return callbacks
