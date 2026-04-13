import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import os

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    经验回放缓冲区
    存储PPO训练所需的经验数据
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化经验回放缓冲区
        
        Args:
            max_size (int): 缓冲区最大容量
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
        logger.info(f"ExperienceBuffer初始化完成，最大容量：{max_size}")
    
    def add(self, experience: Dict[str, torch.Tensor]):
        """
        添加经验到缓冲区
        
        Args:
            experience (Dict[str, torch.Tensor]): 经验数据
        """
        self.buffer.append(experience)
    
    def add_batch(self, experiences: List[Dict[str, torch.Tensor]]):
        """
        批量添加经验到缓冲区
        
        Args:
            experiences (List[Dict[str, torch.Tensor]]): 经验数据列表
        """
        for experience in experiences:
            self.add(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        从缓冲区中采样一批经验
        
        Args:
            batch_size (int): 采样大小
            
        Returns:
            Dict[str, torch.Tensor]: 采样的批量经验
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 随机采样
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # 收集采样的经验
        sampled_experiences = [self.buffer[i] for i in indices]
        
        # 合并经验数据
        batch = {}
        for key in sampled_experiences[0].keys():
            batch[key] = torch.stack([exp[key] for exp in sampled_experiences])
        
        return batch
    
    def clear(self):
        """
        清空缓冲区
        """
        self.buffer.clear()
        logger.info("ExperienceBuffer已清空")
    
    def size(self) -> int:
        """
        获取缓冲区当前大小
        
        Returns:
            int: 缓冲区当前大小
        """
        return len(self.buffer)


class PPOTrainer:
    """
    PPO强化学习框架
    实现标准PPO算法
    """
    
    def __init__(self, 
                 policy_model: nn.Module,
                 value_model: nn.Module,
                 learning_rate: float = 3e-4,
                 clip_param: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_amp: bool = False,
                 device: str = "cuda:0"
                 ):
        """
        初始化PPO训练器
        
        Args:
            policy_model (nn.Module): 策略网络
            value_model (nn.Module): 价值网络
            learning_rate (float): 学习率
            clip_param (float): PPO裁剪参数
            epochs (int): 每个批次的更新次数
            batch_size (int): 批次大小
            gamma (float): 折扣因子
            gae_lambda (float): GAE参数
            entropy_coef (float): 熵正则化系数
            value_loss_coef (float): 价值损失系数
            max_grad_norm (float): 梯度裁剪最大值
            use_amp (bool): 是否使用混合精度
            device (str): 设备
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.clip_param = clip_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.device = device
        
        # 移动模型到设备
        self.policy_model = self.policy_model.to(device)
        self.value_model = self.value_model.to(device)
        
        # 创建优化器
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)
        
        # 创建经验缓冲区
        self.buffer = ExperienceBuffer()
        
        # 混合精度训练
        amp = getattr(torch, "amp", None)
        grad_scaler = getattr(amp, "GradScaler", None) if amp is not None else None
        if grad_scaler is not None:
            self.scaler = grad_scaler('cuda', enabled=use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        logger.info(f"PPOTrainer初始化完成，使用设备：{device}")
    
    def collect_experience(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        收集经验
        
        Args:
            batch_data (Dict[str, torch.Tensor]): 批量数据
            
        Returns:
            Dict[str, torch.Tensor]: 收集的经验
        """
        # 将数据移动到设备
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        
        # 获取数据
        states = batch_data.get('states')
        actions = batch_data.get('actions')
        rewards = batch_data.get('rewards')
        next_states = batch_data.get('next_states')
        masks = batch_data.get('masks')
        
        # 计算当前策略的动作概率和熵
        with torch.no_grad():
            log_probs, entropies = self.policy_model.get_action_probabilities(states, actions)
            values = self.value_model(states)
            next_values = self.value_model(next_states)
        
        # 计算优势函数（GAE）
        advantages = self.compute_advantages(rewards, values, next_values, masks)
        
        # 计算目标值
        returns = advantages + values
        
        # 构造经验数据
        experience = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
        
        # 添加到经验缓冲区
        self.buffer.add(experience)
        
        return experience
    
    def collect_experience_batch(self, 
                                 batch_data: Dict[str, torch.Tensor],
                                 provided_log_probs: Optional[torch.Tensor] = None,
                                 model: Optional[torch.nn.Module] = None
                                 ) -> List[Dict[str, torch.Tensor]]:
        """
        批量收集经验
        
        Args:
            batch_data (Dict[str, torch.Tensor]): 批量数据
            provided_log_probs (Optional[torch.Tensor]): 预先提供的对数概率
            model (Optional[torch.nn.Module]): 可选模型（用于 DDP 兼容）
            
        Returns:
            List[Dict[str, torch.Tensor]]: 收集的经验列表
        """
        # 兼容分布式：优先使用传入的模型，否则使用 self.policy_model
        policy_model = model if model is not None else self.policy_model

        # 将数据移动到设备
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        
        # 获取数据
        states = batch_data.get('states')
        actions = batch_data.get('actions')
        rewards = batch_data.get('rewards')
        next_states = batch_data.get('next_states')
        masks = batch_data.get('masks')
        
        # 计算当前策略的动作概率和熵
        with torch.no_grad():
            if provided_log_probs is not None:
                log_probs = provided_log_probs.to(self.device).detach()
                entropies = torch.zeros_like(log_probs)
            else:
                log_probs, entropies = policy_model.get_action_probabilities(states, actions)
            
            values = self.value_model(states)
            next_values = self.value_model(next_states)
        
        # 计算优势函数（GAE）
        advantages = self.compute_advantages(rewards, values, next_values, masks)
        
        # 计算目标值
        returns = advantages + values
        
        # 构造经验数据列表
        experiences = []
        for i in range(states.size(0)):
            experience = {
                'states': states[i],
                'actions': actions[i],
                'log_probs': log_probs[i],
                'advantages': advantages[i],
                'returns': returns[i],
                'values': values[i]
            }
            experiences.append(experience)
        
        # 批量添加到经验缓冲区
        self.buffer.add_batch(experiences)
        
        return experiences
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          next_values: torch.Tensor,
                          masks: torch.Tensor
                          ) -> torch.Tensor:
        """
        计算优势函数（GAE）
        
        Args:
            rewards (torch.Tensor): 奖励 [batch_size]
            values (torch.Tensor): 价值估计 [batch_size]
            next_values (torch.Tensor): 下一个状态的价值估计 [batch_size]
            masks (torch.Tensor): 掩码 [batch_size]
            
        Returns:
            torch.Tensor: 优势函数 [batch_size]
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if next_values.dim() == 1:
            next_values = next_values.unsqueeze(-1)

        if rewards.dim() == 2:
            return rewards + self.gamma * next_values * masks - values

        if rewards.dim() != 3:
            raise ValueError(f"Unsupported rewards dim: {rewards.dim()}")

        delta = rewards + self.gamma * next_values * masks - values
        advantages = torch.zeros_like(delta)
        last_gae = torch.zeros_like(delta[0])
        for t in reversed(range(delta.size(0))):
            last_gae = delta[t] + self.gamma * self.gae_lambda * masks[t] * last_gae
            advantages[t] = last_gae
        return advantages
    
    def update_policy(self, model: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Args:
            model (Optional[torch.nn.Module]): 可选模型（用于 DDP 兼容）
            
        Returns:
            Dict[str, float]: 训练指标
        """
        # 如果缓冲区数据不足一个批次，则跳过更新（累积经验）
        if self.buffer.size() < self.batch_size:
            # logger.debug(f"经验缓冲区数据不足 ({self.buffer.size()}/{self.batch_size})，跳过策略更新")
            return {}
        
        # 兼容分布式
        policy_model = model if model is not None else self.policy_model
        
        # 计算批次数量
        num_batches = max(self.buffer.size() // self.batch_size, 1)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(self.epochs):
            for batch_idx in range(num_batches):
                # 从缓冲区采样
                batch = self.buffer.sample(self.batch_size)
                
                # 获取批次数据
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # 标准化优势函数
                advantages = advantages.float()
                advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
                adv_std = advantages.std()
                if torch.isfinite(adv_std) and adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                advantages = advantages.clamp(min=-5.0, max=5.0)
                
                # 更新策略网络
                states_for_policy = states.detach()
                states_for_value = states.detach()

                policy_loss, value_loss = self._update_policy_and_value_networks(
                    states_for_policy,
                    actions,
                    old_log_probs,
                    advantages,
                    returns,
                    model=policy_model
                )
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
        
        # 计算平均损失
        avg_policy_loss = total_policy_loss / (self.epochs * num_batches)
        avg_value_loss = total_value_loss / (self.epochs * num_batches)
        avg_total_loss = avg_policy_loss + avg_value_loss
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 构造训练指标
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss
        }
        
        logger.info(f"PPO策略更新完成，总损失：{avg_total_loss:.6f}")
        
        return metrics

    def _update_policy_and_value_networks(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        model: Optional[torch.nn.Module] = None
    ) -> Tuple[float, float]:
        policy_model = model if model is not None else self.policy_model

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        old_log_probs = old_log_probs.detach().float()
        advantages = advantages.detach().float()
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = returns.detach().float()
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            log_probs, entropies = policy_model.get_action_probabilities(states, actions)
            log_probs = log_probs.float()
            entropies = entropies.float()
            values = self.value_model(states).float()

            if log_probs.dim() == 1:
                log_probs = log_probs.unsqueeze(-1)
            if old_log_probs.dim() == 1:
                old_log_probs = old_log_probs.unsqueeze(-1)
            if entropies.dim() == 1:
                entropies = entropies.unsqueeze(-1)
            if advantages.dim() == 1:
                advantages = advantages.unsqueeze(-1)
            if returns.dim() == 1:
                returns = returns.unsqueeze(-1)
            if values.dim() == 1:
                values = values.unsqueeze(-1)

            log_ratio = (log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            ratio = torch.clamp(ratio, min=0.0, max=10.0)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropies.mean()

            value_loss = self.value_loss_coef * F.smooth_l1_loss(values, returns)

            total_loss = policy_loss + value_loss
            if not torch.isfinite(total_loss):
                return 0.0, 0.0

        self.scaler.scale(total_loss).backward(retain_graph=True)

        self.scaler.unscale_(self.policy_optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.policy_optimizer)

        self.scaler.unscale_(self.value_optimizer)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.value_optimizer)

        self.scaler.update()

        return float(policy_loss.item()), float(value_loss.item())

    def _update_policy_network(self, 
                              states: torch.Tensor,
                              actions: torch.Tensor,
                              old_log_probs: torch.Tensor,
                              advantages: torch.Tensor,
                              model: Optional[torch.nn.Module] = None
                              ) -> float:
        """
        更新策略网络
        
        Args:
            states (torch.Tensor): 状态 [batch_size, state_dim]
            actions (torch.Tensor): 动作 [batch_size, action_dim]
            old_log_probs (torch.Tensor): 旧的动作概率 [batch_size]
            advantages (torch.Tensor): 优势函数 [batch_size]
            model (Optional[torch.nn.Module]): 可选模型（用于 DDP 兼容）
            
        Returns:
            float: 策略损失
        """
        # 兼容分布式
        policy_model = model if model is not None else self.policy_model
        
        self.policy_optimizer.zero_grad()
        
        old_log_probs = old_log_probs.detach().float()
        advantages = advantages.detach().float()
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            log_probs, entropies = policy_model.get_action_probabilities(states, actions)
            log_probs = log_probs.float()
            entropies = entropies.float()

            if log_probs.dim() == 1:
                log_probs = log_probs.unsqueeze(-1)
            if old_log_probs.dim() == 1:
                old_log_probs = old_log_probs.unsqueeze(-1)
            if entropies.dim() == 1:
                entropies = entropies.unsqueeze(-1)
            if advantages.dim() == 1:
                advantages = advantages.unsqueeze(-1)

            log_ratio = (log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages

            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropies.mean()
            if not torch.isfinite(policy_loss):
                return 0.0
        
        # 反向传播和优化
        self.scaler.scale(policy_loss).backward()
        self.scaler.unscale_(self.policy_optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.policy_optimizer)
        self.scaler.update()
        
        return policy_loss.item()
    
    def _update_value_network(self, 
                             states: torch.Tensor,
                             returns: torch.Tensor
                             ) -> float:
        """
        更新价值网络
        
        Args:
            states (torch.Tensor): 状态 [batch_size, state_dim]
            returns (torch.Tensor): 目标值 [batch_size]
            
        Returns:
            float: 价值损失
        """
        self.value_optimizer.zero_grad()
        returns = returns.detach().float()
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # 获取当前价值估计
            values = self.value_model(states)
            values = values.float()
            
            # 计算价值损失
            value_loss = self.value_loss_coef * (returns - values).pow(2).mean()
            if not torch.isfinite(value_loss):
                return 0.0
        
        # 反向传播和优化
        self.scaler.scale(value_loss).backward()
        self.scaler.unscale_(self.value_optimizer)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.value_optimizer)
        self.scaler.update()
        
        return value_loss.item()
    
    def train_step(self, 
                   batch_data: Dict[str, torch.Tensor],
                   provided_log_probs: Optional[torch.Tensor] = None,
                   model: Optional[torch.nn.Module] = None
                   ) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            batch_data (Dict[str, torch.Tensor]): 批量数据
            provided_log_probs (Optional[torch.Tensor]): 预先提供的对数概率
            model (Optional[torch.nn.Module]): 可选模型（用于 DDP 兼容）
            
        Returns:
            Dict[str, float]: 训练指标
        """
        # 收集经验
        self.collect_experience_batch(batch_data, provided_log_probs, model=model)
        
        # 更新策略
        metrics = self.update_policy(model=model)
        
        return metrics
    
    def save_model(self, path: str, epoch: int):
        """
        保存模型
        
        Args:
            path (str): 保存路径
            epoch (int): 当前epoch
        """
        # 创建保存目录
        os.makedirs(path, exist_ok=True)
        
        # 保存策略网络
        policy_path = os.path.join(path, f"policy_model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
        }, policy_path)
        
        # 保存价值网络
        value_path = os.path.join(path, f"value_model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.value_model.state_dict(),
            'optimizer_state_dict': self.value_optimizer.state_dict(),
        }, value_path)
        
        logger.info(f"模型已保存到：{path}")
    
    def load_model(self, path: str, epoch: int):
        """
        加载模型
        
        Args:
            path (str): 加载路径
            epoch (int): 加载的epoch
        """
        # 加载策略网络
        policy_path = os.path.join(path, f"policy_model_epoch_{epoch}.pth")
        if os.path.exists(policy_path):
            checkpoint = torch.load(policy_path, map_location=self.device)
            self.policy_model.load_state_dict(checkpoint['model_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"策略网络已加载：{policy_path}")
        else:
            logger.warning(f"策略网络文件不存在：{policy_path}")
        
        # 加载价值网络
        value_path = os.path.join(path, f"value_model_epoch_{epoch}.pth")
        if os.path.exists(value_path):
            checkpoint = torch.load(value_path, map_location=self.device)
            self.value_model.load_state_dict(checkpoint['model_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"价值网络已加载：{value_path}")
        else:
            logger.warning(f"价值网络文件不存在：{value_path}")
    
    def set_learning_rate(self, lr: float):
        """
        设置学习率
        
        Args:
            lr (float): 新的学习率
        """
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr
        
        logger.info(f"学习率已更新为：{lr}")
    
    def get_learning_rate(self) -> float:
        """
        获取当前学习率
        
        Returns:
            float: 当前学习率
        """
        return self.policy_optimizer.param_groups[0]['lr']
    
    def eval(self):
        """
        设置模型为评估模式
        """
        self.policy_model.eval()
        self.value_model.eval()
        logger.info("PPOTrainer已设置为评估模式")
    
    def train(self):
        """
        设置模型为训练模式
        """
        self.policy_model.train()
        self.value_model.train()
        logger.info("PPOTrainer已设置为训练模式")
