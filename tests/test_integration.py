import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cot_generator import ExploratoryCOTGenerator
from models.component_builder import ComponentBuilder
from llm.wrapper import LLMWrapper
from training.trainer import ExploratoryPPITrainer
from training.reward_calculator import MultiScaleRewardCalculator
from training.rl_framework import PPOTrainer
from llm.prompt_designer import PromptDesigner

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration_train_step():
    """
    集成测试：验证 _train_step 的完整流程
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. 模拟配置
    config = {
        'model': {
            'embedding_dim': 64,
            'esm_dim': 128,
            'hidden_dim': 64,
            'num_edge_features': 16,
            'num_edge_relations': 8,
            'rnn_history_dim': 64,
            'rnn_hidden_dim': 64,
            'num_node_features': 128
        },
        'gnn_ppi': {
            'node_representation': {
                'output_dim': 64,
                'hidden_dim': 64,
                'esm_dim': 128
            },
            'graph_attention': {
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4
            }
        },
        'reinforcement_learning': {
            'learning_rate': 1e-4,
            'epochs': 1,
            'batch_size': 2
        },
        'llm': {
            'train_llm': True,
            'model_name': 'mock-llm',
            'num_relations': 8
        },
        'training': {
            'use_amp': False
        },
        'max_steps': 5,
        'beam_size': 3
    }

    # 2. 模拟组件 (Mock some parts to avoid loading heavy models)
    class MockLLMWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(10, 10) # Dummy
            self.relation_classifier = nn.Linear(128, 8)
            self.device = device
            self.hidden_dim = 128
        
        def forward(self, texts, labels=None):
            batch_size = len(texts)
            logits = torch.randn(batch_size, 8, device=device, requires_grad=True)
            return {'logits': logits, 'loss': torch.tensor(0.0, device=device, requires_grad=True)}
            
        def predict(self, texts, return_type='probabilities'):
            logits = torch.randn(len(texts), 8, device=device, requires_grad=True)
            predictions = (logits > 0).float()
            if return_type == 'probabilities':
                return predictions, torch.sigmoid(logits.detach())
            if return_type == 'logits':
                return predictions, logits
            return predictions, predictions

    class MockESMEncoder:
        def __init__(self):
            self.device = device
        def get_batch_embeddings(self, seqs, batch_size=None):
            return [torch.randn(128).to(device) for _ in seqs]

    # 3. 初始化 Trainer 并手动注入 Mock 组件
    trainer = ExploratoryPPITrainer(config)
    trainer.device = device
    
    # 注入 Mock 组件
    trainer.llm_wrapper = MockLLMWrapper().to(device)
    trainer.esm_encoder = MockESMEncoder()
    trainer.prompt_designer = PromptDesigner()
    
    # 初始化 COT Generator
    trainer.cot_generator = ComponentBuilder.build_cot_generator(config, device)
    
    # 初始化 Reward Calculator
    trainer.reward_calculator = MultiScaleRewardCalculator()
    
    # 初始化 Value Network 和 PPO Trainer
    node_dim = 64
    esm_dim = 128
    class ValueNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.fc(x)
            
    value_model = ValueNetwork(node_dim + esm_dim, 64).to(device)
    trainer.ppo_trainer = PPOTrainer(
        policy_model=trainer.cot_generator,
        value_model=value_model,
        device=str(device)
    )
    
    # 初始化主优化器 (用于 Mock LLM)
    trainer.optimizer = torch.optim.Adam(trainer.llm_wrapper.parameters(), lr=1e-4)

    # 4. 构造模拟图数据
    num_nodes = 10
    x = torch.randn(num_nodes, 128) # ESM features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5], 
                               [1, 0, 2, 1, 3, 2, 5, 4]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 16)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.protein_id_to_idx = {f"P{i}": i for i in range(num_nodes)}
    trainer.graph = graph.to(device)

    # 5. 构造模拟 Batch
    batch = {
        'protein_a_id': ['P0', 'P1'],
        'protein_b_id': ['P5', 'P4'],
        'label': torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            device=device,
            dtype=torch.float32,
        )
    }

    # 6. 运行 _train_step
    logger.info("Running _train_step...")
    try:
        loss, avg_reward = trainer._train_step(batch)
        logger.info(f"Step successful. Loss: {loss:.4f}, Avg Reward: {avg_reward:.4f}")

    except Exception as e:
        logger.error(f"Step failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_integration_train_step()
