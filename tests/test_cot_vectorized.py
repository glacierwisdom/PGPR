import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock
from torch_geometric.data import Data
from models.cot_generator import ExploratoryCOTGenerator
from models.node_representations import NodeRepresentation, NeighborRepresentation
from models.attention_mechanism import TargetConditionedAttention
from models.rnn_encoder import PathRNNEncoder

class TestCOTVectorized:
    @pytest.fixture
    def setup_data(self):
        # 1. 创建模拟图数据
        num_nodes = 10
        esm_dim = 320
        edge_dim = 8
        
        x = torch.randn(num_nodes, esm_dim)
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7]
        ], dtype=torch.long)
        edge_attr = torch.randn(edge_index.size(1), edge_dim)
        
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # 2. 创建模拟组件
        node_dim = 256
        rnn_dim = 128
        
        node_rep = MagicMock(spec=NodeRepresentation)
        node_rep.node_dim = node_dim
        # 模拟 node_rep 返回张量
        node_rep.side_effect = lambda x, h: torch.randn(x.size(0), node_dim)
        
        neighbor_rep = MagicMock(spec=NeighborRepresentation)
        neighbor_rep.side_effect = lambda x, e: torch.randn(x.size(0), node_dim)
        
        attention = MagicMock(spec=TargetConditionedAttention)
        attention.esm_dim = esm_dim
        # 模拟 attention 返回权重和 logits
        def mock_attention_forward(e_i, e_j, target, training=True, prev_visited=None):
            num_neighbors = e_j.size(0)
            weights = torch.softmax(torch.randn(num_neighbors), dim=0)
            logits = torch.randn(num_neighbors, 8)
            return weights, logits
        attention.side_effect = mock_attention_forward
        
        rnn_encoder = MagicMock(spec=PathRNNEncoder)
        rnn_encoder.rnn_dim = rnn_dim
        rnn_encoder.edge_dim = edge_dim
        rnn_encoder.get_initial_state.return_value = torch.zeros(1, 1, rnn_dim)
        # 模拟 update_path_encoding 返回新历史和新状态
        rnn_encoder.update_path_encoding.side_effect = lambda h, v, e, s: (
            torch.randn(h.size(0), rnn_dim),
            torch.randn(h.size(0), 1, rnn_dim)
        )
        
        return {
            'graph_data': graph_data,
            'node_rep': node_rep,
            'neighbor_rep': neighbor_rep,
            'attention': attention,
            'rnn_encoder': rnn_encoder,
            'esm_dim': esm_dim,
            'node_dim': node_dim,
            'rnn_dim': rnn_dim,
            'edge_dim': edge_dim
        }

    def test_get_action_probabilities_batch(self, setup_data):
        """测试向量化的 get_action_probabilities 是否能处理批量数据"""
        data = setup_data
        
        # 创建生成器
        generator = ExploratoryCOTGenerator(
            node_representation=data['node_rep'],
            neighbor_representation=data['neighbor_rep'],
            attention_mechanism=data['attention'],
            rnn_encoder=data['rnn_encoder'],
            max_steps=5
        )
        
        # 设置当前图和目标特征
        batch_size = 4
        seq_len = 3
        generator._current_graph = data['graph_data']
        generator._current_target_esms = torch.randn(batch_size, data['esm_dim'])
        
        # 模拟状态和动作
        states = torch.randn(batch_size, data['node_dim'])
        # 动作是节点索引序列
        actions = torch.tensor([
            [0, 1, 3],
            [0, 2, 4],
            [1, 3, 5],
            [2, 4, 6]
        ], dtype=torch.long)
        
        # 调用方法
        log_probs, entropies = generator.get_action_probabilities(states, actions)
        
        # 验证输出
        assert log_probs.shape == (batch_size,)
        assert entropies.shape == (batch_size,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropies).any()
        
        # 验证组件调用次数
        # 对于每个样本，每个步骤（seq_len - 1）都会调用组件
        # 注意：由于向量化，部分调用是批量的
        assert data['node_rep'].call_count == (seq_len - 1)
        # attention 和 neighbor_rep 在循环内，针对每个 batch item 调用
        assert data['attention'].call_count == batch_size * (seq_len - 1)
        assert data['neighbor_rep'].call_count == batch_size * (seq_len - 1)
        assert data['rnn_encoder'].update_path_encoding.call_count == (seq_len - 1)

    def test_gradient_flow(self, setup_data):
        """测试梯度是否能正确流过 get_action_probabilities"""
        data = setup_data
        
        # 使用真实的网络模块代替 Mock，以测试梯度
        class SimpleNodeRep(nn.Module):
            def __init__(self, esm_dim, rnn_dim, node_dim):
                super().__init__()
                self.fc = nn.Linear(esm_dim + rnn_dim, node_dim)
                self.node_dim = node_dim
            def forward(self, x, h):
                return self.fc(torch.cat([x, h], dim=-1))
                
        class SimpleNeighborRep(nn.Module):
            def __init__(self, esm_dim, edge_dim, node_dim):
                super().__init__()
                self.fc = nn.Linear(esm_dim + edge_dim, node_dim)
            def forward(self, x, e):
                return self.fc(torch.cat([x, e], dim=-1))
                
        class SimpleAttention(nn.Module):
            def __init__(self, node_dim, esm_dim):
                super().__init__()
                self.fc = nn.Linear(node_dim * 2 + esm_dim, 1)
                self.esm_dim = esm_dim
            def forward(self, e_i, e_j, target, training=True):
                # e_i: [node_dim], e_j: [N, node_dim], target: [esm_dim]
                num_neighbors = e_j.size(0)
                e_i_expanded = e_i.unsqueeze(0).expand(num_neighbors, -1)
                target_expanded = target.unsqueeze(0).expand(num_neighbors, -1)
                combined = torch.cat([e_i_expanded, e_j, target_expanded], dim=-1)
                logits = self.fc(combined).squeeze(-1)
                weights = torch.softmax(logits, dim=0)
                return weights, torch.randn(num_neighbors, 8) # logits for relations
                
        class SimpleRNN(nn.Module):
            def __init__(self, node_dim, edge_dim, rnn_dim):
                super().__init__()
                self.rnn = nn.GRUCell(rnn_dim, rnn_dim) # 改为 rnn_dim, rnn_dim
                self.rnn_dim = rnn_dim
                self.edge_dim = edge_dim
            def get_initial_state(self, batch_size, device):
                return torch.zeros(batch_size, self.rnn_dim).to(device)
            def update_path_encoding(self, h, v_feat, e_feat, state):
                # v_feat: [B, esm_dim], e_feat: [B, edge_dim]
                input_feat = torch.cat([v_feat[:, :10], e_feat], dim=-1) # 10 + 8 = 18
                if not hasattr(self, 'proj'):
                    self.proj = nn.Linear(input_feat.size(-1), self.rnn_dim).to(v_feat.device)
                new_h = self.rnn(self.proj(input_feat), state)
                return new_h, new_h

        node_rep = SimpleNodeRep(320, 128, 256)
        neighbor_rep = SimpleNeighborRep(320, 8, 256)
        attention = SimpleAttention(256, 320)
        rnn_encoder = SimpleRNN(256, 8, 128)
        
        generator = ExploratoryCOTGenerator(
            node_representation=node_rep,
            neighbor_representation=neighbor_rep,
            attention_mechanism=attention,
            rnn_encoder=rnn_encoder,
            max_steps=5
        )
        
        batch_size = 2
        seq_len = 3
        generator._current_graph = data['graph_data']
        generator._current_target_esms = torch.randn(batch_size, 320)
        
        states = torch.randn(batch_size, 256)
        actions = torch.tensor([
            [0, 1, 3],
            [0, 2, 4]
        ], dtype=torch.long)
        
        log_probs, entropies = generator.get_action_probabilities(states, actions)
        loss = -log_probs.mean()
        loss.backward()
        
        # 检查梯度是否不为空
        has_grad = False
        for name, param in generator.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.norm(param.grad) > 0:
                    has_grad = True
                    break
        
        assert has_grad, "No parameters have non-zero gradients"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
