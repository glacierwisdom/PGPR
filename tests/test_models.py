import pytest
import torch
from models.node_representations import NodeRepresentation, NeighborRepresentation
from models.attention_mechanism import TargetConditionedAttention

class TestNodeRepresentation:
    def test_forward_shapes(self):
        model = NodeRepresentation(esm_dim=8, rnn_dim=4, node_dim=16, hidden_dim=32, dropout=0.0)
        out_single = model(torch.randn(8), torch.randn(4))
        assert out_single.shape == (16,)

        out_batch = model(torch.randn(3, 8), torch.randn(3, 4))
        assert out_batch.shape == (3, 16)

class TestNeighborRepresentation:
    def test_forward_shapes(self):
        model = NeighborRepresentation(neighbor_dim=8, edge_dim=6, node_dim=10, hidden_dim=32, dropout=0.0)
        out = model(torch.randn(5, 8), torch.randn(5, 6))
        assert out.shape == (5, 10)

class TestTargetConditionedAttention:
    def test_forward_shapes(self):
        attn = TargetConditionedAttention(node_dim=10, esm_dim=8, hidden_dim=16, num_relations=7)
        e_i = torch.randn(10)
        e_j = torch.randn(6, 10)
        target = torch.randn(8)
        weights, logits = attn(e_i, e_j, target, training=False)
        assert weights.shape == (6,)
        assert logits.shape[0] == 6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
