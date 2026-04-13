import pytest
import os
import tempfile
import torch
from data.dataset import PPIDataset
import pandas as pd

@pytest.fixture
def temp_tsv_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        temp_path = f.name
        df = pd.DataFrame(
            {
                "protein_a": [
                    "MKTVRQERLKSIVRILERSK",
                    "AAAAAAAAAAAAAAAAAAAA",
                    "MKTVRQERLKSIVRILERSK",
                ],
                "protein_b": [
                    "GGGGGGGGGGGGGGGGGGGG",
                    "CCCCCCCCCCCCCCCCCCCC",
                    "TTTTTTTTTTTTTTTTTTTT",
                ],
                "label": [
                    "[1,0,0,0,0,0,0]",
                    "[0,1,0,0,0,0,0]",
                    "[1,0,0,0,0,0,0]",
                ],
            }
        )
        df.to_csv(temp_path, sep="\t", index=False)

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)

class TestPPIDataset:
    """测试PPIDataset类"""
    
    def test_dataset_load_and_id_stability(self, temp_tsv_file):
        dataset = PPIDataset(temp_tsv_file)
        assert len(dataset) == 3

        item0 = dataset[0]
        item2 = dataset[2]
        assert item0["protein_a"] == item2["protein_a"]
        assert item0["protein_a_id"] == item2["protein_a_id"]
        assert item0["label"].shape[-1] == 7
    
    def test_get_protein_sequences(self, temp_tsv_file):
        dataset = PPIDataset(temp_tsv_file)
        seqs = dataset.get_protein_sequences()
        assert isinstance(seqs, dict)
        assert len(seqs) >= 4
        for pid, seq in seqs.items():
            assert isinstance(pid, str)
            assert isinstance(seq, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
