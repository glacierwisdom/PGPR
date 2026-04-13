import pytest
import torch
import torch.nn as nn
from training.trainer import ExploratoryPPITrainer

class TestTrainerConfigCompatibility:
    def test_cosine_annealing_scheduler_type_is_accepted(self):
        config = {
            "device": {"device_type": "cpu"},
            "distributed": {"use_distributed": False},
            "callbacks": {"early_stopping": False, "model_checkpoint": False, "tensorboard_logger": False},
            "llm": {"train_llm": True, "train_backbone": False},
            "optimizer": {"type": "adamw", "lr": 1e-4, "weight_decay": 0.0},
            "lr_scheduler": {"type": "cosine_annealing", "T_max": 10, "eta_min": 1e-6},
        }
        trainer = ExploratoryPPITrainer(config)
        trainer.llm_wrapper = type(
            "DummyLLM",
            (),
            {"relation_classifier": nn.Linear(4, 7), "model": None, "use_lora": False},
        )()
        trainer._build_optimizer_and_scheduler()
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
