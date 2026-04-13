
import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm.wrapper import LLMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_llm_loading():
    logger.info("Testing LLM Wrapper loading with TinyLlama...")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        wrapper = LLMWrapper(
            model_name=model_name,
            num_relations=8,
            use_lora=True,
            lora_r=8,
            load_in_8bit=False, # Disable 8bit for simple verification/CPU
            device=device,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        logger.info("Model loaded successfully.")
        
        # Test forward pass
        texts = ["Protein A interacts with Protein B.", "Protein C binds to Protein D."]
        logger.info(f"Testing forward pass with {len(texts)} samples...")
        
        outputs = wrapper(texts)
        logits = outputs['logits']
        logger.info(f"Output logits shape: {logits.shape}")
        
        assert logits.shape == (2, 8), f"Expected logits shape (2, 8), got {logits.shape}"
        logger.info("Forward pass successful.")
        
        # Test predict
        logger.info("Testing predict method...")
        preds, probs = wrapper.predict(texts)
        logger.info(f"Predictions: {preds}")
        logger.info(f"Probabilities shape: {probs.shape}")
        
        logger.info("Verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    verify_llm_loading()
