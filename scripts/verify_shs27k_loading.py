import sys
import os
sys.path.append(os.getcwd())
from data.dataset import PPIDataset
import torch
import logging

logging.basicConfig(level=logging.INFO)

def verify_loading():
    print("Verifying SHS27k dataset loading...")
    
    data_path = "data/processed/shs27k_train.csv"
    try:
        dataset = PPIDataset(data_path)
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
        
        # Check first item
        item = dataset[0]
        print("First item keys:", item.keys())
        print("Label shape:", item['label'].shape)
        print("Protein A sequence length:", len(item['protein_a']))
        print("Protein B sequence length:", len(item['protein_b']))
        
        print("\nVerification successful!")
        return True
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False

if __name__ == "__main__":
    verify_loading()
