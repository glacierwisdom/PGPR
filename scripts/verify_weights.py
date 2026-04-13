import torch
checkpoint = torch.load('artifacts/checkpoints/final_model.pth', map_location='cpu')
print(f"Keys in checkpoint: {list(checkpoint.keys())}")
if 'model_state_dict' in checkpoint:
    print(f"Model state dict keys (first 5): {list(checkpoint['model_state_dict'].keys())[:5]}")
if 'llm_state_dict' in checkpoint:
    print(f"LLM state dict keys (first 5): {list(checkpoint['llm_state_dict'].keys())[:5]}")
else:
    print("LLM state dict NOT found in checkpoint!")
