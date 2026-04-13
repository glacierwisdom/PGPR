
import torch
import sys
import bitsandbytes as bnb

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

try:
    print(f"BitsAndBytes version: {bnb.__version__}")
except:
    print("BitsAndBytes not found/error")
