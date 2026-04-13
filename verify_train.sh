#!/bin/bash
#SBATCH --job-name=gapnppi_verify
#SBATCH --partition=cto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=ctolab07
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_train_%j.log
#SBATCH --error=logs/verify_train_%j.err

# Load environment
source /home/gaoziqi/anaconda3/bin/activate llama3

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run verification training
python main.py train \
    --config configs/verify_train.yaml \
    --model-config configs/model.yaml \
    --data-config configs/data.yaml \
    --verbose
