#!/bin/bash
#SBATCH --job-name=cfull
#SBATCH --partition=k2-epsrc-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/cfull_%j.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u ttt_c.py --mode train_eval --lora --lora_r 8 --r 64 --l1 1e-4 \
  --epochs 3 --n_train 7000 --bs 2 --lr 3e-4 --seed 1 \
  --eval_limit 150 --eval_tasks gsm8k,gsm_symbolic --out runs/c_lora8_gated_s1
echo "CFULL_DONE exit=$?"
