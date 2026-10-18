#!/bin/bash
#SBATCH --job-name=b4full
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/b4full_%j.log
# ONE run of the strengthened branch (learnable inner_lr + per-channel gate + enriched core). attn r64, gsm8k eval.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u ttt_branch4.py --mode train_eval --targets attn --r 64 \
  --epochs 3 --n_train 7000 --bs 2 --lr 3e-4 --seed 1 \
  --eval_limit 150 --eval_tasks gsm8k --out runs/b4_attn_s1
echo "B4FULL_DONE exit=$?"
