#!/bin/bash
#SBATCH --job-name=b2smoke
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/b2smoke_%j.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u ttt_branch2.py --mode smoke --targets attn,mlp --r 64 --out runs/b2_smoke
echo "B2SMOKE_EXIT=$?"
