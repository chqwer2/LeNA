#!/bin/bash
#SBATCH --job-name=b4smoke
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/b4smoke_%j.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u ttt_branch4.py --mode smoke --targets attn --r 64 --out runs/b4_smoke
echo "B4SMOKE_EXIT=$?"
