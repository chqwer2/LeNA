#!/bin/bash
#SBATCH --job-name=csmoke
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/csmoke_%j.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u ttt_c.py --mode smoke --lora --lora_r 8 --r 64 --l1 1e-4 --out runs/c_smoke
echo "CSMOKE_EXIT=$?"
