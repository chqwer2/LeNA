#!/bin/bash
#SBATCH --job-name=lbfull
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=1-2%2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/lbfull_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
S=$SLURM_ARRAY_TASK_ID
python -u ttt_lb.py --mode train_eval --lora --lora_r 8 --r 64 --epochs 3 --n_train 7000 --bs 2 --lr 3e-4 \
  --seed $S --eval_limit 60 --out runs/lb_r64lora8_s${S}
echo "LBFULL_DONE s$S exit=$?"
