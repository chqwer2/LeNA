#!/bin/bash
#SBATCH --job-name=bdiag
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/bdiag_%A_%a.log
# Root-cause diagnostic for branch underfit: compare SMOOTHED TRAIN LOSS (fitting) across ablations. attn-only, 2000 steps.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
ILR=(1.0 0 0 1.0 0 0)     # inner_lr: TTT on/off
MID=(1   1 4 1   4 1)     # SwiGLU hidden mult (capacity)
ALR=(3e-4 3e-4 3e-4 1e-3 1e-3 1e-3)  # AdamW lr
I=$SLURM_ARRAY_TASK_ID
python -u ttt_branch3.py --targets attn --r 64 --inner_lr ${ILR[$I]} --mid ${MID[$I]} --lr ${ALR[$I]} \
  --diag_steps 2000 --n_train 4000 --seed 1
echo "BDIAG_EXIT=$? cfg=$I"
