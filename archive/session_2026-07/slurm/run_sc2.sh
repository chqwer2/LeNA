#!/bin/bash
#SBATCH --job-name=gsm_sc2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_sc2_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#      K   N
CFG=("15 8" "15 16" "30 8")
read K N <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "== SC-TTT push aurora_r2 K=$K N=$N limit=80 =="
python -u ttt_sc.py --adapter runs/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b --base_model meta-llama/Llama-2-7b-hf --limit 80 --n $N --ttt_steps $K --ttt_lr 2e-5 --temp 0.8
echo "SC2_DONE K=$K N=$N"
