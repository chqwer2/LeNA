#!/bin/bash
#SBATCH --job-name=gsm_gttt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_gttt_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b" "runs/gsmB_r2_lora_s1/lora")
A=${AD[$SLURM_ARRAY_TASK_ID]}
echo "== GLOBAL-TTT $A =="
python -u ttt_global.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf --limit 160 --n 8 --epochs 3 --lr 1e-4 --temp 0.8 --holdout 0.5
echo "GTTT_DONE $A"
