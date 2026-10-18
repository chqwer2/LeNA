#!/bin/bash
#SBATCH --job-name=gsm_diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_diag_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r2_lora_s1/lora" "runs/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b")
python -u ttt_diag.py --adapter "${AD[$SLURM_ARRAY_TASK_ID]}" --base_model meta-llama/Llama-2-7b-hf --n_y 40 --K 10
echo DIAG_DONE
