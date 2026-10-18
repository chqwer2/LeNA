#!/bin/bash
#SBATCH --job-name=gsm_sc3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_sc3_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r2_lora_s1/lora" "runs/gsm_01_lora_r8_s1/lora" "runs/gsmG_aurora_none_s1/lena_aurora_dim_none_after_b")
A=${AD[$SLURM_ARRAY_TASK_ID]}
echo "== SC-TTT sweep $A K=15 N=8 limit=80 =="
python -u ttt_sc.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf --limit 80 --n 8 --ttt_steps 15 --ttt_lr 2e-5 --temp 0.8
echo "SC3_DONE $A"
