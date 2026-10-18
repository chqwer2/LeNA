#!/bin/bash
#SBATCH --job-name=gsm_scrk
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_scrk_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r1_lora_s1/lora" "runs/gsmB_r4_lora_s1/lora" "runs/gsm_01_lora_r8_s1/lora")
A=${AD[$SLURM_ARRAY_TASK_ID]}
echo "== SC-aware rank-sweep $A =="
python -u sc_train.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf \
  --n_train 250 --n_sample 6 --epochs 2 --lr 5e-5 --temp 0.9 --eval_limit 80 --eval_n 8
echo "SCRK_DONE $A"
