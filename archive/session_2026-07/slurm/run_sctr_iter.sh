#!/bin/bash
#SBATCH --job-name=gsm_scit
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_scit_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r2_lora_s1/lora" "runs/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b")
A=${AD[$SLURM_ARRAY_TASK_ID]}
echo "== iterative SC-aware $A rounds=3 =="
python -u sc_train_iter.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf \
  --n_train 250 --n_sample 6 --epochs 1 --rounds 3 --lr 5e-5 --temp 0.9 --eval_limit 80 --eval_n 8
echo "SCIT_DONE $A"
