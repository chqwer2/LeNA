#!/bin/bash
#SBATCH --job-name=gsm_ttt2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_ttt2_%A_%a.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#      lr     K
CFG=("1e-5 5" "1e-5 20" "3e-6 10")
read LR K <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "== gentle TTT lora_r2 lr=$LR K=$K =="
python -u ttt_eval.py --adapter runs/gsmB_r2_lora_s1/lora --base_model meta-llama/Llama-2-7b-hf --limit 50 --ttt_steps $K --ttt_lr $LR
echo "TTT2_DONE lr=$LR K=$K"
