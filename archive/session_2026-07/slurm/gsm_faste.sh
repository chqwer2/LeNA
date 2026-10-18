#!/bin/bash
#SBATCH --job-name=gsm_fe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_fe_%A_%a.log
# Lighter GSM8K eval on already-trained LeNA adapters (train was fine; the 300x256
# eval was too heavy on slow GPUs). 150 examples, 160 new tokens -- enough for the mean.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
DIRS=(gsmL_r8_s1 gsmL_r8_s2 gsmL_r8_s3 gsmL_r16_s1)
D=${DIRS[$SLURM_ARRAY_TASK_ID]}
AD=$(ls -d /mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$D/*/ 2>/dev/null|head -1)
[ -z "$AD" ]||[ ! -f "$AD/adapter_model.safetensors" ] && { echo "no adapter $D"; exit 0; }
echo "[FE] $D"
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k \
  --adapter "$AD" --out "$AD/gsm_fast.json"
echo "FE_EXIT=$? tag=$D"
