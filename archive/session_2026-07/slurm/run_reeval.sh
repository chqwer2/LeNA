#!/bin/bash
#SBATCH --job-name=reeval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-23
#SBATCH --output=/mnt/scratch2/users/hchen/logs/reeval_%A_%a.log
# High-power re-eval (500 samples, noise ~±0.02) of all r1-8 lora/aurora adapters
# to resolve the regime map with tight error bars (150-sample was underpowered).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
LA=lena_aurora_dim_none_after_b
CFG=(
 "runs/gsmB_r1_lora_s1/lora"   "runs/gsmB_r1_lora_s2/lora"
 "runs/gsmB_r1_aurora_s1/$LA"  "runs/gsmB_r1_aurora_s2/$LA"
 "runs/gsmB_r2_lora_s1/lora"   "runs/gsmB_r2_lora_s2/lora"  "runs/gsmB_r2_lora_s3/lora" "runs/gsmB_r2_lora_s4/lora" "runs/gsmB_r2_lora_s5/lora"
 "runs/gsmB_r2_aurora_s1/$LA"  "runs/gsmB_r2_aurora_s2/$LA" "runs/gsmB_r2_aurora_s3/$LA" "runs/gsmB_r2_aurora_s4/$LA" "runs/gsmB_r2_aurora_s5/$LA"
 "runs/gsmB_r4_lora_s1/lora"   "runs/gsmB_r4_lora_s2/lora"
 "runs/gsmB_r4_aurora_s1/$LA"  "runs/gsmB_r4_aurora_s2/$LA"
 "runs/gsm_01_lora_r8_s1/lora" "runs/gsm_07_lora_r8_s2/lora" "runs/gsm_08_lora_r8_s3/lora"
 "runs/gsmG_aurora_none_s1/$LA" "runs/gsmG_aurora_none_s2/$LA" "runs/gsmG_aurora_none_s3/$LA"
)
A=${CFG[$SLURM_ARRAY_TASK_ID]}
echo "REEVAL $A"
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 500 --tasks gsm8k \
  --adapter "$A" --out "$A/gsm_500.json" || echo "FAILED $A"
echo "REEVAL_DONE $A"
