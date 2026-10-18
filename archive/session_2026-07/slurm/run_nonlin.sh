#!/bin/bash
#SBATCH --job-name=nonlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/nonlin_%j.log
# Mechanism probe: nonlinearity-usage vs rank on trained aurora adapters (r=2,4,8) + lora control.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
R=runs
declare -a AD=(
  "$R/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b"
  "$R/gsmB_r4_aurora_s1/lena_aurora_dim_none_after_b"
  "$R/gsmG_aurora_none_s1/lena_aurora_dim_none_after_b"
  "$R/gsmB_r2_lora_s1/lora"
  "$R/gsmB_r4_lora_s1/lora"
)
for a in "${AD[@]}"; do
  echo "================ $a ================"
  python -u measure_nonlin.py --adapter "$a" --base_model meta-llama/Llama-2-7b-hf --n_q 24 || echo "FAILED $a"
done
echo "NONLIN_DONE"
