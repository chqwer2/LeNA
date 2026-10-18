#!/bin/bash
#SBATCH --job-name=ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/ablate_%j.log
# Linearize-ablation: replace phi with best-linear M*, re-eval. drop=value of nonlinearity per rank.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
R=runs
# orig acc (this seed): r2_s1=0.22  r4_s1=0.16  r8=0.2333
declare -a AD=(
  "$R/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b"
  "$R/gsmB_r4_aurora_s1/lena_aurora_dim_none_after_b"
  "$R/gsmG_aurora_none_s1/lena_aurora_dim_none_after_b"
)
for a in "${AD[@]}"; do
  echo "================ $a ================"
  python -u ablate_linearize.py --adapter "$a" --base_model meta-llama/Llama-2-7b-hf --n_fit 24 --limit 150 || echo "FAILED $a"
done
echo "ABLATE_DONE"
