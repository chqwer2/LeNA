#!/usr/bin/env bash
# Usage: bash run_langood.sh <limit:0=all> <tag> <batch_size>   (runs inside the a100 allocation)
set -euo pipefail
LIMIT=$1; TAG=$2; BS=${3:-8}
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/commonsense
BASE=meta-llama/Meta-Llama-3-8B
for M in lena lora; do
  echo "=== $M gen $(date)"
  python evaluate_lena.py --base_model $BASE --run_dir runs/llama3_$M --datasets commonsenseqa,mmlu \
     --batch_size $BS --min_new_tokens 1 --limit $LIMIT --out_name langood_gen_$TAG
  echo "=== $M score $(date)"
  python score_langood.py --base_model $BASE --run_dir runs/llama3_$M --datasets commonsenseqa,mmlu \
     --limit $LIMIT --out_name langood_score_$TAG
done
echo "=== ALL DONE $(date)"
