#!/bin/bash
#SBATCH --job-name=st_reev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/st_reev_%j.log
# Re-eval saved Steerable-LoRA adapters with the device fix (train already succeeded).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
for OUT in runs/gsmSF_*; do
  AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
  [ -z "$AD" ] && continue
  [ -f "$AD/adapter_model.safetensors" ] || continue
  [ -f "$AD/gsm_fast.json" ] && continue   # skip already-eval'd
  echo "REEVAL $AD"
  python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k \
    --adapter "$AD" --out "$AD/gsm_fast.json" && echo "OK $(basename $OUT)" || echo "FAILED $AD"
done
echo "STEER_REEVAL_DONE"
