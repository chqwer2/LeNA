#!/bin/bash
#SBATCH --job-name=reeval
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/reeval_%j.log
# Batched fair re-eval (first-#### extraction) of ALL saved gsmB adapters -> consistent trusted rank curve, no retrain.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments

DIRS=$(ls -d runs/gsmB_*/*/ 2>/dev/null | grep -v checkpoint)
echo "[REEVAL] adapters to eval:"; echo "$DIRS" | wc -l
for d in $DIRS; do
  [ -f "${d}adapter_config.json" ] || continue
  tag=$(echo "$d" | grep -oE "gsmB_r[0-9]+_[a-z]+_s[0-9]+")
  t0=$SECONDS
  python -u eval_fast.py --adapter "$d" --tasks gsm8k --limit 500 --bs 16 --maxtok 128 --out "${d}gsm_fast.json" 2>&1 | grep -E "\[FAST\]|\[eval-fast\]|Error|CUDA" | tail -3
  echo "REEVAL_ONE $tag dt=$((SECONDS-t0))s $(tr -d '\n ' < ${d}gsm_fast.json 2>/dev/null)"
done
echo REEVAL_ALL_DONE
