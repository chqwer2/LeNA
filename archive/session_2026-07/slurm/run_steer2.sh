#!/bin/bash
#SBATCH --job-name=gsm_st2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_st2_%A_%a.log
# Steer design probes: capacity trend (bigger k) + extreme rank (r=1).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#     r  p   k  seed
CFG=("2 32 8 1" "1 16 4 1" "1 16 4 2")
read R P K SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
A=$((2*R))
export LENA_STEER=1 LENA_STEER_P=$P LENA_STEER_K=$K
TAG=gsmS_steer_r${R}_p${P}k${K}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[steer2] r=$R p=$P k=$K seed=$SEED"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations identity --lena_flex_mode dim --lena_gate_type none \
  --lora_r $R --lora_alpha $A --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python - <<PYEOF
from safetensors.torch import load_file
import glob
f=glob.glob("$AD/adapter_model.safetensors")[0]; sd=load_file(f)
sk=[k for k in sd if "steer" in k]; ws=[float(sd[k].abs().sum()) for k in sk if k.endswith("ws")]
print(f"[CHECK] steer_keys={len(sk)} ws_abs_sum={ws}")
PYEOF
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMS2_EXIT=$? tag=$TAG"
