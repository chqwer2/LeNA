#!/bin/bash
#SBATCH --job-name=gsm_svd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_svd_%A_%a.log
# SVD-init Steerable LoRA: U = top-p left singular vectors of base weight (well-conditioned,
# weight-aligned directions). Test if principled directions beat LoRA (random steer=0.153<lora0.194).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoid fragmentation OOM
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#     r  p   k  seed
CFG=("2 8 2 1" "2 16 4 1")
read R P K SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
A=$((2*R))
export LENA_STEER=1 LENA_STEER_SVD=1 LENA_STEER_P=$P LENA_STEER_K=$K
TAG=gsmSV_r${R}_p${P}k${K}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[steerSVD] r=$R p=$P k=$K seed=$SEED (U=top-p left singular vecs)"
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
sk=[k for k in sd if "steer" in k]; ws=sum(float(sd[k].abs().sum()) for k in sk if k.endswith("ws"))
print(f"[CHECK] steer_keys={len(sk)} ws_total={ws:.4f}")
PYEOF
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMSV_EXIT=$? tag=$TAG"
