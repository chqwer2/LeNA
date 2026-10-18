#!/bin/bash
#SBATCH --job-name=st_vfy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/st_vfy_%j.log
# FAST verify: does steer now train (ws != 0) after the optimizer/trainable fix?
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
export LENA_STEER=1 LENA_STEER_P=16 LENA_STEER_K=4
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/gsmS_verify
echo "[verify] short-train steer r2 p16k4"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations identity --lena_flex_mode dim --lena_gate_type none \
  --lora_r 2 --lora_alpha 4 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 1 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed 1 --max_train_samples 600 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python - <<PYEOF
from safetensors.torch import load_file
import glob
f=glob.glob("$AD/adapter_model.safetensors")[0]; sd=load_file(f)
sk=[k for k in sd if "steer" in k]
ws=sum(float(sd[k].abs().sum()) for k in sk if k.endswith("ws"))
U =sum(float(sd[k].abs().sum()) for k in sk if k.endswith(".U"))
print(f"[VERIFY] steer_keys={len(sk)} ws_total={ws:.4f} U_total={U:.2f}  (ws>0 => STEER TRAINS)")
PYEOF
echo "VERIFY_DONE"
