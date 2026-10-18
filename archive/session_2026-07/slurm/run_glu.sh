#!/bin/bash
#SBATCH --job-name=gsm_glu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_glu_%A_%a.log
# GLU multiplicative bottleneck adapter. Compare vs aurora_r2=0.257, lora_r2=0.194.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
CFG=("2 1" "1 1")   # r seed
read R SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
A=$((2*R))
TAG=gsmGLU_r${R}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[glu] r=$R seed=$SEED"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations glu --lena_flex_mode dim --lena_gate_type none \
  --lora_r $R --lora_alpha $A --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMGLU_EXIT=$? tag=$TAG"
