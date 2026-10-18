#!/bin/bash
#SBATCH --job-name=gsm_ag2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_ag2_%j.log
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
TAG=gsmAG2_r2_s1; OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations auroglu --lena_flex_mode dim --lena_gate_type none \
  --lora_r 2 --lora_alpha 4 --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto --learning_rate 3e-4 --seed 1 --max_train_samples 7000 --output_dir "$OUT"
[ $? -ne 0 ] && { echo TRAIN_FAIL; exit 1; }
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "AG2_DONE tag=$TAG"
