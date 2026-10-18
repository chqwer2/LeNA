#!/bin/bash
#SBATCH --job-name=gsm_comp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_comp_%A_%a.log
# Competitor benchmark on GSM8K r8: do AFA-LoRA/AuroRA/LoRAN also beat LoRA r8 (0.203)?
# Positions LeNA (0.224) against the field the reviewers named. Explicit flags.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
ACTS=(afa aurora loran); ACT=${ACTS[$SLURM_ARRAY_TASK_ID]}
TAG=gsmC_${ACT}_r8_s1
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmC] competitor=$ACT r=8"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations $ACT --lena_flex_mode dim --lena_gate_type none \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed 1 --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMC_EXIT=$? tag=$TAG"
