#!/bin/bash
#SBATCH --job-name=gsm_ctx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_ctx_%A_%a.log
# The novel axis: context (cross-token) gate vs input (per-token) gate on aurorag.
# Baselines: aurora 0.287, LeNA 0.224, LoRA r16 0.290.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#   id  gatemode seed
CFG=("0 context 1" "1 context 2" "2 context 3" "3 input 1" "4 input 2" "5 input 3")
read ID GM SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
TAG=gsmX_aurorag_${GM}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmX] aurorag gate=$GM seed=$SEED"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations aurorag --lena_flex_mode dim \
  --lena_gate_type sigmoid --gate_strength soft --lena_gate_mode $GM --lena_gate_init 0.0 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMX_EXIT=$? tag=$TAG"
