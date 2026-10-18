#!/bin/bash
#SBATCH --job-name=gsm_af
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_af_%A_%a.log
# auroraf = AuroRA + provable LoRA fallback via per-dim interp gate (starts AT AuroRA).
# Does it match/beat aurora_none=0.231 while gaining a provable fallback? +input gate variant.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#   id  act      gate    seed
CFG=("0 auroraf none 1" "1 auroraf none 2" "2 auroraf none 3" "3 auroraf input 1" "4 auroraf input 2" "5 auroraf input 3")
read ID ACT GATE SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
GARG="--lena_gate_type none"; [ "$GATE" = "input" ] && GARG="--lena_gate_type sigmoid --gate_strength soft --lena_gate_mode input --lena_gate_init 0.0"
TAG=gsmF_${ACT}_${GATE}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmF] $ACT gate=$GATE seed=$SEED"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations $ACT --lena_flex_mode dim $GARG \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMF_EXIT=$? tag=$TAG"
