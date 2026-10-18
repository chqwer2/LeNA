#!/bin/bash
#SBATCH --job-name=gsm_ag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_ag_%A_%a.log
# Stand on AuroRA (0.287), add gate + provable fallback. Does it beat 0.287? 3 seeds each.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#   id  act      gate    seed
CFG=("0 aurora  none  1" "1 aurora  none  2" "2 aurorag input 1" "3 aurorag input 2" "4 aurorag input 3" "5 aurorag none 1" "6 aurora none 3")
read ID ACT GATE SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
GARG="--lena_gate_type none"; [ "$GATE" = "input" ] && GARG="--lena_gate_type sigmoid --gate_strength soft --lena_gate_mode input --lena_gate_init 0.0"
TAG=gsmG_${ACT}_${GATE}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmG] $ACT gate=$GATE seed=$SEED"
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
echo "GSMG_EXIT=$? tag=$TAG"
