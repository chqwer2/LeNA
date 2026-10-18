#!/bin/bash
#SBATCH --job-name=lowdata
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/mnt/scratch2/users/hchen/logs/lowdata_%A_%a.log
# Sample-efficiency test: does nonlinear AuroRA beat LoRA when TRAIN DATA is scarce? r8 fixed.
# N in {200,500,1000} x method{aurora,lora} x seed{1,2}. Fair batched eval_fast (first-####, gsm8k limit 500).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments

NS=(200 200 200 200 500 500 500 500 1000 1000 1000 1000)
METHS=(aurora aurora lora lora aurora aurora lora lora aurora aurora lora lora)
SEEDS=(1 2 1 2 1 2 1 2 1 2 1 2)
I=$SLURM_ARRAY_TASK_ID
N=${NS[$I]}; NAME=${METHS[$I]}; SEED=${SEEDS[$I]}
R=8; A=16; TARG="q_proj,k_proj,v_proj,up_proj,down_proj"
TAG=low_N${N}_${NAME}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[LOWDATA] I=$I N=$N NAME=$NAME SEED=$SEED OUT=$OUT"

if [ "$NAME" = "lora" ]; then
  python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf --dataset "openai/gsm8k:main" \
    --methods lora --lora_r $R --lora_alpha $A --lora_dropout 0.05 --lora_target_modules $TARG \
    --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto --learning_rate 3e-4 \
    --seed $SEED --max_train_samples $N --output_dir "$OUT"
else
  python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf --dataset "openai/gsm8k:main" \
    --methods lena --lena_activations aurora --lena_flex_mode dim --lena_gate_type none \
    --lora_r $R --lora_alpha $A --lora_dropout 0.05 --lora_target_modules $TARG \
    --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto --learning_rate 3e-4 \
    --seed $SEED --max_train_samples $N --output_dir "$OUT"
fi
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && { echo "LOWDATA_TRAINFAIL $TAG"; exit 1; }

AD=$(ls -d $OUT/*/ 2>/dev/null | grep -v checkpoint | head -1)
python -u eval_fast.py --base_model meta-llama/Llama-2-7b-hf --adapter "$AD" \
  --tasks gsm8k --limit 500 --bs 16 --maxtok 128 --out "$AD/gsm_fast.json"
echo "LOWDATA_DONE $TAG $(tr -d '\n ' < $AD/gsm_fast.json 2>/dev/null)"
