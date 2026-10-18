#!/bin/bash
#SBATCH --job-name=rankfill
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/mnt/scratch2/users/hchen/logs/rankfill_%A_%a.log
# Lock the efficiency win: resolve r4 anomaly (aurora s3,4,5 -> n5) + fill r8 (aurora/lora s1,2,3).
# aurora = nonlinear low-rank adapter (lena+aurora activation, flex dim, no gate). Same fair eval (eval_generate limit 500).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments

RANKS=(4 4 4 4 4 4 8 8 8 8 8 8)
METHS=(aurora aurora aurora lora lora lora aurora aurora aurora lora lora lora)
SEEDS=(3 4 5 3 4 5 1 2 3 1 2 3)
I=$SLURM_ARRAY_TASK_ID
R=${RANKS[$I]}; NAME=${METHS[$I]}; SEED=${SEEDS[$I]}
A=$((2*R))
TARG="q_proj,k_proj,v_proj,up_proj,down_proj"
TAG=gsmB_r${R}_${NAME}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[RANKFILL] I=$I R=$R NAME=$NAME SEED=$SEED OUT=$OUT"

if [ "$NAME" = "lora" ]; then
  python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf --dataset "openai/gsm8k:main" \
    --methods lora --lora_r $R --lora_alpha $A --lora_dropout 0.05 --lora_target_modules $TARG \
    --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto --learning_rate 3e-4 \
    --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
else
  python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf --dataset "openai/gsm8k:main" \
    --methods lena --lena_activations aurora --lena_flex_mode dim --lena_gate_type none \
    --lora_r $R --lora_alpha $A --lora_dropout 0.05 --lora_target_modules $TARG \
    --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto --learning_rate 3e-4 \
    --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
fi
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && { echo "RANKFILL_TRAINFAIL $TAG"; exit 1; }

AD=$(ls -d $OUT/*/ 2>/dev/null | head -1)
echo "[RANKFILL] eval AD=$AD"
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --adapter "$AD" \
  --tasks gsm8k --limit 500 --out "$AD/gsm_500.json"
echo "RANKFILL_DONE tag=$TAG em=$(grep -o '\"exact_match\"[^,}]*' $AD/gsm_500.json 2>/dev/null | head -1)"
