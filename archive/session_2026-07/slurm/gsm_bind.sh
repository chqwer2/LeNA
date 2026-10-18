#!/bin/bash
#SBATCH --job-name=gsm_bind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_bind_%A_%a.log
# Core thesis test: does nonlinearity(+provable fallback) separate from LoRA when RANK BINDS?
# r=8 doesn't bind (all ~0.23). Sweep r=2,4 x {lora, aurora, auroraf} x 2 seeds.
# WIN = auroraf > aurora > lora with the gap GROWING as r shrinks.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#    r  act      seed
CFG=("2 lora 1" "2 lora 2" "2 aurora 1" "2 aurora 2" "2 auroraf 1" "2 auroraf 2" \
     "4 lora 1" "4 lora 2" "4 aurora 1" "4 aurora 2" "4 auroraf 1" "4 auroraf 2")
read R ACT SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
A=$((2*R))
if [ "$ACT" = "lora" ]; then
  METH="--methods lora"; ACTARG=""
else
  METH="--methods lena"; ACTARG="--lena_activations $ACT --lena_flex_mode dim --lena_gate_type none"
fi
TAG=gsmB_r${R}_${ACT}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmB] r=$R act=$ACT seed=$SEED alpha=$A"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" $METH $ACTARG \
  --lora_r $R --lora_alpha $A --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMB_EXIT=$? tag=$TAG"
