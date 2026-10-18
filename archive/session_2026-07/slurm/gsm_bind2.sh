#!/bin/bash
#SBATCH --job-name=gsm_bnd2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_bnd2_%A_%a.log
# Reinforce the r=2 BINDING signal (aurora~0.27 vs lora~0.167, gap +0.10 vs r8's +0.028)
# + probe extreme r=1. More seeds to nail the mean under high 150-sample variance.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#    r  act      seed   (r1: s1,s2 ; r2: s3,s4,s5)
CFG=("1 lora 1" "1 lora 2" "1 aurora 1" "1 aurora 2" "1 auroraf 1" "1 auroraf 2" \
     "2 lora 3" "2 lora 4" "2 lora 5" "2 aurora 3" "2 aurora 4" "2 aurora 5" \
     "2 auroraf 3" "2 auroraf 4" "2 auroraf 5")
read R ACT SEED <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
A=$((2*R))
if [ "$ACT" = "lora" ]; then
  METH="--methods lora"; ACTARG=""
else
  METH="--methods lena"; ACTARG="--lena_activations $ACT --lena_flex_mode dim --lena_gate_type none"
fi
TAG=gsmB_r${R}_${ACT}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmB2] r=$R act=$ACT seed=$SEED alpha=$A"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" $METH $ACTARG \
  --lora_r $R --lora_alpha $A --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed $SEED --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMB2_EXIT=$? tag=$TAG"
