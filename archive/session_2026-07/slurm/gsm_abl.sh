#!/bin/bash
#SBATCH --job-name=gsm_abl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_abl_%A_%a.log
# Does AuroRA's compressed bottleneck explain its win? Add it to LeNA. Explicit flags.
# (LoRA r8=0.203, LeNA r8=0.224, AuroRA=0.287, LoRA r16=0.290)
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
#   id  act        compress  gate
CFG=("0 rankmixc 4 input" "1 rankmixc 4 none" "2 rankmixc 2 input" "3 rankmixc 8 input")
read ID ACT COMP GATE <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
GARG="--lena_gate_type none"; [ "$GATE" = "input" ] && GARG="--lena_gate_type sigmoid --gate_strength soft --lena_gate_mode input --lena_gate_init 0.0"
TAG=gsmA_${ACT}_c${COMP}_${GATE}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[gsmA] $ACT compress=$COMP gate=$GATE"
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "openai/gsm8k:main" --methods lena \
  --lena_activations $ACT --lena_flex_mode dim $GARG \
  --lena_activation_kwargs_json "{\"compress\":$COMP}" \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 2 --cutoff_len 512 --device auto \
  --learning_rate 3e-4 --seed 1 --max_train_samples 7000 --output_dir "$OUT"
RC=$?; echo "TRAIN_EXIT=$RC"; [ $RC -ne 0 ] && exit 1
AD=$(ls -d $OUT/*/ 2>/dev/null|head -1)
python -u eval_generate.py --base_model meta-llama/Llama-2-7b-hf --limit 150 --tasks gsm8k --adapter "$AD" --out "$AD/gsm_fast.json"
echo "GSMA_EXIT=$? tag=$TAG"
