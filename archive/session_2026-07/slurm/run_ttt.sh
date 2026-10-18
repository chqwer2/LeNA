#!/bin/bash
#SBATCH --job-name=gsm_ttt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/scratch2/users/hchen/logs/gsm_ttt_%A_%a.log
# TTT probe: per-sample test-time adaptation on the question. static(K=0) vs TTT(K).
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
AD=("runs/gsmB_r2_lora_s1/lora" "runs/gsmB_r2_aurora_s1/lena_aurora_dim_none_after_b")
A=${AD[$SLURM_ARRAY_TASK_ID]}
echo "== TTT on $A =="
python -u ttt_eval.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf \
  --limit 50 --ttt_steps 10 --ttt_lr 1e-3
echo "TTT_DONE $A"
