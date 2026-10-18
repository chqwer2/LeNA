#!/bin/bash
#SBATCH --job-name=ablate2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/scratch2/users/hchen/logs/ablate2_%A_%a.log
# Multi-seed ablation confirmation: is the SIGN of linearize-effect stable?
# s1 done: r2->0.307(up) r4->0.233(up) r8->0.173(down). Now more seeds per rank.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
L=lena_aurora_dim_none_after_b
CFG=(
 "runs/gsmB_r2_aurora_s2/$L"
 "runs/gsmB_r2_aurora_s3/$L"
 "runs/gsmB_r2_aurora_s4/$L"
 "runs/gsmB_r4_aurora_s2/$L"
 "runs/gsmG_aurora_none_s2/$L"
 "runs/gsmG_aurora_none_s3/$L"
)
A=${CFG[$SLURM_ARRAY_TASK_ID]}
echo "================ $A ================"
python -u ablate_linearize.py --adapter "$A" --base_model meta-llama/Llama-2-7b-hf --n_fit 24 --limit 150 || echo "FAILED $A"
echo "ABLATE2_DONE"
