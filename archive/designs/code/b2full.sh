#!/bin/bash
#SBATCH --job-name=b2full
#SBATCH --partition=k2-epsrc-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/mnt/scratch2/users/hchen/logs/b2full_%A_%a.log
# Improve the BRANCH: does adding an MLP-parallel branch (full coverage like LoRA) beat attn-only + beat LoRA?
# A=attn r64 (~50M), B=attn+mlp r64 (~101M), C=attn+mlp r32 (~50M, param-matched to A). 2 seeds. Fair first-#### eval.
set -uo pipefail
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/scratch2/users/hchen/FLoRA/Experiments

CFGS=(A A B B C C)
SEEDS=(1 2 1 2 1 2)
I=$SLURM_ARRAY_TASK_ID
C=${CFGS[$I]}; SEED=${SEEDS[$I]}
case $C in
  A) TARG="attn";     R=64;;
  B) TARG="attn,mlp"; R=64;;
  C) TARG="attn,mlp"; R=32;;
esac
TAG=b2_${C}_s${SEED}
OUT=/mnt/scratch2/users/hchen/FLoRA/Experiments/runs/$TAG
echo "[B2FULL] I=$I CFG=$C TARGETS=$TARG R=$R SEED=$SEED OUT=$OUT"
python -u ttt_branch2.py --mode train_eval --targets "$TARG" --r $R \
  --epochs 3 --n_train 7000 --bs 2 --lr 3e-4 --seed $SEED \
  --eval_limit 200 --eval_tasks gsm8k,gsm_symbolic --out "$OUT"
echo "B2FULL_DONE $TAG exit=$?"
