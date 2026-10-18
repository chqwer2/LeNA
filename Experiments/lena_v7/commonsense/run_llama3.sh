#!/usr/bin/env bash
# LLaMA3-8B commonsense runs for the PR paper (T1, T4, T7). Usage: bash run_llama3.sh <run-id> [gpu]
# Data (from LLM-Adapters): commonsense_170k.json in this folder, dataset/<name>/test.json for the 8 tasks:
#   git clone https://github.com/AGI-Edgerunners/LLM-Adapters && cp LLM-Adapters/ft-training_set/commonsense_170k.json . \
#   && cp -r LLM-Adapters/dataset .
# Rank 32, alpha 64, lr 1e-4, 3 epochs, as in DoRA's LLaMA3-8B script. LoRA-dagger rank 50 matches LeNA's count.
set -euo pipefail
ID=$1; GPU=${2:-0}
BASE=meta-llama/Meta-Llama-3-8B
COMMON="--base_model $BASE --data_path commonsense_170k.json --lora_r 32 --lora_alpha 64 --learning_rate 1e-4 \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 --cutoff_len 256 --val_set_size 120 --eval_step 80 --save_step 80"
case $ID in
  lena)        FLAGS="" ;;                                   # T1 / T4 default (learned omega)
  lena_d)      FLAGS="--dora" ;;                             # T1 LeNA-D
  lora)        FLAGS="--adapter lora" ;;                     # T4 LoRA(r), T7
  lora_dagger) FLAGS="--adapter lora"; COMMON="${COMMON/--lora_r 32 --lora_alpha 64/--lora_r 50 --lora_alpha 100}" ;;
  dora)        FLAGS="--adapter lora --dora" ;;              # T7
  fixed)       FLAGS="--mode fixed" ;;                       # T4 curve group
  affine)      FLAGS="--mode plain --knots 2" ;;
  predictor)   FLAGS="--mode predictor" ;;
  plain)       FLAGS="--mode plain" ;;
  js)          FLAGS="--js_only" ;;                          # T4 step group (omega = 0)
  frozen_a2)   FLAGS="--freeze_A2" ;;                        # T4 reads and prior
  id_prior)    FLAGS="--prior identity" ;;
  u_prior)     FLAGS="--prior u_pred" ;;
  *) echo "unknown run $ID"; exit 1 ;;
esac
OUT=runs/llama3_${ID}
CUDA_VISIBLE_DEVICES=$GPU python finetune_lena.py $COMMON $FLAGS --output_dir $OUT 2>&1 | tee $OUT.train.log
CUDA_VISIBLE_DEVICES=$GPU python evaluate_lena.py --base_model $BASE --run_dir $OUT --batch_size 1 2>&1 | tee $OUT.eval.log
