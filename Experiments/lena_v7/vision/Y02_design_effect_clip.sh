source /mnt/scratch2/users/hchen/software/miniforge3/etc/profile.d/conda.sh; conda activate judge; export HF_HOME=/mnt/scratch2/users/hchen/hf_home HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1; set -eo pipefail; cd /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision
nvidia-smi -L
# fig_design_effect (M1): CLIP LeNA-S inference-only token dump. Smoke (4 imgs) then 64/set. Hard cap 25 min.
END=$(( $(date +%s) + 1500 ))
timeout 420 python diag_design_effect_clip.py --out runs/diag_design_effect_clip_smoke --n 4 --bs 4
timeout $(( END - $(date +%s) )) python diag_design_effect_clip.py --out runs/diag_design_effect_clip --n 64 --bs 16
