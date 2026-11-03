source /mnt/scratch2/users/hchen/software/miniforge3/etc/profile.d/conda.sh; conda activate judge; export HF_HOME=/mnt/scratch2/users/hchen/hf_home HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1; set -eo pipefail; cd /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision
nvidia-smi -L
# fig_design_live: inference only (LoRA 1e-4, LeNA cam_lena_lr5e-5, LeNA-S K05). Smoke (4 patches/hospital) then 800/hospital. Hard cap 29 min total.
END=$(( $(date +%s) + 1740 ))
timeout 420 python diag_design_live.py --out runs/diag_design_live_smoke --n 4 --per_cat 1 --bs 4
timeout $(( END - $(date +%s) )) python diag_design_live.py --out runs/diag_design_live --n 800 --per_cat 6 --bs 100
