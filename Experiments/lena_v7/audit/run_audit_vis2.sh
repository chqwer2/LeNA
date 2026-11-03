source /mnt/scratch2/users/hchen/software/miniforge3/etc/profile.d/conda.sh; conda activate judge; export HF_HOME=/mnt/scratch2/users/hchen/hf_home HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
A=/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/audit; cd /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision
echo "VSTART $(date)"
timeout 1800 python $A/audit_dtype.py --task cam --n 2000 --ckpt runs/cam_lena_lr5e-5/best.pt --out $A/dtype_cam_v2.json; echo "VCAM_RC $?"
timeout 1800 python $A/audit_dtype.py --task clip --n 1000 --ckpt runs/clip_lena_s1/adapter.pt --out $A/dtype_clip_v2.json; echo "VCLIP_RC $?"
echo "VIS_DONE $(date)"
