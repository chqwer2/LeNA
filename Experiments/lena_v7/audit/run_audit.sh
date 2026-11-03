source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh; export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/audit; V=/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision
cd $V
echo "START $(date)"
timeout 1800 python $A/audit_dtype.py --task cam --n 2000 --ckpt runs/cam_lena_lr5e-5/best.pt --out $A/dtype_cam.json; echo "CAM_RC $?"
timeout 1800 python $A/audit_dtype.py --task clip --n 1000 --ckpt runs/clip_lena_s1/adapter.pt --out $A/dtype_clip.json; echo "CLIP_RC $?"
mkdir -p $A/math_lena_run; cp ../math/runs/math_lena/lena_config.json $A/math_lena_run/; ln -sf /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/math/runs/math_lena/adapter.pt $A/math_lena_run/adapter.pt
cd ../math
timeout 3600 python eval_math.py --run_dir $A/math_lena_run --test $A/gsm100.json --batch_size 16 --compile --tag _bs16; echo "BS16_RC $?"
timeout 4800 python eval_math.py --run_dir $A/math_lena_run --test $A/gsm100.json --batch_size 1 --compile --tag _bs1; echo "BS1_RC $?"
echo "AUDIT_DONE $(date)"
