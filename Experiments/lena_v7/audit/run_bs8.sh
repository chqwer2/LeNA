source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh; export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/audit; cd /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/math
T0=$(date +%s); echo "START $(date)"
timeout 7200 python eval_math.py --run_dir $A/math_lena_run --test $A/gsm8k500_s0.json --batch_size 8 --compile --tag _gsm_bs8; echo "GSM_RC $?"
LEFT=$((9000 - ($(date +%s) - T0))); echo "LEFT $LEFT"
if [ $LEFT -gt 1800 ]; then timeout $LEFT python eval_math.py --run_dir $A/math_lena_run --test $A/math500_s0.json --batch_size 8 --compile --tag _math_bs8; echo "MATH_RC $?"; else echo "MATH_SKIPPED budget"; fi
echo "BS8_DONE $(date)"
