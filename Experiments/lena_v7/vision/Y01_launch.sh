Q=/mnt/scratch2/users/hchen/queue/mig3gd/side; n=Y01_design_live_dump
[ -n "$(ls $Q/running/*.sh 2>/dev/null)" ] && { echo "BUSY $(ls $Q/running)"; exit 1; }
cp /mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision/$n.sh $Q/running/$n.sh
nohup bash -c "echo \"\$(date +%F_%T) SIDE START $n on 10347348\" >> $Q/side.log; srun --jobid=10347348 --overlap bash $Q/running/$n.sh > $Q/running/$n.log 2>&1; rc=\$?; echo \$rc > $Q/running/$n.rc; d=done; [ \$rc -ne 0 ] && d=failed; mv $Q/running/$n.sh $Q/running/$n.log $Q/running/$n.rc $Q/\$d/; echo \"\$(date +%F_%T) SIDE END $n rc=\$rc -> \$d\" >> $Q/side.log" > /dev/null 2>&1 &
echo launched
