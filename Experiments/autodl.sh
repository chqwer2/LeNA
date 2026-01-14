source /etc/network_turbo  # 加速 VPN

unset http_proxy && unset https_proxy # 取消 VPN



conda install -c conda-forge gh  -y
gh auth login    # Login in


cd   /root/autodl-tmp
git  clone https://github.com/chqwer2/FLoRA




export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1



source /etc/network_turbo  # 加速 VPN
cd   /root/autodl-tmp
cd FLoRA/Experiments


#tokenizer = AutoTokenizer.from_pretrained(
#    model_path,
#    local_files_only=True,
#)


#-------- Download Model -----------
export HF_HOME=$HOME/autodl-tmp/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
pip install huggingface_hub

source /etc/network_turbo  # 加速 VPN

python - << 'EOF'

from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Meta-Llama-3-8B", resume_download=True)
print("download complete")
EOF



# -------- Debug Save

python your_script.py 2>&1 | tee output.log



# --------------- Auto Close
/usr/bin/shutdown

python train.py; /usr/bin/shutdown      # 用;拼接意味着前边的指令不管执行成功与否，都会执行shutdown命令
python train.py && /usr/bin/shutdown    # 用&&拼接表示前边的命令执行成功后才会执行shutdown。请根据自己的需要选择



