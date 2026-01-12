source /etc/network_turbo  # 加速 VPN

unset http_proxy && unset https_proxy # 取消 VPN



conda install -c conda-forge gh  -y
gh auth login    # Login in


cd   /root/autodl-tmp
git  clone https://github.com/chqwer2/FLoRA

cd   /root/autodl-tmp
cd FLoRA/Experiments




#-------- Download Model -----------
export HF_HOME=$HOME/autodl-tmp/hf_home
export HF_HUB_CACHE=$HF_HOME/hub

source /etc/network_turbo  # 加速 VPN

python - << 'EOF'

from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-2-7b-hf", resume_download=True)
print("download complete")
EOF





