source /etc/network_turbo  # 加速 VPN

unset http_proxy && unset https_proxy # 取消 VPN



conda install -c conda-forge gh  -y
gh auth login    # Login in

pip install  datasets



export HF_HOME=$HOME/autodl-tmp/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=1200


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





