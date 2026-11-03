"""ImageNet-1k validation (50k) -> val50k.pkl, same format as val.pkl: (image bytes list, labels, zeros).
  python I05_clip_val50k_prep.py download   # login node (network; HF token from the environment), shards only
  python I05_clip_val50k_prep.py build      # compute node, HF_HUB_OFFLINE=1, reads the cached shards"""
import os, pickle, sys
from huggingface_hub import hf_hub_download
R = "ILSVRC/imagenet-1k"; out = "/mnt/scratch2/users/hchen/data/lena_v7/imagenet_pkl/val50k.pkl"
files = [hf_hub_download(R, f"data/validation-{k:05d}-of-00014.parquet", repo_type="dataset") for k in range(14)]
print("shards", len(files), flush=True)
if sys.argv[1] == "build":
    import pyarrow.parquet as pq
    imgs, ys = [], []
    for f in files:
        t = pq.read_table(f, columns=["image", "label"])
        imgs += [d["bytes"] for d in t.column("image").to_pylist()]; ys += t.column("label").to_pylist()
    assert len(ys) == 50000, len(ys)
    pickle.dump((imgs, ys, [0] * len(ys)), open(out + ".tmp", "wb"), protocol=4); os.replace(out + ".tmp", out)
    print("val50k", len(ys), len(set(ys)), flush=True)
