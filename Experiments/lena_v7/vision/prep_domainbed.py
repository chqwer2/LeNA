"""Convert the HF mirrors of DomainBed PACS (flwrlabs/pacs) and Office-Home (flwrlabs/office-home) parquet files into
one pickle per domain: (list of encoded image bytes, list of int labels, list of source paths), plus meta.json with
domain sizes and class names. Run once on the login node with an env that has pyarrow (conda env `lena`).

  python -I prep_domainbed.py /mnt/scratch2/users/hchen/data/lena_v7/domainbed
"""
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

root = sys.argv[1]
for src, dst in (("pacs", "PACS"), ("officehome", "OfficeHome")):
    files = sorted(glob.glob(os.path.join(root, "raw", src, "data", "*.parquet")))
    per = defaultdict(lambda: ([], [], []))
    names = None
    for f in files:
        pf = pq.ParquetFile(f)
        if names is None:
            hf = json.loads(pf.schema_arrow.metadata[b"huggingface"])
            names = hf["info"]["features"]["label"].get("names")
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pydict()
            for im, d, y in zip(t["image"], t["domain"], t["label"]):
                b = per[d]
                b[0].append(im["bytes"]); b[1].append(int(y)); b[2].append(im["path"])
    os.makedirs(os.path.join(root, dst), exist_ok=True)
    meta = dict(classes=names, domains={})
    for d, (ims, ys, ps) in sorted(per.items()):
        pickle.dump((ims, ys, ps), open(os.path.join(root, dst, f"{d}.pkl"), "wb"), protocol=4)
        meta["domains"][d] = dict(n=len(ys), n_classes=len(set(ys)), min_per_class=min(Counter(ys).values()))
    meta["n_total"] = sum(v["n"] for v in meta["domains"].values())
    json.dump(meta, open(os.path.join(root, dst, "meta.json"), "w"), indent=1)
    print(dst, json.dumps({k: v for k, v in meta.items() if k != "classes"}), "classes", len(names or []), flush=True)
