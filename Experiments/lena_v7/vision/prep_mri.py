"""Glioma MRI -> per-visit axial slice stacks for mri_sam_seg.py. Reads two layouts and maps both to one geometry
and one intensity normalisation, so that a test-cohort shift is scanner/site, not preprocessing:

  brats  BraTS 2023/2024 release: <cohort>/<case>/<case>-{t1c,t2w,t2f,seg}.nii.gz (240x240x155, labels 1/2/3)
  long   harmonized longitudinal: <cohort>/<patient>/<visit>/{t1ce,t2,flair,seg_gt}.nii.gz (192x192x160, [0,1]);
         only seg_gt with seg_gt.json aligned=true. Never seg.nii.gz (our own model prediction merged in).

Common processing: reorient to RAS (nib.as_closest_canonical), brain = nonzero in any of the three sequences,
z-score per sequence inside the brain, clip [-5, 5], background -3 (or --norm minmax: 0.5/99.5
percentiles inside the brain -> [0, 1], background 0), in-plane crop/pad to 192x192 centred on the
brain bounding box, all axial slices kept.

  img.npy  float16 (Z, 3, 192, 192)   channels T1c, T2, FLAIR
  seg.npy  uint8   (Z, 192, 192)      0 bg, 1 NCR, 2 ED, 3 ET (label 4 = resection cavity -> 0)
  index.json per cohort

Run in an env with nibabel (e.g. envs/progression):
  python prep_mri.py --layout brats --src /mnt/scratch2/users/hchen/brats/data --cohorts BraTS-2023-GLI --limit 40
"""
import argparse
import json
import os
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np

OUT = "/mnt/scratch2/users/hchen/data/lena_v7/mri"
SRC = {"brats": "/mnt/scratch2/users/hchen/brats/data", "long": "/mnt/scratch2/users/hchen/data/longitudinal"}
FILES = {"brats": ("t1c", "t2w", "t2f", "seg"), "long": ("t1ce", "t2", "flair", "seg_gt")}
SIZE = 192
# Source labels -> 1 NCR-like, 2 ED, 3 ET, so that WT = {1,2,3}, TC = {1,3}, ET = {3} for every cohort.
# BraTS-PEDs 2023 releases three labels like GLI (ET, NC = NET+CC+necrosis, ED; arXiv 2305.17033 s2.1.1) and on disk
# label 3 is T1c-bright (mean z 1.40), 1 is the large non-enhancing part, 2 the small FLAIR-bright edema -> identity.
LABEL_MAPS = {}


def load(p):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=np.float32)


def crop_pad(a, cy, cx, fill):
    """(..., H, W) -> (..., SIZE, SIZE) window centred at (cy, cx)."""
    out = np.full(a.shape[:-2] + (SIZE, SIZE), fill, a.dtype)
    y0, x0 = cy - SIZE // 2, cx - SIZE // 2
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(a.shape[-2], y0 + SIZE), min(a.shape[-1], x0 + SIZE)
    out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[..., sy0:sy1, sx0:sx1]
    return out


def prep_visit(paths, odir, lmap=None, norm="z"):
    if not all(os.path.exists(p) for p in paths):
        return None
    vols = [load(p) for p in paths[:3]]
    seg = load(paths[3]).round().astype(np.uint8)
    if any(v.shape != seg.shape for v in vols):
        return None
    brain = np.any(np.stack(vols) != 0, axis=0)
    if brain.sum() < 1000:
        return None
    bg = -3.0 if norm == "z" else 0.0
    img = np.full((3,) + brain.shape, bg, np.float32)
    for c, v in enumerate(vols):
        x = v[brain]
        if norm == "z":
            img[c][brain] = np.clip((x - x.mean()) / (x.std() + 1e-6), -5, 5)
        else:                                                         # minmax: 0.5/99.5 percentiles -> [0, 1]
            lo, hi = np.percentile(x, [0.5, 99.5])
            img[c][brain] = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    if lmap:
        seg = np.select([seg == k for k in lmap], [np.uint8(v) for v in lmap.values()], 0).astype(np.uint8)
    else:
        seg[seg > 3] = 0                                              # resection cavity is not tumour
    img, seg, brain = img.transpose(3, 0, 1, 2), seg.transpose(2, 0, 1), brain.transpose(2, 0, 1)   # (Z, ..., X, Y)
    ys, xs = np.where(brain.any(0))
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    img, seg, brain = crop_pad(img, cy, cx, bg), crop_pad(seg, cy, cx, 0), crop_pad(brain, cy, cx, False)
    os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(odir, "img.npy"), img.astype(np.float16))
    np.save(os.path.join(odir, "seg.npy"), seg)
    bz = np.where(brain.any((1, 2)))[0].tolist()
    tz = np.where((seg > 0).any((1, 2)))[0].tolist()
    return dict(path=odir, n_slices=int(img.shape[0]), brain_slices=bz, tumor_slices=tz,
                tumor_vox=int((seg > 0).sum()), brain_extent=[int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)])


def visits(layout, src, cohort):
    """Yields (patient, visit, [t1c, t2, flair, seg] paths)."""
    f = FILES[layout]
    root = os.path.join(src, cohort)
    for case in sorted(os.listdir(root)):
        d = os.path.join(root, case)
        if not os.path.isdir(d):
            continue
        if layout == "brats":
            yield case.rsplit("-", 1)[0], case.rsplit("-", 1)[1], [os.path.join(d, f"{case}-{s}.nii.gz") for s in f]
            continue
        for vis in sorted(os.listdir(d)):
            vd = os.path.join(d, vis)
            meta = os.path.join(vd, "seg_gt.json")
            if not os.path.isdir(vd) or not os.path.exists(meta) or not json.load(open(meta)).get("aligned"):
                continue
            yield case, vis, [os.path.join(vd, f"{s}.nii.gz") for s in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=list(FILES), required=True)
    p.add_argument("--src", default=None)
    p.add_argument("--out", default=OUT)
    p.add_argument("--cohorts", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0, help="visits per cohort (smoke)")
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--norm", choices=["z", "minmax"], default="z", help="minmax: write to a separate --out")
    a = p.parse_args()
    src = a.src or SRC[a.layout]
    for c in a.cohorts:
        jobs = list(visits(a.layout, src, c))
        if a.limit:
            jobs = jobs[:a.limit]
        args = [(paths, os.path.join(a.out, c, f"{pat}__{vis}"), LABEL_MAPS.get(c), a.norm) for pat, vis, paths in jobs]
        with Pool(a.procs) as pool:
            outs = pool.starmap(prep_visit, args, chunksize=1)
        idx, n_skip = [], 0
        for (pat, vis, _), e in zip(jobs, outs):
            if e is None:
                n_skip += 1; continue
            phase = re.sub(r"_\d+$", "", re.sub(r"^\d+_", "", vis)) if a.layout == "long" else ""
            idx.append(dict(cohort=c, patient=pat, visit=vis, phase=phase, **e))
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
        json.dump(dict(cohort=c, layout=a.layout, files=FILES[a.layout], label_map=LABEL_MAPS.get(c), visits=idx),
                  open(os.path.join(a.out, c, "index.json"), "w"), indent=1)
        ph = {}
        for e in idx:
            ph[e["phase"]] = ph.get(e["phase"], 0) + 1
        print("COHORT_DONE", c, len(idx), "visits", n_skip, "skipped", json.dumps(ph), flush=True)

if __name__ == "__main__":
    main()
