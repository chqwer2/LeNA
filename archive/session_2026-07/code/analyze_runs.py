"""Aggregate LeNA/LoRA run directories into the comparison table the paper needs.

For every run it reports the trainable adapter parameter count (read straight from
adapter_model.safetensors, so "matched params" is a measurement and not a claim),
the per-dataset token accuracy, the average across datasets, and -- for LeNA runs --
how much of the model actually went nonlinear according to the gate where-map.
Runs that differ only by seed are collapsed into mean +/- std.

Usage:
    python Experiments/analyze_runs.py runs/                 # table
    python Experiments/analyze_runs.py runs/ --csv out.csv
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict


def adapter_param_count(run_dir):
    """Total trainable adapter params, from the safetensors header."""
    path = os.path.join(run_dir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        return None
    try:
        from safetensors import safe_open
        total = 0
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                n = 1
                for d in shape:
                    n *= d
                total += n
        return total
    except Exception:
        return None


def read_metrics(run_dir):
    path = os.path.join(run_dir, "test_metrics_by_dataset.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    accs = {}
    for ds, m in blob.items():
        for k, v in m.items():
            if k.endswith("_token_acc"):
                accs[ds] = float(v)
    return accs or None


def read_gate_map(run_dir):
    path = os.path.join(run_dir, "lena_gate_map.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        gm = json.load(f)
    vals = [float(v) for v in gm.values()]
    if not vals:
        return None
    return {
        "n_gated": len(vals),
        "mean_openness": sum(vals) / len(vals),
        "frac_open": sum(1 for v in vals if v > 0.5) / len(vals),
    }


def parse_tag(run_dir):
    """Pull method / rank / seed / gate-L1 out of the run directory path."""
    name = os.path.relpath(run_dir).replace(os.sep, "/")
    out = {"method": None, "rank": None, "seed": None, "gate_l1": None}
    m = re.search(r"/(e\d)_([a-z]+)_r(\d+)_s(\d+)/", name + "/")
    if m:
        out.update(method=m.group(2), rank=int(m.group(3)), seed=int(m.group(4)))
    m = re.search(r"/e3_l1_([0-9.e+-]+)_s(\d+)/", name + "/")
    if m:
        out.update(method="lena", gate_l1=m.group(1), seed=int(m.group(2)), rank=16)
    if out["method"] is None:
        for cand in ("lena_d", "lena", "dora", "lora"):
            if cand in name:
                out["method"] = cand
                break
    return out


def find_runs(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "test_metrics_by_dataset.json" in filenames:
            yield dirpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing run subdirectories")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(find_runs(args.root)):
        accs = read_metrics(run_dir)
        if not accs:
            continue
        tag = parse_tag(run_dir)
        rows.append({
            "run": os.path.relpath(run_dir, args.root),
            **tag,
            "params": adapter_param_count(run_dir),
            "accs": accs,
            "avg": sum(accs.values()) / len(accs),
            "gate": read_gate_map(run_dir),
        })

    if not rows:
        print(f"No finished runs under {args.root} "
              "(a run counts as finished once it writes test_metrics_by_dataset.json)")
        return

    datasets = sorted({d for r in rows for d in r["accs"]})

    print(f"{'run':<34}{'method':<8}{'r':>4}{'seed':>5}{'params':>12}", end="")
    for d in datasets:
        print(f"{d.split('/')[-1][:12]:>13}", end="")
    print(f"{'avg':>8}{'%open':>8}")
    print("-" * (34 + 8 + 4 + 5 + 12 + 13 * len(datasets) + 16))

    for r in rows:
        params = f"{r['params']:,}" if r["params"] else "?"
        print(f"{r['run'][:33]:<34}{str(r['method']):<8}{str(r['rank'] or ''):>4}"
              f"{str(r['seed'] or ''):>5}{params:>12}", end="")
        for d in datasets:
            print(f"{r['accs'].get(d, float('nan')):>13.4f}", end="")
        gate = f"{r['gate']['frac_open']*100:.1f}" if r["gate"] else "-"
        print(f"{r['avg']:>8.4f}{gate:>8}")

    # collapse seeds: everything except the seed identifies a configuration
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["rank"], r["gate_l1"])].append(r["avg"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print("\nAcross seeds (avg accuracy):")
        for (method, rank, l1), vals in sorted(multi.items(), key=lambda kv: str(kv[0])):
            label = f"{method} r={rank}" + (f" l1={l1}" if l1 else "")
            print(f"  {label:<28} {statistics.mean(vals):.4f} +/- "
                  f"{statistics.stdev(vals):.4f}  (n={len(vals)})")
    else:
        print("\nOnly one seed per configuration so far -- no std yet.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "method", "rank", "seed", "gate_l1", "params",
                        *datasets, "avg", "frac_open"])
            for r in rows:
                w.writerow([r["run"], r["method"], r["rank"], r["seed"], r["gate_l1"],
                            r["params"], *[r["accs"].get(d, "") for d in datasets],
                            f"{r['avg']:.6f}",
                            f"{r['gate']['frac_open']:.4f}" if r["gate"] else ""])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
