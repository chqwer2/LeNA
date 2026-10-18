"""Linearize-ablation: measure how LOAD-BEARING the nonlinearity is, per rank.

For a trained aurora adapter, replace each activation phi(z) by its BEST LINEAR fit
z @ M* (M* = argmin ||phi(z) - z M||), turning the adapter into an effectively-linear
rank-r adapter with the SAME B, A. Re-evaluate GSM8K exact-match.

  drop = acc(original) - acc(linearized)   = the accuracy VALUE of the nonlinearity.

Prediction (mechanism): drop is LARGE at r=2 (nonlinearity load-bearing) and ~0 at r=8
(nonlinearity present but not needed -> explains the regime finding).

Usage: python ablate_linearize.py --adapter <dir> --n_fit 24 --limit 150
"""
import argparse, os, glob, json
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import eval_generate as EG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--n_fit", type=int, default=24)
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    cache_dir = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache_dir)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    r_nom = None
    cfgp = glob.glob(os.path.join(args.adapter, "adapter_config.json"))
    if cfgp:
        r_nom = json.load(open(cfgp[0])).get("r")

    # collect act submodules
    act_mods = {}
    for mn, module in model.named_modules():
        act = getattr(module, "act", None)
        if isinstance(act, nn.ModuleDict):
            for k, sub in act.items():
                if sub.__class__.__name__ != "Identity":
                    act_mods[f"{mn}.{k}"] = sub
    if not act_mods:
        print(f"NO_ACT adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} (linear lora)"); return

    # ---- phase 1: probe to fit M* per module ----
    caps = {nm: {"z": [], "p": []} for nm in act_mods}
    hks = []
    def mk(nm):
        def h(m, i, o):
            caps[nm]["z"].append(i[0].detach().reshape(-1, i[0].shape[-1]).float().cpu())
            caps[nm]["p"].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
        return h
    for nm, sub in act_mods.items():
        hks.append(sub.register_forward_hook(mk(nm)))
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    with torch.no_grad():
        for i in range(min(args.n_fit, len(ds))):
            ids = tok(ds[i]["question"], return_tensors="pt", truncation=True, max_length=160).to(dev)
            model(**ids)
    for h in hks:
        h.remove()
    Mstar = {}
    for nm, d in caps.items():
        Z = torch.cat(d["z"], 0); P = torch.cat(d["p"], 0)
        Mstar[nm] = torch.linalg.lstsq(Z, P).solution.to(dev)  # (r,r)
    del caps

    # ---- phase 2: patch act forward -> z @ M* (linear surrogate) ----
    patch_hks = []
    def mk_patch(nm):
        M = Mstar[nm]
        def h(m, i, o):
            z = i[0]
            return (z.float() @ M).to(z.dtype)
        return h
    for nm, sub in act_mods.items():
        patch_hks.append(sub.register_forward_hook(mk_patch(nm)))

    # ---- phase 3: eval gsm8k with linearized adapter ----
    cfg = EG.TASKS["gsm8k"]
    dds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"], cache_dir=cache_dir)
    ems = []
    n = min(args.limit, len(dds))
    for i in range(n):
        prompt, golds = EG.prompt_and_gold("gsm8k", dds[i])
        pred = EG.generate(model, tok, prompt, dev, cfg["max_new_tokens"])
        em, _ = EG.score("gsm8k", pred, golds)
        ems.append(em)
    acc_lin = sum(ems) / max(n, 1)
    print(f"ABLATE adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"r_nom={r_nom} n={n} acc_linearized={acc_lin:.4f}")


if __name__ == "__main__":
    main()
