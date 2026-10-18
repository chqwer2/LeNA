"""Mechanism probe: how much NONLINEARITY does a trained aurora adapter actually use,
and does usage grow as rank binds (small r)?

For each aurora activation module phi: R^r->R^r, we capture the real code z and phi(z)
over GSM8K tokens, then fit the BEST LINEAR map M (r x r): phi(z) ~ z @ M.
  nonlin_ratio = ||phi(z) - z@M||_F / ||phi(z)||_F     (0 => phi acts linearly on data)
  ident_dev    = ||phi(z) - z||_F   / ||z||_F           (deviation from LoRA/identity)
Also confirms the update output rank <= r (nonlinearity does NOT multiply output rank).

Prediction (mechanism for the regime finding): nonlin_ratio DECREASES with r.
Usage: python measure_nonlin.py --adapter <dir> --base_model ... [--n_q 24]
"""
import argparse, json, os, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--n_q", type=int, default=24)
    args = ap.parse_args()

    cache_dir = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache_dir)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # nominal r from adapter_config
    cfgp = glob.glob(os.path.join(args.adapter, "adapter_config.json"))
    r_nom = None
    if cfgp:
        r_nom = json.load(open(cfgp[0])).get("r")

    # locate aurora/lena activation submodules (skip Identity => lora has none)
    import torch.nn as nn
    caps = {}   # name -> {"z": [], "phi": []}
    hooks = []
    def mk(name):
        def hook(mod, inp, out):
            z = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
            p = out.detach().reshape(-1, out.shape[-1]).float().cpu()
            caps[name]["z"].append(z); caps[name]["phi"].append(p)
        return hook
    n_act = 0
    for mod_name, module in model.named_modules():
        act = getattr(module, "act", None)
        if isinstance(act, nn.ModuleDict):
            for k, sub in act.items():
                if sub.__class__.__name__ == "Identity":
                    continue
                nm = f"{mod_name}.{k}"
                caps[nm] = {"z": [], "phi": []}
                hooks.append(sub.register_forward_hook(mk(nm)))
                n_act += 1
    print(f"[probe] adapter={args.adapter} r_nom={r_nom} act_modules={n_act}")
    if n_act == 0:
        print("NO_ACT (likely linear LoRA) -> nonlin_ratio=0 by construction"); return

    # run GSM8K tokens forward
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    texts = [ds[i]["question"] for i in range(min(args.n_q, len(ds)))]
    dev = next(model.parameters()).device
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=160).to(dev)
            model(**ids)
    for h in hooks:
        h.remove()

    # per-module: best-linear residual + identity deviation + output-subspace note
    import math
    nl, idv, effr = [], [], []
    for nm, d in caps.items():
        if not d["z"]:
            continue
        Z = torch.cat(d["z"], 0)      # (N, r)
        P = torch.cat(d["phi"], 0)    # (N, r)
        # best linear M: P ~ Z @ M
        sol = torch.linalg.lstsq(Z, P)
        M = sol.solution
        res = (P - Z @ M).norm() / (P.norm() + 1e-9)
        dev_id = (P - Z).norm() / (Z.norm() + 1e-9)
        # effective rank of phi(z) output over data (should be <= r)
        s = torch.linalg.svdvals(P - P.mean(0, keepdim=True))
        er = float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))  # participation ratio
        nl.append(float(res)); idv.append(float(dev_id)); effr.append(er)
    import statistics as st
    print(f"RESULT adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"r_nom={r_nom} n_mod={len(nl)} "
          f"nonlin_ratio_mean={st.mean(nl):.4f} nonlin_ratio_med={st.median(nl):.4f} "
          f"ident_dev_mean={st.mean(idv):.4f} eff_out_rank_mean={st.mean(effr):.2f}")


if __name__ == "__main__":
    main()
