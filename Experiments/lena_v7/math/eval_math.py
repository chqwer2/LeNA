"""GSM8K (1319) and MATH (5000) evaluation of a finetune_math.py run, as in PiSSA: the pissa-dataset metamath/test.json
prompts as given, greedy decoding, up to 1024 new tokens, answers checked with PiSSA's utils/test_acc.py functions
(extract_answer_number for GSM8K, process_math_results for MATH; loaded from the PiSSA clone). HF generate with a KV
cache replaces vLLM; LeNA prefix statistics ride along through lena_stream.

  python eval_math.py --run_dir runs/math_lena [--limit 200] [--force_mode fixed]
"""
import argparse
import contextlib
import fractions
import json
import os
import random
import sys
import time
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, install_lena_hooks, lena_stream, lena_override  # noqa: E402


def pissa_checkers(path):
    """Function definitions of PiSSA utils/test_acc.py (the file parses arguments at import time). Its third-party
    `fraction.Fraction` is used only to parse 'a/b' strings, which `fractions.Fraction` parses identically."""
    sys.modules.setdefault("fraction", types.SimpleNamespace(Fraction=fractions.Fraction))
    src = open(path).read()
    src = src[:src.index("parser = argparse.ArgumentParser()")]
    ns = {}
    exec(compile(src, path, "exec"), ns)
    return ns["extract_answer_number"], ns["process_math_results"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--test", default="/mnt/scratch2/users/hchen/data/lena_v7/pissa-dataset/metamath/test.json")
    p.add_argument("--pissa", default="/mnt/scratch2/users/hchen/PiSSA/utils/test_acc.py")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="n of each type (first n, or a random n with --sample_seed)")
    p.add_argument("--sample_seed", type=int, default=-1, help=">= 0: random n per type (MATH test.json is sorted by subject; the first n are mostly precalculus)")
    p.add_argument("--force_mode", default=None, help="evaluation-time override: fixed | plain")
    p.add_argument("--tag", default="")
    p.add_argument("--reverse", action="store_true", help="process the length-sorted batches from the longest end")
    p.add_argument("--save_every", type=int, default=10, help="write the resume file every n batches")
    p.add_argument("--neff", action="store_true", help="R1: Stein noise variance x design effect 1+(n-1)rho_tok (causal: prefix)")
    p.add_argument("--stein_log", action="store_true", help="log the mean Stein s/bend of all LeNA fits (no change)")
    p.add_argument("--compile", action="store_true", help="torch.compile the LeNA fit path (same outputs)")
    p.add_argument("--triton", action="store_true", help="fused Triton kernels for the mode-lena fit (lena_triton.py)")
    a = p.parse_args()
    lena_v7.set_compile(a.compile)
    lena_v7.set_triton(a.triton)
    extract_gsm, check_math = pissa_checkers(a.pissa)

    cfg = json.load(open(os.path.join(a.run_dir, "lena_config.json")))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], padding_side="left", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, device_map={"": 0})
    kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None)
    inject(model, cfg["targets"], adapter=cfg["adapter"], **kw)
    sd = torch.load(os.path.join(a.run_dir, "adapter.pt"), map_location="cpu")
    for n_, prm in model.named_parameters():
        if n_ in sd:
            prm.data = sd[n_].to(prm.device, torch.float32)
    model.eval(); install_lena_hooks(model)
    if a.neff or a.stein_log:                                        # R1 runtime patch / Stein-s logging (lena_mri_ext)
        import atexit
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))
        import lena_mri_ext
        assert not a.triton, "--neff/--stein_log need the PyTorch fit path (no --triton)"
        lena_mri_ext.install(model, neff=a.neff)
        lena_mri_ext.EXT["acc"] = {}

        def _dump_s():
            acc = lena_mri_ext.EXT["acc"]
            st = {k: acc[k] / max(acc.get("n", 1), 1) for k in ("s", "bend") if k in acc}
            st["n_fits"] = acc.get("n", 0)
            print("STEIN_ALL", json.dumps(st), flush=True)
            json.dump(st, open(os.path.join(a.run_dir, "stein_eval_math.json"), "w"))
        atexit.register(_dump_s)

    data = json.load(open(a.test))
    if a.limit:
        pick = lambda L: [L[i] for i in sorted(random.Random(a.sample_seed).sample(range(len(L)), min(a.limit, len(L))))] if a.sample_seed >= 0 else L[:a.limit]
        data = pick([d for d in data if d["type"] == "gsm8k"]) + pick([d for d in data if d["type"] == "math"])
    order = sorted(range(len(data)), key=lambda i: len(data[i]["instruction"]))       # length-sorted batches
    if a.reverse:
        order = order[::-1]
    outs, t0 = [None] * len(data), time.time()
    tag = a.tag or (f"_{a.force_mode}" if a.force_mode else "")
    tag = tag + (f"_s{a.sample_seed}" if a.sample_seed >= 0 else "")
    part = os.path.join(a.run_dir, f"partial{tag}_limit{a.limit}.json")         # resume after a crash (same data/order)
    if os.path.exists(part):
        done = json.load(open(part)); outs = [done.get(str(i)) for i in range(len(data))]
        print(f"resumed {sum(o is not None for o in outs)}/{len(data)} from {part}", flush=True)
    ov = lena_override(mode=a.force_mode) if a.force_mode else contextlib.nullcontext()
    with ov:
        for b in range(0, len(order), a.batch_size):
            idx = order[b:b + a.batch_size]
            if all(outs[i] is not None for i in idx):
                continue
            enc = tok([data[i]["instruction"] for i in idx], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad(), lena_stream(model):
                seq = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for i, s in zip(idx, tok.batch_decode(seq[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                outs[i] = s
            if (b // a.batch_size) % a.save_every == 0:
                json.dump({str(i): o for i, o in enumerate(outs) if o is not None}, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
                print(f"{b + len(idx)}/{len(data)} [{time.time() - t0:.0f}s]", flush=True)
    res = {"gsm8k": [], "math": []}
    recs = []
    for d, o in zip(data, outs):
        if d["type"] == "gsm8k":
            y = extract_gsm(o); ok = y is not None and float(y) == float(d["output"])
        else:
            ok = bool(check_math(o, d["output"]))
        res[d["type"]].append(ok); recs.append(dict(d, response=o, correct=ok))
    summ = {k: sum(v) / max(len(v), 1) for k, v in res.items()} | {f"n_{k}": len(v) for k, v in res.items()}
    json.dump(recs, open(os.path.join(a.run_dir, f"responses{tag}.json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(a.run_dir, f"eval{tag}.json"), "w"), indent=1)
    print("RESULT", json.dumps(summ), f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
