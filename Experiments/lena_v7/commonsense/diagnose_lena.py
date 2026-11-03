"""Element checks (E1-E8) and insight dumps (V1-V8) for a commonsense LeNA / LoRA run.

Accuracy here is candidate scoring (sum log-probability of "the correct answer is <cand>" after the protocol prompt,
argmax over the task's candidates), on the first --n examples of each test set. It is a diagnostic accuracy, cheaper
than and different from the protocol's beam-search accuracy (evaluate_lena.py), and is used only to compare
conditions on the same checkpoint.

Conditions (LeNA runs):  learned | fixed (s = 0) | plain (s = 1) | id_prior | self_static (own prompt statistics at
every position) | swap (statistics of a donor prompt from another task) | swap matrix (donor task x target task).
Dumps: per example and module, at the last prompt token: s, F, p, fitted ordinates theta; s>0 by prefix position;
token-level s for a few prompts; (u, v) scatters for a few modules; the same statistics on GSM8K prompts (shift).

  python diagnose_lena.py --base_model meta-llama/Meta-Llama-3-8B --run_dir runs/llama3_lena --n 200
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, install_lena_hooks, lena_layers, lena_override  # noqa: E402
from finetune_lena import load_tokenizer  # noqa: E402
from evaluate_lena import ALL, generate_prompt  # noqa: E402

CANDS = {"boolq": ["true", "false"], "piqa": ["solution1", "solution2"],
         "social_i_qa": ["answer1", "answer2", "answer3"], "hellaswag": [f"ending{i}" for i in range(1, 5)],
         "winogrande": ["option1", "option2"], "ARC-Challenge": [f"answer{i}" for i in range(1, 6)],
         "ARC-Easy": [f"answer{i}" for i in range(1, 6)], "openbookqa": [f"answer{i}" for i in range(1, 5)]}
POS_BINS = [0, 8, 16, 32, 64, 96, 128, 192, 256, 1024]


def candidates(ds, instruction):
    c = CANDS[ds]
    if ds.startswith("ARC"):
        c = [a for a in c if a.lower() in instruction.lower()] or c[:4]
    return c


class Scorer:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.layers = lena_layers(model)
        self.names = [n for n, m in model.named_modules() if m in set(self.layers)]

    def _batch(self, prompt, cands):
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        seqs, spans = [], []
        for c in cands:
            c_ids = self.tok(prompt + "the correct answer is " + c, add_special_tokens=True)["input_ids"]
            seqs.append(c_ids); spans.append(len(p_ids))
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):                                   # right padding: earlier positions unaffected
            ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        return ids.to(self.model.device), mask.to(self.model.device), spans, [len(s) for s in seqs], len(p_ids)

    @torch.no_grad()
    def score(self, prompt, cands, **override):
        ids, mask, spans, lens, _ = self._batch(prompt, cands)
        with lena_override(**override):
            logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
        lp = F.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        tot = [lp[i, spans[i] - 1:lens[i] - 1].sum().item() for i in range(len(cands))]
        return max(range(len(cands)), key=lambda i: tot[i]), tot

    @torch.no_grad()
    def record(self, prompt, log=False):
        """Run the prompt alone; keep each layer's last-position statistics (for injection) and, with log, the
        per-layer diagnostics of every position."""
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        for L in self.layers:
            L.log_stats = log
        with lena_override(stats="record"):
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
        for L in self.layers:
            L.log_stats = False
        return [L._rec for L in self.layers], ids.shape[1]

    def set_inject(self, stats):
        for L, s in zip(self.layers, stats):
            L._inj = s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--adapter_file", default="adapter.pt")
    p.add_argument("--data_root", default="dataset")
    p.add_argument("--n", type=int, default=200, help="examples per task")
    p.add_argument("--n_matrix", type=int, default=50, help="examples per task for the donor x target matrix")
    p.add_argument("--n_shift", type=int, default=400, help="GSM8K prompts for the shift score")
    p.add_argument("--out", default="diag_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true", help="torch.compile the LeNA fit path (same outputs)")
    args = p.parse_args()
    lena_v7.set_compile(args.compile)
    random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = json.load(open(os.path.join(args.run_dir, "lena_config.json")))
    tok = load_tokenizer(args.base_model); tok.padding_side = "right"
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if cfg.get("bf16") else torch.float16) if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype,
                                                 device_map={"": 0} if cuda else {"": "cpu"})
    kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None)
    inject(model, cfg["targets"], adapter=cfg["adapter"], **kw)
    sd = torch.load(os.path.join(args.run_dir, args.adapter_file), map_location="cpu")
    for n_, prm in model.named_parameters():
        if n_ in sd:
            prm.data = sd[n_].to(prm.device, torch.float32)
    model.eval(); install_lena_hooks(model)
    S = Scorer(model, tok)
    is_lena = cfg["adapter"] == "lena" and len(S.layers) > 0
    out_dir = os.path.join(args.run_dir, "diag"); os.makedirs(out_dir, exist_ok=True)

    data = {ds: json.load(open(os.path.join(args.data_root, ds, "test.json")))[:args.n] for ds in ALL}
    conds = ["learned"] + (["fixed", "plain", "id_prior", "self_static", "swap"] if is_lena else [])
    acc = {c: {ds: 0 for ds in ALL} for c in conds}
    dump = dict(names=S.names, tasks=[], s=[], F=[], p=[], theta=[], correct=[], pos_hist=None, tokens=[],
                scatter=[])
    nb = len(POS_BINS) - 1
    pos_pos = torch.zeros(len(S.layers), nb, dtype=torch.float64); pos_cnt = torch.zeros(nb, dtype=torch.float64)
    scatter_mods = [i for i, n in enumerate(S.names) if any(n.endswith(f"layers.{l}.mlp.{m}") for l in (4, 16, 28)
                                                            for m in ("up_proj", "down_proj"))]
    t0 = time.time()
    for ti, ds in enumerate(ALL):
        others = [d for d in ALL if d != ds]
        for k, ex in enumerate(data[ds]):
            prompt = generate_prompt(ex["instruction"]); cands = candidates(ds, ex["instruction"])
            gold = cands.index(ex["answer"]) if ex["answer"] in cands else -1
            pred, _ = S.score(prompt, cands)
            acc["learned"][ds] += int(pred == gold)
            if not is_lena:
                continue
            for c, ov in (("fixed", dict(mode="fixed")), ("plain", dict(mode="plain")),
                          ("id_prior", dict(prior="identity"))):
                acc[c][ds] += int(S.score(prompt, cands, **ov)[0] == gold)
            log = k < 3                                               # token-level dumps for a few prompts
            own, N = S.record(prompt, log=True)
            S.set_inject(own); acc["self_static"][ds] += int(S.score(prompt, cands, stats="inject")[0] == gold)
            dd = random.choice(others); dex = random.choice(data[dd])
            don, _ = S.record(generate_prompt(dex["instruction"]))
            S.set_inject(don); acc["swap"][ds] += int(S.score(prompt, cands, stats="inject")[0] == gold)
            # dumps from the logged prompt pass (re-run with log, since record() above with log=True stored .last)
            ls = [L.last for L in S.layers]
            dump["tasks"].append(ds); dump["correct"].append(int(pred == gold))
            dump["s"].append(torch.stack([d["s"][0, -1] for d in ls]).half().cpu())        # (M, r)
            dump["F"].append(torch.stack([d["F"][0, -1] for d in ls]).float().cpu())
            dump["p"].append(torch.stack([d["p"][0, -1] for d in ls]).to(torch.uint8).cpu())
            dump["theta"].append(torch.stack([d["theta"][0, -1] for d in ls]).half().cpu())  # (M, r, K)
            spos = torch.stack([(d["s"][0] > 0).double().mean(-1) for d in ls]).cpu()        # (M, N)
            for b in range(nb):
                lo, hi = POS_BINS[b], min(POS_BINS[b + 1], N)
                if hi > lo:
                    pos_pos[:, b] += spos[:, lo:hi].sum(1); pos_cnt[b] += hi - lo
            if log:
                dump["tokens"].append(dict(task=ds, ids=tok(prompt)["input_ids"],
                                           s=torch.stack([d["s"][0].mean(-1) for d in ls]).half().cpu()))  # (M, N)
                dump["scatter"].append(dict(task=ds, mods=scatter_mods,
                                            u=torch.stack([ls[i]["u"][0] for i in scatter_mods]).half().cpu(),
                                            v=torch.stack([ls[i]["v"][0] for i in scatter_mods]).half().cpu(),
                                            theta=torch.stack([ls[i]["theta"][0, -1] for i in scatter_mods]).cpu()))
            for L in S.layers:
                L.last = None
        print(f"{ds}: " + ", ".join(f"{c} {100 * acc[c][ds] / len(data[ds]):.1f}" for c in conds)
              + f"  [{time.time() - t0:.0f}s]", flush=True)

    summary = {c: {ds: acc[c][ds] / len(data[ds]) for ds in ALL} for c in conds}
    for c in conds:
        summary[c]["avg"] = sum(summary[c][ds] for ds in ALL) / len(ALL)
    res = dict(n=args.n, adapter=cfg["adapter"], scoring_acc=summary)

    if is_lena:
        dump["pos_hist"] = dict(bins=POS_BINS, frac_pos=(pos_pos / pos_cnt.clamp_min(1)).float())
        dump["rho"] = torch.stack([L.rho.detach().float().cpu() for L in S.layers])             # (M, r, K)
        dump["omega"] = torch.stack([L.omega.detach().float().cpu() for L in S.layers])
        # donor task x target task matrix (V2)
        mat = torch.zeros(len(ALL), len(ALL))
        for ti, ds in enumerate(ALL):
            for k, ex in enumerate(data[ds][:args.n_matrix]):
                prompt = generate_prompt(ex["instruction"]); cands = candidates(ds, ex["instruction"])
                gold = cands.index(ex["answer"]) if ex["answer"] in cands else -1
                for di, dd in enumerate(ALL):
                    pool = [e for e in data[dd] if e is not ex]
                    don, _ = S.record(generate_prompt(random.choice(pool)["instruction"]))
                    S.set_inject(don)
                    mat[di, ti] += int(S.score(prompt, cands, stats="inject")[0] == gold)
            mat[:, ti] /= min(args.n_matrix, len(data[ds]))
            print(f"matrix target {ds} done [{time.time() - t0:.0f}s]", flush=True)
        res["swap_matrix"] = dict(tasks=ALL, acc=mat.tolist(), note="rows donor task, columns target task")
        # shift: statistics on GSM8K prompts (never trained on) vs commonsense prompts
        gsm = json.load(open(os.path.join(args.data_root, "gsm8k", "test.json")))[:args.n_shift]
        sh = dict(s=[], F=[])
        for ex in gsm:
            S.record(generate_prompt(ex["instruction"]), log=True)
            ls = [L.last for L in S.layers]
            sh["s"].append(torch.stack([d["s"][0, -1] for d in ls]).half().cpu())
            sh["F"].append(torch.stack([d["F"][0, -1] for d in ls]).float().cpu())
            for L in S.layers:
                L.last = None
        dump["shift_gsm8k"] = {k: torch.stack(v) for k, v in sh.items()}
        for k in ("s", "F", "p", "theta"):
            dump[k] = torch.stack(dump[k])
        torch.save(dump, os.path.join(out_dir, f"{args.out}_dump.pt"))
    json.dump(res, open(os.path.join(out_dir, f"{args.out}.json"), "w"), indent=1)
    print(json.dumps({c: round(100 * summary[c]["avg"], 2) for c in conds}))


if __name__ == "__main__":
    main()
