"""Generation-based evaluation for the heterogeneous task mix.

eval_choice.py can only rank a fixed set of options, which covers the commonsense
suite and the GLUE label tasks. The tasks that make the mixture genuinely
heterogeneous do not have options at all -- their output spaces are a number, a span
of the passage, a free label -- and those are exactly the tasks whose conflict with
"emit a letter" the input-conditional claim is about. This scores them by generating.

Metrics
  gsm8k : exact match on the final number (the standard protocol)
  squad : SQuAD exact match and token F1 over the gold answer set
  glue  : exact match on the label word

Runs against a SAVED adapter, so any finished run can be scored without retraining.

Usage:
    python Experiments/eval_generate.py --adapter runs/cf_lena_n3/<tag> --tasks gsm8k,squad
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASKS = {
    "gsm8k": dict(path="openai/gsm8k", config="main", split="test", max_new_tokens=160),
    "squad": dict(path="rajpurkar/squad", config=None, split="validation", max_new_tokens=32),
    "mnli": dict(path="nyu-mll/glue", config="mnli", split="validation_matched", max_new_tokens=8),
    "sst2": dict(path="nyu-mll/glue", config="sst2", split="validation", max_new_tokens=8),
}


def prompt_and_gold(task, ex):
    """Prompt ending in 'Answer:' plus the gold answer(s), matching training format."""
    if task == "gsm8k":
        p = ("### Task: Solve the math problem. End with '#### <number>'.\n\n"
             f"Question: {ex['question']}\n\nAnswer:")
        return p, [str(ex["answer"]).split("####")[-1].strip()]
    if task == "squad":
        p = ("### Task: Answer the question using a span from the passage.\n\n"
             f"Passage:\n{ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:")
        return p, list(ex["answers"]["text"])
    if task == "mnli":
        p = ("### Task: Does the premise entail the hypothesis?\n"
             "Reply with entailment, neutral or contradiction.\n\n"
             f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n\nAnswer:")
        return p, [{0: "entailment", 1: "neutral", 2: "contradiction"}[int(ex["label"])]]
    if task == "sst2":
        p = ("### Task: Is the sentiment of the sentence positive or negative?\n\n"
             f"Sentence: {ex['sentence']}\n\nAnswer:")
        return p, [{0: "negative", 1: "positive"}[int(ex["label"])]]
    raise ValueError(task)


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def score(task, pred, golds):
    """-> (exact_match, f1). f1 is only meaningful for squad."""
    if task == "gsm8k":
        p, g = last_number(pred), last_number(golds[0])
        return float(p is not None and g is not None and p == g), 0.0
    if task in ("mnli", "sst2"):
        p = _norm(pred).split()
        return float(bool(p) and p[0] == _norm(golds[0])), 0.0
    # squad: max over the gold answer set
    em = max(float(_norm(pred) == _norm(g)) for g in golds)
    best_f1 = 0.0
    for g in golds:
        pt, gt = _norm(pred).split(), _norm(g).split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n:
            prec, rec = n / len(pt), n / len(gt)
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return em, best_f1


@torch.no_grad()
def generate(model, tok, prompt, device, max_new_tokens, max_len=1024):
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > max_len:
        ids = ids[:, -max_len:]
    out = model.generate(ids.to(device), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--tasks", default="gsm8k,squad,mnli,sst2")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        # NOT device_map="auto": PEFT then loads the adapter weights on CPU while the
        # base is sharded on GPU, and the first A(x) dies on a device mismatch.
        device_map={"": 0} if torch.cuda.is_available() else None,
        cache_dir=cache_dir)
    if args.adapter and not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[eval-gen] adapter {args.adapter}")
    else:
        print("[eval-gen] frozen base model")
    model.eval()
    device = next(model.parameters()).device

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        cfg = TASKS[task]
        ds = load_dataset(cfg["path"], cfg["config"], trust_remote_code=True) if cfg["config"] \
            else load_dataset(cfg["path"], trust_remote_code=True)
        data = ds[cfg["split"]]
        ems, f1s, n = [], [], 0
        for ex in data:
            prompt, golds = prompt_and_gold(task, ex)
            pred = generate(model, tok, prompt, device, cfg["max_new_tokens"])
            em, f1 = score(task, pred, golds)
            ems.append(em)
            f1s.append(f1)
            n += 1
            if n >= args.limit:
                break
        res = {"exact_match": sum(ems) / max(n, 1), "n": n}
        if task == "squad":
            res["f1"] = sum(f1s) / max(n, 1)
        results[task] = res
        print(f"{task:8s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in res.items()), flush=True)

    out = args.out or (os.path.join(args.adapter, "generate_metrics.json")
                       if args.adapter and not args.base_only else "generate_metrics_base.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
