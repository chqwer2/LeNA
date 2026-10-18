"""Answer accuracy for the commonsense suite, by scoring each option's likelihood.

Why this exists: training reports `token_acc`, next-token accuracy over the WHOLE
sequence. Each example is a passage + question + options + a one-token answer inside
a 512-token window, so >95% of the scored tokens are the passage and the question --
text that is identical across adapters and essentially fixed by the frozen base. The
metric is therefore dominated by generic language modelling and compresses every
method into the same band (we measured 0.709-0.715 for LoRA r=4/8/16 and for every
LeNA variant, against 0.540 for the frozen base). It cannot see the thing we are
trying to compare.

This scores only the answer: build the same prompt used in training, truncated at
"Answer: ", score each candidate option under the model, take the argmax, and
compare with the gold option. That is what the LLM-Adapters / commonsense-170k
line of work reports.

Runs against a SAVED adapter, so finished runs can be re-scored without retraining.

Usage:
    python Experiments/eval_choice.py --adapter runs/e1_lora_r4_s1/lora
    python Experiments/eval_choice.py --base_only          # frozen base
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPECS = [
    ("google/boolq", None),
    ("ybisk/piqa", None),
    ("allenai/social_i_qa", None),
    ("Rowan/hellaswag", None),
    ("allenai/winogrande", "winogrande_xl"),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/openbookqa", None),
]


def _letters(n):
    return [chr(ord("A") + i) for i in range(n)]


def to_choice_example(dp, ex):
    """(prompt ending in 'Answer: ', [candidate strings], gold index) or None.

    The prompt must match format_example_to_text() in Llama_Adaptation.py exactly up
    to the answer, otherwise the model is scored out of distribution.
    """
    dp = dp.lower().strip()

    if dp == "google/boolq":
        p = ("### Task: Answer the question based on the passage.\n\n"
             f"Passage:\n{ex['passage']}\n\n"
             f"Question: {ex['question']}\n\n"
             "Answer:")
        return p, [" yes", " no"], (0 if bool(ex["answer"]) else 1)

    if dp in ("piqa", "ybisk/piqa"):
        sols = [ex.get("sol1", ""), ex.get("sol2", "")]
        try:
            gold = int(ex.get("label", -1))
        except Exception:
            return None
        if gold not in (0, 1):
            return None
        p = ("### Task: Choose the best solution.\n\n"
             f"Goal: {ex.get('goal', '')}\n"
             f"A. {sols[0]}\n"
             f"B. {sols[1]}\n\n"
             "Answer:")
        return p, [" A", " B"], gold

    if dp == "allenai/social_i_qa":
        try:
            lab = int(ex.get("label", -1))
        except Exception:
            return None
        if lab not in (1, 2, 3):
            return None
        p = ("### Task: Choose the best answer.\n\n"
             f"Context: {ex['context']}\n"
             f"Question: {ex['question']}\n"
             f"A. {ex['answerA']}\n"
             f"B. {ex['answerB']}\n"
             f"C. {ex['answerC']}\n\n"
             "Answer:")
        return p, [" A", " B", " C"], lab - 1

    if dp == "rowan/hellaswag":
        ctx = ex.get("ctx") or ((ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip())
        endings = list(ex.get("endings") or [])
        try:
            gold = int(ex.get("label", -1))
        except Exception:
            return None
        if not (0 <= gold < len(endings)):
            return None
        body = "\n".join(f"{c}. {t}" for c, t in zip(_letters(len(endings)), endings))
        p = ("### Task: Pick the most plausible continuation.\n\n"
             f"Context: {ctx}\n\n{body}\n\nAnswer:")
        return p, [f" {c}" for c in _letters(len(endings))], gold

    if dp.startswith("allenai/winogrande"):
        try:
            ans = int(ex.get("answer", -1))
        except Exception:
            return None
        if ans not in (1, 2):
            return None
        p = ("### Task: Fill in the blank with the correct option.\n\n"
             f"Sentence: {ex['sentence']}\n"
             f"A. {ex['option1']}\n"
             f"B. {ex['option2']}\n\n"
             "Answer:")
        return p, [" A", " B"], ans - 1

    if dp in ("allenai/ai2_arc", "allenai/openbookqa"):
        stem = ex.get("question") if dp == "allenai/ai2_arc" else ex.get("question_stem")
        ch = ex.get("choices", {})
        labels, texts = ch.get("label", []), ch.get("text", [])
        key = ex.get("answerKey", "")
        if key not in labels:
            return None
        body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        p = ("### Task: Choose the correct answer.\n\n"
             f"Question: {stem}\n\n{body}\n\nAnswer:")
        return p, [f" {lab}" for lab in labels], labels.index(key)

    return None


@torch.no_grad()
def score_options(model, tok, prompt, options, device, max_len=512):
    """Mean per-token log-likelihood of each option continuing the prompt."""
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    if ids.numel() > max_len - 8:
        ids = ids[-(max_len - 8):]          # keep the tail: question + options
    scores = []
    for opt in options:
        opt_ids = tok(opt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        full = torch.cat([ids, opt_ids]).unsqueeze(0).to(device)
        logits = model(full).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = full[0, 1:]
        n_opt = opt_ids.numel()
        # length-normalized, so options of different token length compare fairly
        scores.append(lp[-n_opt:].gather(-1, tgt[-n_opt:].unsqueeze(-1)).mean().item())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", default=None, help="directory holding adapter_model.safetensors")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--limit", type=int, default=500, help="examples per dataset")
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
        print(f"[eval] loaded adapter {args.adapter}")
    else:
        print("[eval] frozen base model, no adapter")
    model.eval()
    device = next(model.parameters()).device

    results, accs = {}, []
    for dp, name in SPECS:
        ds = load_dataset(dp, name, trust_remote_code=True) if name else \
            load_dataset(dp, trust_remote_code=True)
        split = "validation" if "validation" in ds else ("test" if "test" in ds else "train")
        data = ds[split]
        tag = f"{dp}:{name}" if name else dp

        n_ok = n_tot = 0
        for ex in data:
            item = to_choice_example(dp, ex)
            if item is None:
                continue
            prompt, options, gold = item
            scores = score_options(model, tok, prompt, options, device)
            pred = int(max(range(len(scores)), key=scores.__getitem__))
            n_ok += int(pred == gold)
            n_tot += 1
            if n_tot >= args.limit:
                break
        acc = n_ok / max(n_tot, 1)
        results[tag] = {"acc": acc, "n": n_tot}
        accs.append(acc)
        print(f"{tag:42s} acc={acc:.4f}  (n={n_tot})", flush=True)

    avg = sum(accs) / len(accs)
    results["AVERAGE"] = avg
    print(f"{'AVERAGE':42s} acc={avg:.4f}")

    out = args.out or (os.path.join(args.adapter, "choice_metrics.json")
                       if args.adapter and not args.base_only else "choice_metrics_base.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
