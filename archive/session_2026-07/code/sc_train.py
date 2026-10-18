"""SC-aware (self-consistency-aware) training of a low-rank adapter, STaR-style.
Fold the self-consistency objective into TRAINING: sample chains on TRAIN questions,
keep the ones that reach the GOLD answer (rejection sampling), fine-tune the adapter on
these self-generated correct chains -> the r2 adapter learns to generate correct reasoning
more CONSISTENTLY, so both greedy and sc_vote improve at test time.

Reports test greedy + sc_vote BEFORE and AFTER SC-aware training (held-out gsm8k test).
Usage: python sc_train.py --adapter <dir> --n_train 300 --n_sample 6 --epochs 2 --lr 5e-5 --temp 0.9 --eval_limit 100
"""
import argparse, os, re, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import eval_generate as EG


def sample_chains(model, tok, prompt, dev, max_new, n, temp):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp,
                             num_return_sequences=n, pad_token_id=tok.eos_token_id)
    plen = ids["input_ids"].shape[1]
    return [tok.decode(o[plen:], skip_special_tokens=True) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_sample", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--eval_limit", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=8)
    args = ap.parse_args()

    cache = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    adapt = [(n, p) for n, p in model.named_parameters()
             if any(t in n for t in ("lora_", ".act.", ".gate.", ".steer.", "lena_"))]
    for _, p in adapt: p.requires_grad_(True)

    cfg = EG.TASKS["gsm8k"]; mnt = cfg["max_new_tokens"]
    test = load_dataset(cfg["path"], cfg["config"], split="test", cache_dir=cache)
    train = load_dataset(cfg["path"], cfg["config"], split="train", cache_dir=cache)
    print(f"[sc-train] adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"n_train={args.n_train} n_sample={args.n_sample} epochs={args.epochs} lr={args.lr}")

    def gold_num(ex):
        return EG.last_number(ex["answer"].split("####")[-1])

    def eval_test(limit, do_vote):
        gc = vc = 0
        for i in range(limit):
            p, g = EG.prompt_and_gold("gsm8k", test[i])
            gc += EG.score("gsm8k", EG.generate(model, tok, p, dev, mnt), g)[0]
            if do_vote:
                ch = sample_chains(model, tok, p, dev, mnt, args.eval_n, args.temp)
                val = [a for a in (EG.last_number(c) for c in ch) if a is not None]
                if val: vc += EG.score("gsm8k", str(Counter(val).most_common(1)[0][0]), g)[0]
        return gc/limit, (vc/limit if do_vote else 0)

    # BEFORE
    g0, v0 = eval_test(args.eval_limit, True)
    print(f"[BEFORE] greedy={g0:.4f}  sc_vote={v0:.4f}")

    # Phase A: rejection-sample correct chains on TRAIN
    data = []
    for i in range(args.n_train):
        p = EG.prompt_and_gold("gsm8k", train[i])[0]; gold = gold_num(train[i])
        if gold is None: continue
        for c in sample_chains(model, tok, p, dev, mnt, args.n_sample, args.temp):
            if EG.last_number(c) == gold:
                data.append(p + c)
    print(f"[phase-A] collected {len(data)} correct self-generated chains from {args.n_train} train Q")

    # Phase B: SC-aware fine-tune on correct chains
    opt = torch.optim.AdamW([p for _, p in adapt], lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        idx = torch.randperm(len(data)).tolist()
        for b in range(0, len(idx), 4):
            texts = [data[j] for j in idx[b:b+4]]
            batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(dev)
            opt.zero_grad(); out = model(**batch, labels=batch["input_ids"]); out.loss.backward(); opt.step()
    model.eval()

    # AFTER
    g1, v1 = eval_test(args.eval_limit, True)
    print(f"[RESULT] BEFORE greedy={g0:.4f} vote={v0:.4f} | AFTER greedy={g1:.4f} vote={v1:.4f} "
          f"(dgreedy={g1-g0:+.4f} dvote={v1-v0:+.4f})")


if __name__ == "__main__":
    main()
