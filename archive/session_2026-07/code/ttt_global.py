"""GLOBAL self-consistency TTT (transductive). NOT per-sample backprop.
  Phase 1: for the test questions, sample N chains, majority-vote -> keep consensus chains
           as pseudo-labeled data (uses UNLABELED test inputs; transductive TTT protocol).
  Phase 2: ONE global fine-tune of the low-rank adapter on all consensus chains.
  Phase 3: per-sample is just a FORWARD (greedy) with the globally-adapted adapter.
Compares: static_greedy(pre) | sc_vote | global_greedy(post). Optional held-out eval split.
Usage: python ttt_global.py --adapter <dir> --limit 120 --n 8 --epochs 2 --lr 1e-4 --temp 0.8 --holdout 0.5
"""
import argparse, os, math, torch
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
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--holdout", type=float, default=0.5)  # frac of questions used ONLY for eval (no adapt)
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
    ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"], cache_dir=cache)
    M = min(args.limit, len(ds))
    n_eval = int(M * args.holdout)
    adapt_idx = list(range(n_eval, M))     # adapt on these
    eval_idx = list(range(0, n_eval))      # eval on these (held out from adaptation)
    print(f"[global-ttt] adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"M={M} adapt={len(adapt_idx)} eval={len(eval_idx)} n={args.n} epochs={args.epochs} lr={args.lr}")

    def greedy_acc(idxs):
        c = 0
        for i in idxs:
            p, g = EG.prompt_and_gold("gsm8k", ds[i])
            c += EG.score("gsm8k", EG.generate(model, tok, p, dev, mnt), g)[0]
        return c / max(len(idxs), 1)

    # baseline greedy on eval split (pre-adapt)
    pre = greedy_acc(eval_idx)

    # Phase 1: consensus pseudo-data from ADAPT split + sc_vote acc on eval split
    pseudo = []
    for i in adapt_idx:
        p, g = EG.prompt_and_gold("gsm8k", ds[i])
        chains = sample_chains(model, tok, p, dev, mnt, args.n, args.temp)
        ans = [EG.last_number(c) for c in chains]
        valid = [a for a in ans if a is not None]
        if not valid: continue
        maj = Counter(valid).most_common(1)[0][0]
        for c, a in zip(chains, ans):
            if a == maj: pseudo.append(p + c)
    # sc_vote acc on eval split (per-sample forward, N samples, vote) -- no adaptation
    vc = 0
    for i in eval_idx:
        p, g = EG.prompt_and_gold("gsm8k", ds[i])
        chains = sample_chains(model, tok, p, dev, mnt, args.n, args.temp)
        valid = [a for a in (EG.last_number(c) for c in chains) if a is not None]
        if valid:
            maj = Counter(valid).most_common(1)[0][0]
            vc += EG.score("gsm8k", str(maj), g)[0]
    sc_vote = vc / max(len(eval_idx), 1)
    print(f"[phase1] pseudo_chains={len(pseudo)}  static_greedy(eval)={pre:.4f}  sc_vote(eval)={sc_vote:.4f}")

    # Phase 2: ONE global fine-tune of adapter on consensus chains
    opt = torch.optim.AdamW([p for _, p in adapt], lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        idx = torch.randperm(len(pseudo)).tolist()
        for b in range(0, len(idx), 4):
            texts = [pseudo[j] for j in idx[b:b+4]]
            batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(dev)
            opt.zero_grad()
            out = model(**batch, labels=batch["input_ids"])
            out.loss.backward(); opt.step()
    model.eval()

    # Phase 3: per-sample forward (greedy) on eval split with globally-adapted adapter
    post = greedy_acc(eval_idx)
    print(f"[RESULT] static_greedy={pre:.4f}  sc_vote={sc_vote:.4f}  GLOBAL_ttt_greedy={post:.4f}  "
          f"(post-static={post-pre:+.4f}  post-vote={post-sc_vote:+.4f})")


if __name__ == "__main__":
    main()
