"""Self-Consistency TTT: per-sample, sample N reasoning chains, majority-vote the answer,
then TTT-adapt the low-rank adapter on the chains that reached the consensus (self-training
toward the model's own agreement), and greedy-decode the final answer. Reset per sample.

Reports THREE numbers so we can tell what actually helps:
  static  = greedy, no adaptation (the r2 baseline)
  sc_vote = majority vote of N samples (strong baseline; SC alone already boosts)
  sc_ttt  = adapt toward consensus, then greedy   <- our method
The interesting claim: sc_ttt > sc_vote (adaptation internalizes consensus, beyond voting),
or at least sc_ttt >> static (a rank-2 adapter reaches much higher via test-time compute).

Usage: python ttt_sc.py --adapter <dir> --limit 40 --n 8 --ttt_steps 5 --ttt_lr 2e-5 --temp 0.8
"""
import argparse, os, torch
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
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ttt_steps", type=int, default=5)
    ap.add_argument("--ttt_lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    cache = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
    for _, p in adapt:
        p.requires_grad_(True)
    print(f"[sc-ttt] adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"n={args.n} K={args.ttt_steps} lr={args.ttt_lr} temp={args.temp} limit={args.limit}")

    cfg = EG.TASKS["gsm8k"]
    ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"], cache_dir=cache)
    N = min(args.limit, len(ds)); mnt = cfg["max_new_tokens"]
    s_static = s_vote = s_ttt = 0

    for i in range(N):
        prompt, golds = EG.prompt_and_gold("gsm8k", ds[i])
        # 1) static greedy
        g0 = EG.generate(model, tok, prompt, dev, mnt)
        s_static += EG.score("gsm8k", g0, golds)[0]
        # 2) sample N chains, majority vote
        chains = sample_chains(model, tok, prompt, dev, mnt, args.n, args.temp)
        ans = [EG.last_number(c) for c in chains]
        valid = [a for a in ans if a is not None]
        if not valid:
            continue
        maj = Counter(valid).most_common(1)[0][0]
        s_vote += EG.score("gsm8k", str(maj), golds)[0]
        # 3) TTT on consensus chains -> greedy
        good = [c for c, a in zip(chains, ans) if a == maj][:4]
        if good:
            snap = [(n_, p.detach().clone()) for n_, p in adapt]
            opt = torch.optim.AdamW([p for _, p in adapt], lr=args.ttt_lr)
            batch = tok([prompt + c for c in good], return_tensors="pt", padding=True,
                        truncation=True, max_length=768).to(dev)
            for _ in range(args.ttt_steps):
                opt.zero_grad()
                out = model(**batch, labels=batch["input_ids"])
                out.loss.backward(); opt.step()
            gt = EG.generate(model, tok, prompt, dev, mnt)
            s_ttt += EG.score("gsm8k", gt, golds)[0]
            with torch.no_grad():
                d = dict(snap)
                for n_, p in adapt: p.copy_(d[n_])
        else:
            s_ttt += EG.score("gsm8k", g0, golds)[0]

    print(f"[RESULT] static={s_static/N:.4f}  sc_vote={s_vote/N:.4f}  sc_ttt={s_ttt/N:.4f}  "
          f"(ttt-static={ (s_ttt-s_static)/N:+.4f}  ttt-vote={(s_ttt-s_vote)/N:+.4f})")


if __name__ == "__main__":
    main()
