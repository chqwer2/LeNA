"""DIAGNOSTIC: does per-sample TTT corrupt the GLOBAL adapter?
Adapt on ONE question X (its self-consistency consensus chains), then measure greedy acc on
40 UNRELATED held-out questions Y (+ the adapter weight-change norm). Sweep lr.
If acc(Y) drops as lr grows (with no lr that avoids it), per-sample adaptation of the single
tuned adapter is the root cause of the instability -> motivates an ISOLATED per-sample branch.
Usage: python ttt_diag.py --adapter <dir>
"""
import argparse, os, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import eval_generate as EG


def sample_chains(model, tok, prompt, dev, mn, n, temp):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=mn, do_sample=True, temperature=temp,
                             num_return_sequences=n, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--n_y", type=int, default=40)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    cache = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter); model.eval()
    dev = next(model.parameters()).device
    adapt = [(n, p) for n, p in model.named_parameters()
             if any(t in n for t in ("lora_", ".act.", ".gate.", "lena_"))]
    for _, p in adapt: p.requires_grad_(True)
    cfg = EG.TASKS["gsm8k"]; mn = cfg["max_new_tokens"]
    ds = load_dataset(cfg["path"], cfg["config"], split="test", cache_dir=cache)

    def accY():
        c = 0
        for i in range(args.n_y):
            p, g = EG.prompt_and_gold("gsm8k", ds[i])
            c += EG.score("gsm8k", EG.generate(model, tok, p, dev, mn), g)[0]
        return c / args.n_y

    base = accY()
    print(f"[diag] adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} baseline acc(Y)={base:.4f}")
    # X = a question NOT in Y; build its consensus chains once
    xp = EG.prompt_and_gold("gsm8k", ds[args.n_y + 5])[0]
    chains = sample_chains(model, tok, xp, dev, mn, 8, 0.8)
    val = [a for a in (EG.last_number(c) for c in chains) if a is not None]
    maj = Counter(val).most_common(1)[0][0] if val else None
    good = [xp + c for c, a in zip(chains, (EG.last_number(c) for c in chains)) if a == maj] or [xp + chains[0]]

    snap = [(n, p.detach().clone()) for n, p in adapt]
    for lr in [1e-5, 1e-4, 1e-3]:
        # restore
        with torch.no_grad():
            d = dict(snap)
            for n, p in adapt: p.copy_(d[n])
        opt = torch.optim.AdamW([p for _, p in adapt], lr=lr)
        model.train()
        batch = tok(good, return_tensors="pt", padding=True, truncation=True, max_length=768).to(dev)
        for _ in range(args.K):
            opt.zero_grad(); out = model(**batch, labels=batch["input_ids"]); out.loss.backward(); opt.step()
        model.eval()
        dnorm = sum((p.detach() - dict(snap)[n]).norm().item() ** 2 for n, p in adapt) ** 0.5
        aY = accY()
        print(f"[diag] lr={lr:.0e} K={args.K}: acc(Y)={aY:.4f}  drop={base-aY:+.4f}  ||dTheta||={dnorm:.3f}")
    print("[diag] DONE")


if __name__ == "__main__":
    main()
