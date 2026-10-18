"""TTT (test-time training) probe: per-sample adapt the low-rank adapter on the QUESTION
(self-supervised LM loss, NO answer -> no label leak), then generate. Reset per sample.

Tests the thesis: extreme-low-rank adapters are capacity-starved; instead of adding
parameters (which destabilized -> steer failed), add TEST-TIME COMPUTE.

  static (K=0) vs TTT (K>0) on the SAME adapter. TTT>static => "compute for params" works.
Usage: python ttt_eval.py --adapter <dir> --limit 50 --ttt_steps 10 --ttt_lr 1e-3
"""
import argparse, os, glob, copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import eval_generate as EG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--ttt_steps", type=int, default=10)
    ap.add_argument("--ttt_lr", type=float, default=1e-3)
    args = ap.parse_args()

    cache = os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device

    # adapter params to TTT-adapt (lora_A/B, act, gate, steer...) -> everything not the frozen base
    adapt = [(n, p) for n, p in model.named_parameters()
             if any(t in n for t in ("lora_", ".act.", ".gate.", ".steer.", "lena_"))]
    for _, p in adapt:
        p.requires_grad_(True)
    print(f"[ttt] adapter={os.path.basename(os.path.dirname(args.adapter.rstrip('/')))} "
          f"adapt_params={len(adapt)} K={args.ttt_steps} lr={args.ttt_lr} limit={args.limit}")

    cfg = EG.TASKS["gsm8k"]
    ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"], cache_dir=cache)
    n = min(args.limit, len(ds))

    def run(K):
        ems = []
        for i in range(n):
            prompt, golds = EG.prompt_and_gold("gsm8k", ds[i])
            if K > 0:
                snap = [(n_, p.detach().clone()) for n_, p in adapt]
                opt = torch.optim.AdamW([p for _, p in adapt], lr=args.ttt_lr)
                q = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(dev)
                for _ in range(K):
                    opt.zero_grad()
                    out = model(**q, labels=q["input_ids"])
                    out.loss.backward()
                    opt.step()
            with torch.no_grad():
                pred = EG.generate(model, tok, prompt, dev, cfg["max_new_tokens"])
            em, _ = EG.score("gsm8k", pred, golds); ems.append(em)
            if K > 0:  # restore
                with torch.no_grad():
                    d = dict(snap)
                    for n_, p in adapt: p.copy_(d[n_])
        return sum(ems) / max(n, 1)

    acc0 = run(0)
    print(f"[RESULT] static(K=0) acc={acc0:.4f}")
    accK = run(args.ttt_steps)
    print(f"[RESULT] TTT(K={args.ttt_steps}) acc={accK:.4f}  delta={accK-acc0:+.4f}")


if __name__ == "__main__":
    main()
