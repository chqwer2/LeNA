"""ICLR-substantive test: ISOLATED test-time training of the two-pathway adapter.

Global pathway (lora_Ag/Bg, on the causal-mean context) + base = FROZEN. Local pathway
(lora_A/lora_B, on the per-token deviation = 'individual') = the ONLY thing adapted, per
example, by K self-supervised gradient steps (LM loss on the prompt), then reset. This tests
whether adapting an ISOLATED low-rank branch improves accuracy WITHOUT corrupting the frozen
global (the diagnostic showed adapting the single tuned adapter corrupts; the claim here is
isolation makes per-example adaptation safe & useful).

Paired per example: score with K=0 (no adapt) AND K steps, so K=0 == the frozen two-path
baseline and K>0 == +test-time adaptation, on the exact same examples (low variance).

Requires LENA_TWOPATH=1 so both pathways are reconstructed at load.
Usage: python eval_ttt.py --adapter runs/mtc_2path_aurora/<sub> --K 5 --lr 1e-3 --limit 100
"""
import argparse, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_choice import SPECS, to_choice_example, score_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tasks", default="ybisk/piqa,allenai/social_i_qa,allenai/ai2_arc:ARC-Easy,allenai/openbookqa")
    args = ap.parse_args()
    cache = os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=cache)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None, cache_dir=cache)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device

    # ISOLATION: adapt ONLY the local pathway (lora_A / lora_B), freeze global (lora_Ag/Bg) + base.
    local, ag_bg = [], 0
    for n, p in model.named_parameters():
        is_local = ((".lora_A." in n or ".lora_B." in n) and "lora_Ag" not in n and "lora_Bg" not in n)
        if "lora_Ag" in n or "lora_Bg" in n: ag_bg += 1
        p.requires_grad_(is_local)
        if is_local:
            p.data = p.data.float()   # adapt in fp32 for stability
            local.append(p)
    print(f"[ttt] adapting {len(local)} local params, froze global pathway ({ag_bg} Ag/Bg tensors) + base",
          flush=True)
    snap = [p.detach().clone() for p in local]

    def want(tag):
        return any(tag == t.strip() or tag.startswith(t.strip()) for t in args.tasks.split(","))

    print(f"[ttt] K={args.K} lr={args.lr} limit={args.limit}", flush=True)
    grand0 = grandK = grandN = 0
    for dp, name in SPECS:
        tag = f"{dp}:{name}" if name else dp
        if not want(tag): continue
        ds = load_dataset(dp, name, trust_remote_code=True) if name else load_dataset(dp, trust_remote_code=True)
        split = "validation" if "validation" in ds else ("test" if "test" in ds else "train")
        data = ds[split]
        ok0 = okK = ntot = 0
        for ex in data:
            item = to_choice_example(dp, ex)
            if item is None: continue
            prompt, options, gold = item
            # restore frozen local state
            with torch.no_grad():
                for p, s in zip(local, snap): p.copy_(s)
            # K=0 score (frozen two-path baseline)
            model.eval()
            with torch.no_grad():
                sc0 = score_options(model, tok, prompt, options, dev)
            ok0 += int(int(max(range(len(sc0)), key=sc0.__getitem__)) == gold)
            # K steps self-supervised LM adaptation on the prompt (local branch only)
            opt = torch.optim.SGD(local, lr=args.lr)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(dev)
            for _ in range(args.K):
                opt.zero_grad()
                out = model(**ids, labels=ids["input_ids"])
                out.loss.backward(); opt.step()
            with torch.no_grad():
                scK = score_options(model, tok, prompt, options, dev)
            okK += int(int(max(range(len(scK)), key=scK.__getitem__)) == gold)
            ntot += 1
            if ntot >= args.limit: break
        a0, aK = ok0/max(ntot,1), okK/max(ntot,1)
        grand0 += ok0; grandK += okK; grandN += ntot
        print(f"{tag:38s} K0={a0:.4f} K{args.K}={aK:.4f} d={aK-a0:+.4f} (n={ntot})", flush=True)
    print(f"[ttt] OVERALL K0={grand0/max(grandN,1):.4f} K{args.K}={grandK/max(grandN,1):.4f} "
          f"d={(grandK-grand0)/max(grandN,1):+.4f} (n={grandN})")


if __name__ == "__main__":
    main()
