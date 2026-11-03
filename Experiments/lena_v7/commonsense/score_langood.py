"""Candidate log-likelihood scoring (diagnose_lena.Scorer, condition 'learned', no override) on the language-OOD
sets. Same scorer for LeNA and LoRA. Writes per-example predictions for paired bootstrap.

  python score_langood.py --base_model meta-llama/Meta-Llama-3-8B --run_dir runs/llama3_lena --datasets commonsenseqa,mmlu
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, install_lena_hooks  # noqa: E402
from finetune_lena import load_tokenizer  # noqa: E402
from evaluate_lena import generate_prompt  # noqa: E402
from diagnose_lena import Scorer  # noqa: E402


def cands_of(instruction):
    m = re.search(r"Answer format: (answer\d(?:/answer\d)*)", instruction)
    return m.group(1).split("/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--adapter_file", default="adapter.pt"); p.add_argument("--data_root", default="dataset")
    p.add_argument("--datasets", default="commonsenseqa,mmlu"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_name", default="langood_score")
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.run_dir, "lena_config.json")))
    tok = load_tokenizer(args.base_model); tok.padding_side = "right"
    dtype = torch.bfloat16 if cfg.get("bf16") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map={"": 0})
    kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None)
    inject(model, cfg["targets"], adapter=cfg["adapter"], **kw)
    sd = torch.load(os.path.join(args.run_dir, args.adapter_file), map_location="cpu")
    missing = [k for k in sd if k not in model.state_dict()]
    assert not missing, missing[:5]
    for n_, prm in model.named_parameters():
        if n_ in sd:
            prm.data = sd[n_].to(prm.device, torch.float32)
    model.eval(); install_lena_hooks(model)
    S = Scorer(model, tok)
    out_dir = os.path.join(args.run_dir, args.out_name); os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for ds in args.datasets.split(","):
        data = json.load(open(os.path.join(args.data_root, ds, "test.json")))
        if args.limit:
            data = data[:args.limit]
        t0, correct, recs = time.time(), 0, []
        for k, ex in enumerate(data):
            cands = cands_of(ex["instruction"])
            pi, tot = S.score(generate_prompt(ex["instruction"]), cands)
            pred = cands[pi]; correct += int(pred == ex["answer"])
            recs.append(dict(answer=ex["answer"], pred=pred, logp=tot))
            if k % 100 == 0:
                print(f"{ds} {k}/{len(data)} acc so far {correct / (k + 1):.3f} [{time.time() - t0:.0f}s]", flush=True)
        summary[ds] = dict(acc=correct / len(data), n=len(data), seconds=time.time() - t0, adapter=cfg["adapter"])
        json.dump(recs, open(os.path.join(out_dir, f"{ds}.json"), "w"))
        print(f"{ds}: {100 * correct / len(data):.1f} ({correct}/{len(data)})", flush=True)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
