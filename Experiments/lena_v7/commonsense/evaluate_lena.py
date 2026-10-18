"""Commonsense evaluation for LeNA v7 / LoRA / DoRA adapters saved by finetune_lena.py.

Same protocol as LLM-Adapters / DoRA commonsense_evaluate.py: dataset/<name>/test.json, the Alpaca prompt, beam
search (4 beams, temperature 0.1, top_p 0.75, top_k 40, 32 new tokens), first matching answer string. LeNA's causal
prefix statistics are carried through the KV cache and reordered with the beams (lena_stream + install_lena_hooks).

  python evaluate_lena.py --base_model meta-llama/Meta-Llama-3-8B --run_dir runs/llama3_lena_r32 \
      --datasets boolq,piqa,social_i_qa,hellaswag,winogrande,ARC-Challenge,ARC-Easy,openbookqa
"""
import argparse
import contextlib
import json
import os
import re
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import inject, install_lena_hooks, lena_stream  # noqa: E402
from finetune_lena import load_tokenizer  # noqa: E402

ALL = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
PATTERNS = {"boolq": r"true|false", "piqa": r"solution1|solution2", "hellaswag": r"ending1|ending2|ending3|ending4",
            "winogrande": r"option1|option2"}
for _d in ("social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"):
    PATTERNS[_d] = r"answer1|answer2|answer3|answer4|answer5"


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def extract_answer(dataset, sentence):
    found = re.findall(PATTERNS[dataset], sentence.strip())
    return found[0] if found else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--adapter_file", default="adapter.pt")
    p.add_argument("--datasets", default=",".join(ALL))
    p.add_argument("--data_root", default="dataset")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="first n examples (smoke tests)")
    p.add_argument("--no_cache", action="store_true", help="recompute every prefix (check of the streaming path)")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "lena_config.json")))
    tok = load_tokenizer(args.base_model)
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if cfg.get("bf16") else torch.float16) if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype,
                                                 device_map={"": 0} if cuda else {"": "cpu"})
    kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None)
    inject(model, cfg["targets"], adapter=cfg["adapter"], **kw)
    sd = torch.load(os.path.join(args.run_dir, args.adapter_file), map_location="cpu")
    missing = [k for k in sd if k not in model.state_dict()]
    assert not missing, missing[:5]
    for n_, prm in model.named_parameters():
        if n_ in sd:
            prm.data = sd[n_].to(prm.device, torch.float32)
    model.eval()
    install_lena_hooks(model)
    gen = GenerationConfig(temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=32,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                           use_cache=not args.no_cache)

    out_dir = os.path.join(args.run_dir, "eval"); os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for ds in args.datasets.split(","):
        data = json.load(open(os.path.join(args.data_root, ds, "test.json")))
        if args.limit:
            data = data[:args.limit]
        correct, records, t0 = 0, [], time.time()
        for i in tqdm(range(0, len(data), args.batch_size), desc=ds):
            batch = data[i:i + args.batch_size]
            enc = tok([generate_prompt(d["instruction"]) for d in batch], return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            stream = contextlib.nullcontext() if args.no_cache else lena_stream(model)   # state only with a cache
            with torch.no_grad(), stream:
                seq = model.generate(**enc, generation_config=gen)
            outs = [o.split("### Response:")[-1].strip() for o in tok.batch_decode(seq, skip_special_tokens=True)]
            for d, o in zip(batch, outs):
                pred = extract_answer(ds, o)
                correct += int(pred == d["answer"])
                records.append(dict(d, output_pred=o, pred=pred))
        acc = correct / len(data)
        summary[ds] = dict(acc=acc, n=len(data), seconds=time.time() - t0)
        json.dump(records, open(os.path.join(out_dir, f"{ds}.json"), "w"), indent=1)
        print(f"{ds}: {100 * acc:.1f} ({correct}/{len(data)})")
    if len(summary) == len(ALL):
        summary["avg"] = sum(v["acc"] for v in summary.values()) / len(ALL)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: (round(100 * v["acc"], 1) if isinstance(v, dict) else round(100 * v, 1))
                      for k, v in summary.items()}))


if __name__ == "__main__":
    main()
