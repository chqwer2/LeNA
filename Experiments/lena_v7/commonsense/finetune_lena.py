"""Commonsense fine-tuning with LeNA v7 (or the LoRA / LoRA-dagger / DoRA baselines in the same code base).

Protocol of LLM-Adapters / DoRA (commonsense_reasoning/finetune.py): commonsense_170k.json, Alpaca-style prompt,
cutoff 256, val 120, 3 epochs, warmup 100, AdamW, targets q,k,v,up,down. Only adapter tensors are saved.

Example (LLaMA3-8B, rank 32):
  python finetune_lena.py --base_model meta-llama/Meta-Llama-3-8B --data_path commonsense_170k.json \
      --output_dir runs/llama3_lena_r32 --adapter lena --lora_r 32 --lora_alpha 64 --learning_rate 1e-4
"""
import argparse
import json
import math
import os
import sys

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import inject, install_lena_hooks, adapter_tensors, trainable_count  # noqa: E402


# copied verbatim from DoRA commonsense_reasoning/finetune.py
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


def adapter_kwargs(args):
    if args.adapter == "lora":
        return dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, dora=args.dora)
    return dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, K=args.knots, lo=args.grid_lo,
                hi=args.grid_hi, tau=args.tau, mode=args.mode, prior=args.prior, learn_omega=not args.js_only,
                freeze_A2=args.freeze_A2, dora=args.dora, causal=True, use_checkpoint=True)


def load_tokenizer(base_model):
    if "llama-3" in base_model.lower():
        tok = AutoTokenizer.from_pretrained(base_model)
    else:
        try:                                                     # the protocol uses the slow tokenizer for LLaMA 1/2
            tok = LlamaTokenizer.from_pretrained(base_model)
        except ImportError:                                      # no sentencepiece (local smoke tests)
            tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token_id = 0
    tok.padding_side = "left"
    return tok


class AdapterTrainer(transformers.Trainer):
    """Saves only adapter tensors; keeps the adapter with the best validation loss."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(adapter_tensors(self.model), os.path.join(output_dir, "adapter.pt"))

    def evaluate(self, *a, **kw):
        metrics = super().evaluate(*a, **kw)
        loss = metrics.get("eval_loss")
        if loss is not None and loss < getattr(self, "_best", math.inf):
            self._best = loss
            torch.save(adapter_tensors(self.model), os.path.join(self.args.output_dir, "best_adapter.pt"))
        return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--data_path", default="commonsense_170k.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adapter", choices=["lena", "lora"], default="lena")
    p.add_argument("--mode", choices=["lena", "fixed", "plain", "predictor"], default="lena")
    p.add_argument("--prior", choices=["learned", "identity", "u_pred"], default="learned")
    p.add_argument("--js_only", action="store_true", help="omega fixed at 0 (James-Stein scale)")
    p.add_argument("--freeze_A2", action="store_true")
    p.add_argument("--dora", action="store_true", help="LeNA-D, or DoRA with --adapter lora")
    p.add_argument("--knots", type=int, default=16)
    p.add_argument("--grid_lo", type=float, default=-3.0)
    p.add_argument("--grid_hi", type=float, default=3.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--cutoff_len", type=int, default=256)
    p.add_argument("--val_set_size", type=int, default=120)
    p.add_argument("--eval_step", type=int, default=80)
    p.add_argument("--save_step", type=int, default=80)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true", help="bf16 instead of the protocol's fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1, help="smoke tests")
    args = p.parse_args()

    transformers.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if args.bf16 else torch.float16) if cuda else torch.float32   # CPU: smoke tests only
    tok = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)} if cuda else {"": "cpu"})
    for prm in model.parameters():
        prm.requires_grad_(False)
    targets = args.target_modules.split(",")
    names = inject(model, targets, adapter=args.adapter, **adapter_kwargs(args))
    # adapter tensors in fp32 (statistics are fp32 anyway); the frozen base stays in fp16/bf16
    for n_, prm in model.named_parameters():
        if prm.requires_grad or any(t in n_ for t in (".rho", ".omega", ".A2")):
            prm.data = prm.data.float()
    install_lena_hooks(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_train, n_all = trainable_count(model), sum(q.numel() for q in model.parameters())
    info = dict(vars(args), adapted_modules=len(names), trainable=n_train, total=n_all,
                trainable_pct=100.0 * n_train / (n_all - n_train), adapter_kwargs=adapter_kwargs(args), targets=targets)
    json.dump(info, open(os.path.join(args.output_dir, "lena_config.json"), "w"), indent=1)
    print(f"adapted {len(names)} modules; trainable {n_train} ({info['trainable_pct']:.3f}%)")

    def tokenize(prompt, add_eos_token=True):
        r = tok(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if r["input_ids"][-1] != tok.eos_token_id and len(r["input_ids"]) < args.cutoff_len and add_eos_token:
            r["input_ids"].append(tok.eos_token_id)
            r["attention_mask"].append(1)
        r["labels"] = r["input_ids"].copy()
        return r

    data = load_dataset("json", data_files=args.data_path)
    fn = lambda d: tokenize(generate_prompt(d))  # noqa: E731  (train_on_inputs=True, as in the protocol)
    cols = data["train"].column_names
    if args.val_set_size > 0:
        split = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
        train_data = split["train"].shuffle().map(fn, remove_columns=cols)
        val_data = split["test"].shuffle().map(fn, remove_columns=cols)
    else:
        train_data, val_data = data["train"].shuffle().map(fn, remove_columns=cols), None

    trainer = AdapterTrainer(
        model=model, train_dataset=train_data, eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            warmup_steps=args.warmup_steps, num_train_epochs=args.num_epochs, max_steps=args.max_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            fp16=cuda and not args.bf16, bf16=cuda and args.bf16, logging_steps=10, optim="adamw_torch",
            eval_strategy="steps" if val_data is not None else "no",
            eval_steps=args.eval_step if val_data is not None else None,
            save_strategy="steps", save_steps=args.save_step, save_total_limit=3,
            output_dir=args.output_dir, report_to="none", remove_unused_columns=False, seed=args.seed),
        data_collator=transformers.DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True),
    )
    model.config.use_cache = False
    trainer.train()
    best = os.path.join(args.output_dir, "best_adapter.pt")
    if os.path.exists(best):
        model.load_state_dict(torch.load(best), strict=False)
    torch.save(adapter_tensors(model), os.path.join(args.output_dir, "adapter.pt"))
    json.dump(trainer.state.log_history, open(os.path.join(args.output_dir, "log_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
