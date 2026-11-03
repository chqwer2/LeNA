"""MetaMathQA fine-tuning with LeNA v7 or LoRA in the same code base, following PiSSA's LLaMA-2-7B protocol
(MuLabPKU/PiSSA scripts/metamath_llama2_7b/run_lora.sh): first 100k rows of fxmeng/pissa-dataset metamath/train.json,
Alpaca prompt, loss on the response only, max length 512, 1 epoch, lr 2e-5, cosine, warmup 0.03, weight decay 0,
batch 128, r = alpha = 128, dropout 0, targets q,k,v,o,gate,up,down, bf16.

  python finetune_math.py --adapter lena --output_dir runs/math_lena
"""
import argparse
import copy
import glob
import json
import os
import sys

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "commonsense"))
import lena_v7  # noqa: E402
from lena_v7 import inject, install_lena_hooks, adapter_tensors, trainable_count  # noqa: E402
from finetune_lena import AdapterTrainer  # noqa: E402

IGNORE = -100
PROMPT = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Response:")                # PiSSA train.py L21-24


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/pissa-dataset/metamath/train.json")
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adapter", choices=["lena", "lora", "aurora", "lena_aurora"], default="lena")
    p.add_argument("--mode", default="lena")
    p.add_argument("--r", type=int, default=128)
    p.add_argument("--alpha", type=float, default=128)
    p.add_argument("--knots", type=int, default=16)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint-* in output_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", help="torch.compile the LeNA fit path (same outputs)")
    p.add_argument("--triton", action="store_true", help="fused Triton kernels for the mode-lena fit (lena_triton.py)")
    p.add_argument("--omega_lr_mult", type=float, default=1.0, help="H18: lr factor for the per-layer Stein scale omega only (no decay)")
    p.add_argument("--neff", action="store_true", help="R1: Stein noise variance x design effect 1+(n-1)rho_tok (causal: prefix)")
    p.add_argument("--stein_log", action="store_true", help="log the mean Stein s/bend of all LeNA fits (no change)")
    p.add_argument("--inner_ckpt", action="store_true", help="checkpoint inside each LeNA layer as well (redundant under the model's gradient checkpointing; costs one extra forward of the fit)")
    a = p.parse_args()
    lena_v7.set_compile(a.compile)
    lena_v7.set_triton(a.triton)
    transformers.set_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base_model, model_max_length=a.max_len, padding_side="right", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    for prm in model.parameters():
        prm.requires_grad_(False)
    targets = a.target_modules.split(",")
    kw = dict(r=a.r, alpha=a.alpha, dropout=0.0)
    if a.adapter in ("lena", "lena_aurora"):
        kw.update(K=a.knots, mode=a.mode, causal=True, use_checkpoint=a.inner_ckpt)
    names = inject(model, targets, adapter=a.adapter, **kw)
    for n_, prm in model.named_parameters():
        if prm.requires_grad or any(t in n_ for t in (".rho", ".omega", ".A2")):
            prm.data = prm.data.float()
    if a.neff or a.stein_log:                                        # R1 runtime patch / Stein-s logging (lena_mri_ext)
        import atexit
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))
        import lena_mri_ext
        assert not a.triton, "--neff/--stein_log need the PyTorch fit path (no --triton)"
        lena_mri_ext.install(model, neff=a.neff)
        lena_mri_ext.EXT["acc"] = {}

        def _dump_s():
            acc = lena_mri_ext.EXT["acc"]
            st = {k: acc[k] / max(acc.get("n", 1), 1) for k in ("s", "bend") if k in acc}
            st["n_fits"] = acc.get("n", 0)
            print("STEIN_ALL", json.dumps(st), flush=True)
            json.dump(st, open(os.path.join(a.output_dir, "stein_all.json"), "w"))
        atexit.register(_dump_s)
    install_lena_hooks(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_tr = trainable_count(model)
    info = dict(vars(a), adapted_modules=len(names), trainable=n_tr, adapter_kwargs=kw, targets=targets,
                trainable_pct=100.0 * n_tr / (sum(q.numel() for q in model.parameters()) - n_tr))
    json.dump(dict(info, adapter=a.adapter, bf16=True), open(os.path.join(a.output_dir, "lena_config.json"), "w"),
              indent=1)
    print(f"adapted {len(names)} modules; trainable {n_tr} ({info['trainable_pct']:.3f}%)", flush=True)

    ds = load_dataset("json", data_files=a.data, split=f"train[:{a.n_train}]")   # file order, as sub_task :100000

    def tokenize(ex):
        src = PROMPT.format_map(dict(instruction=ex["instruction"]))
        full = src + f"{ex['output']}\n{tok.eos_token}"
        ids = tok(full, truncation=True, max_length=a.max_len)["input_ids"]
        n_src = len(tok(src, truncation=True, max_length=a.max_len)["input_ids"])
        lab = copy.deepcopy(ids); lab[:n_src] = [IGNORE] * min(n_src, len(lab))
        return dict(input_ids=ids, attention_mask=[1] * len(ids), labels=lab)

    train = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)
    TrainerCls = AdapterTrainer
    if a.omega_lr_mult != 1.0:                                       # H18: omega in its own AdamW group
        class TrainerCls(AdapterTrainer):
            def create_optimizer(self):
                if self.optimizer is None:
                    om = [q for n_, q in self.model.named_parameters() if q.requires_grad and n_.endswith(".omega")]
                    ids = {id(q) for q in om}
                    rest = [q for q in self.model.parameters() if q.requires_grad and id(q) not in ids]
                    self.optimizer = torch.optim.AdamW(
                        [dict(params=rest), dict(params=om, lr=self.args.learning_rate * a.omega_lr_mult, weight_decay=0.0)],
                        lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
                    print("PARAM_GROUPS", [dict(n=sum(q.numel() for q in g["params"]), lr=g["lr"]) for g in self.optimizer.param_groups], flush=True)
                return self.optimizer

            def _save(self, output_dir=None, state_dict=None):
                super()._save(output_dir, state_dict)
                t = torch.stack([q.detach().float() for n_, q in self.model.named_parameters() if n_.endswith(".omega")])
                print("OMEGA_STATS", json.dumps(dict(step=self.state.global_step, mean=float(t.mean()), min=float(t.min()),
                                                     max=float(t.max()), absmean=float(t.abs().mean()))), flush=True)
    trainer = TrainerCls(
        model=model, train_dataset=train,
        args=transformers.TrainingArguments(
            output_dir=a.output_dir, per_device_train_batch_size=a.micro_batch_size,
            gradient_accumulation_steps=a.batch_size // a.micro_batch_size, num_train_epochs=1, max_steps=a.max_steps,
            learning_rate=a.lr, weight_decay=0.0, warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True,
            logging_steps=5, save_strategy="steps", save_steps=100, save_total_limit=2, report_to="none",
            remove_unused_columns=False, seed=a.seed, optim="adamw_torch"),
        data_collator=transformers.DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE,
                                                          return_tensors="pt"),
    )
    model.config.use_cache = False
    ckpts = sorted(glob.glob(os.path.join(a.output_dir, "checkpoint-*")), key=lambda d: int(d.rsplit("-", 1)[1]))
    trainer.train(resume_from_checkpoint=ckpts[-1] if a.resume and ckpts else None)
    torch.save(adapter_tensors(model), os.path.join(a.output_dir, "adapter.pt"))
    json.dump(trainer.state.log_history, open(os.path.join(a.output_dir, "log_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
