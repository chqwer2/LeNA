import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple

import torch
from datasets import load_dataset, DatasetDict
import datasets

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# If your fork exports FloraConfig from peft
from peft import FloraConfig


# -------------------------
# Cache dir (your snippet)
# -------------------------
preferred_dir = "/Users/haochen/Documents/hf_models"
fallback_dir = "/media/cbtil3/9feaf350-913e-4def-8114-f03573c04364"
cache_dir = preferred_dir if os.path.isdir(preferred_dir) else fallback_dir


# -------------------------
# Device helpers
# -------------------------
def pick_device(device_arg: str):
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def device_sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# -------------------------
# Verbose dataset download
# -------------------------
def enable_dataset_download_verbose():
    datasets.logging.set_verbosity_info()
    # datasets.logging.set_verbosity_debug()


# -------------------------
# Dataset -> "text" builders
# -------------------------
def _join_choices(choices: List[Tuple[str, str]]) -> str:
    # choices: list of (label, text)
    return "\n".join([f"{lab}. {txt}" for lab, txt in choices])


def format_example_to_text(data_path: str, ex: Dict[str, Any]) -> str:
    """
    Convert one dataset example into a single supervised text string:
    prompt + correct answer (so it becomes a causal LM training sample).
    """
    dp = data_path.lower().strip()

    # ---- google/boolq ----
    # fields: question, passage, answer (bool)
    if dp == "google/boolq":
        answer = "yes" if bool(ex["answer"]) else "no"
        return (
            "### Task: Answer the question based on the passage.\n\n"
            f"Passage:\n{ex['passage']}\n\n"
            f"Question: {ex['question']}\n\n"
            f"Answer: {answer}"
        )

    # ---- ybisk/piqa ----
    # fields: goal, sol1, sol2, label (0/1)
    if dp == "ybisk/piqa":
        label = int(ex["label"]) if ex.get("label", -1) is not None else -1
        solutions = [ex["sol1"], ex["sol2"]]
        correct = solutions[label] if label in (0, 1) else ""
        return (
            "### Task: Choose the best solution.\n\n"
            f"Goal: {ex['goal']}\n"
            f"A. {ex['sol1']}\n"
            f"B. {ex['sol2']}\n\n"
            f"Answer: {('A' if label == 0 else 'B') if label in (0, 1) else ''}\n"
            f"{correct}"
        )

    # ---- allenai/social_i_qa ----
    # fields: context, question, answerA, answerB, answerC, label ("1"/"2"/"3" or int)
    if dp == "allenai/social_i_qa":
        raw = ex.get("label", None)
        try:
            lab = int(raw) if raw is not None else -1
        except Exception:
            lab = -1
        options = {1: ex["answerA"], 2: ex["answerB"], 3: ex["answerC"]}
        correct = options.get(lab, "")
        letter = {1: "A", 2: "B", 3: "C"}.get(lab, "")
        return (
            "### Task: Choose the best answer.\n\n"
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"A. {ex['answerA']}\n"
            f"B. {ex['answerB']}\n"
            f"C. {ex['answerC']}\n\n"
            f"Answer: {letter}\n"
            f"{correct}"
        )

    # ---- Rowan/hellaswag ----
    # common fields: ctx, endings (list), label (0..3)
    # some variants: ctx_a, ctx_b, activity_label etc — we'll use what's present.
    if dp == "rowan/hellaswag":
        ctx = ex.get("ctx") or (
            (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
        )
        endings = ex.get("endings", [])
        label = int(ex["label"]) if ex.get("label", None) is not None else -1
        choices = [(chr(ord("A") + i), endings[i]) for i in range(len(endings))]
        correct = endings[label] if 0 <= label < len(endings) else ""
        return (
            "### Task: Pick the most plausible continuation.\n\n"
            f"Context: {ctx}\n\n"
            f"{_join_choices(choices)}\n\n"
            f"Answer: {chr(ord('A') + label) if 0 <= label < len(endings) else ''}\n"
            f"{correct}"
        )

    # ---- allenai/winogrande (winogrande_xl) ----
    # fields: sentence (with _), option1, option2, answer ("1"/"2")
    if dp == "allenai/winogrande":
        sentence = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        raw = ex.get("answer", None)
        try:
            ans = int(raw) if raw is not None else -1
        except Exception:
            ans = -1
        correct = opt1 if ans == 1 else (opt2 if ans == 2 else "")
        letter = "A" if ans == 1 else ("B" if ans == 2 else "")
        return (
            "### Task: Fill in the blank with the correct option.\n\n"
            f"Sentence: {sentence}\n"
            f"A. {opt1}\n"
            f"B. {opt2}\n\n"
            f"Answer: {letter}\n"
            f"{correct}"
        )

    # ---- allenai/ai2_arc (ARC-Easy) ----
    # fields: question: {stem, choices:[{label,text}...]}, answerKey
    if dp == "allenai/ai2_arc":
        q = ex["question"]
        stem = q["stem"]
        choices = [(c["label"], c["text"]) for c in q["choices"]]
        ans_key = ex.get("answerKey", "")
        # find matching choice text
        choice_dict = {lab: txt for lab, txt in choices}
        correct = choice_dict.get(ans_key, "")
        return (
            "### Task: Choose the correct answer.\n\n"
            f"Question: {stem}\n\n"
            f"{_join_choices(choices)}\n\n"
            f"Answer: {ans_key}\n"
            f"{correct}"
        )

    # ---- allenai/openbookqa (main) ----
    # fields: question_stem, choices:{label:[...], text:[...]}, answerKey
    if dp == "allenai/openbookqa":
        stem = ex.get("question_stem", "")
        choices_obj = ex.get("choices", {})
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        choices = list(zip(labels, texts))
        ans_key = ex.get("answerKey", "")
        choice_dict = {lab: txt for lab, txt in choices}
        correct = choice_dict.get(ans_key, "")
        return (
            "### Task: Choose the correct answer.\n\n"
            f"Question: {stem}\n\n"
            f"{_join_choices(choices)}\n\n"
            f"Answer: {ans_key}\n"
            f"{correct}"
        )

    # ---- fallback ----
    # If you pass a dataset that already has "text", use it
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"]

    # Otherwise just dump something readable
    return "### Example\n" + json.dumps(ex, ensure_ascii=False)


def add_text_column(dataset: DatasetDict, data_path: str) -> DatasetDict:
    """
    Ensure every split has a 'text' column.
    """
    def _mapper(ex):
        return {"text": format_example_to_text(data_path, ex)}

    out = {}
    for split_name, split_ds in dataset.items():
        out[split_name] = split_ds.map(_mapper, desc=f"Building text ({split_name})")
    return DatasetDict(out)


def normalize_splits(ds: DatasetDict) -> DatasetDict:
    """
    Make sure we have train and test.
    - If there's validation but no test, use validation as test.
    - If only train exists, we will split later.
    """
    if "train" not in ds:
        # pick the first split as train
        first = list(ds.keys())[0]
        ds = DatasetDict({"train": ds[first]})

    if "test" not in ds and "validation" in ds:
        ds = DatasetDict({**ds, "test": ds["validation"]})

    return ds


# -------------------------
# Fixed-length tokenization + split
# -------------------------
def tokenize_and_split(dataset: DatasetDict, data_path: str, tokenizer, cutoff_len: int, val_set_size: int):
    dataset = normalize_splits(dataset)

    # build "text" if missing
    if "text" not in dataset["train"].column_names:
        dataset = add_text_column(dataset, data_path=data_path)

    def tokenize_function(examples):
        out = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=cutoff_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    # remove all original columns (keep only tokenized fields)
    remove_cols = dataset["train"].column_names

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing",
    )

    # if dataset already has test split, keep it; otherwise split train
    if "test" not in tokenized:
        val_size = min(val_set_size, max(1, len(tokenized["train"]) // 10))
        split = tokenized["train"].train_test_split(test_size=val_size, seed=42)
        tokenized = DatasetDict({"train": split["train"], "test": split["test"]})

    return tokenized


# -------------------------
# Trainable params control
# -------------------------
def set_trainable_only_flora(model):
    """
    Freezes everything EXCEPT FLoRA-related params:
      - A/B
      - activation module params
      - gate params
    Works if your injected modules are FloraLinear/FloraConv1D with:
      A, B, act, gate_after_a, gate_after_b (ModuleDicts)
    """
    for p in model.parameters():
        p.requires_grad = False

    for _, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls not in ("FloraLinear", "FloraConv1D"):
            continue

        for attr in ("A", "B", "act", "gate_after_a", "gate_after_b", "flora_A", "flora_B", "flora_act"):
            d = getattr(mod, attr, None)
            if d is None:
                continue

            if isinstance(d, torch.nn.ModuleDict):
                for _, sub in d.items():
                    for p in sub.parameters(recurse=True):
                        p.requires_grad = True
            elif isinstance(d, torch.nn.Module):
                for p in d.parameters(recurse=True):
                    p.requires_grad = True

    return model


def count_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# -------------------------
# Optional forward timing (mean only)
# -------------------------
def mean_forward_ms(model, tokenizer, device, seq_len=128, iters=20, warmup=5):
    model.eval()

    enc = tokenizer("hello", return_tensors="pt")
    ids = enc["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if ids.shape[1] < seq_len:
        pad = torch.full((ids.shape[0], seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, :seq_len]

    ids = ids.to(device)
    attn = (ids != pad_id).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=ids, attention_mask=attn)
        device_sync(device)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(input_ids=ids, attention_mask=attn)
            device_sync(device)
            times.append(time.perf_counter() - t0)

    return 1000.0 * (sum(times) / len(times))


# -------------------------
# Build PEFT config
# -------------------------
def build_peft_config(
    method: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    flora_activation: str,
    flora_flex_mode: str,
    flora_activation_kwargs: Dict,
    flora_gate_type: str,
    flora_gate_position: str,
    flora_debug: bool = False,
):
    method = method.lower()
    if method in ("lora", "dora"):
        return LoraConfig(
            use_dora=(method == "dora"),
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

    if method == "flora":
        return FloraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            flora_activation=flora_activation,
            flora_flex_mode=flora_flex_mode,
            flora_activation_kwargs=flora_activation_kwargs,
            flora_gate_type=flora_gate_type,
            flora_gate_position=flora_gate_position,
            flora_debug=flora_debug,
            flora_debug_verbose=False,
            flora_debug_forward=False,
            flora_debug_check_nan=True,
        )

    raise ValueError(f"Unknown method: {method}")


from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def build_grouped_optimizer(
    model,
    base_lr: float,
    act_lr_mult: float = 0.1,     # activation LR = base_lr * act_lr_mult
    gate_lr_mult: float = 0.5,    # gate LR = base_lr * gate_lr_mult
    weight_decay: float = 0.01,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
):
    # Names that should NOT use weight decay (LayerNorm, biases)
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [n for n in decay_parameters if not n.endswith(".bias")]

    # Helper: classify params by name
    def is_act(n: str) -> bool:
        # covers: FloraLinear.act.<key>.<param>, Flex* params, etc.
        return (".act." in n) or ("flora_act" in n) or (n.endswith(".knots_y")) or (n.endswith(".c"))

    def is_gate(n: str) -> bool:
        return (".gate_after_" in n) or ("flora_gate" in n) or ("gate_after" in n)

    def is_adapter(n: str) -> bool:
        # LoRA/DoRA names vary; for your FloraLinear it might be A/B or flora_A/flora_B
        return (".A." in n) or (".B." in n) or ("flora_A" in n) or ("flora_B" in n) or ("lora_" in n)

    # Collect trainable params
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    groups = {
        "adapter_decay": [],
        "adapter_nodecay": [],
        "act_nodecay": [],
        "gate_nodecay": [],
        "other_decay": [],
        "other_nodecay": [],
    }

    for n, p in named:
        use_decay = n in decay_parameters

        if is_act(n):
            groups["act_nodecay"].append(p)     # usually no WD for activation shaping
        elif is_gate(n):
            groups["gate_nodecay"].append(p)    # no WD for gates
        elif is_adapter(n):
            (groups["adapter_decay"] if use_decay else groups["adapter_nodecay"]).append(p)
        else:
            (groups["other_decay"] if use_decay else groups["other_nodecay"]).append(p)

    # Build param_groups (skip empty)
    param_groups = []

    def add(ps, lr, wd):
        if len(ps) > 0:
            param_groups.append({"params": ps, "lr": lr, "weight_decay": wd})

    add(groups["adapter_decay"],     base_lr, weight_decay)
    add(groups["adapter_nodecay"],   base_lr, 0.0)

    add(groups["act_nodecay"],       base_lr * act_lr_mult, 0.0)
    add(groups["gate_nodecay"],      base_lr * gate_lr_mult, 0.0)

    # If you truly only train flora params, these "other" groups will be empty.
    add(groups["other_decay"],       base_lr, weight_decay)
    add(groups["other_nodecay"],     base_lr, 0.0)

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    return optimizer

import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: [batch, seq, vocab]
    # labels: [batch, seq]

    # next-token prediction: compare logits at t-1 to label at t
    preds = np.argmax(logits, axis=-1)

    # shift so we're evaluating next-token accuracy
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]

    # ignore -100 labels (if any)
    mask = shift_labels != -100
    if mask.sum() == 0:
        return {"token_acc": 0.0}

    correct = (shift_preds == shift_labels) & mask
    token_acc = correct.sum() / mask.sum()

    return {"token_acc": float(token_acc)}


# -------------------------
# Main training function (single run)
# -------------------------
def train_one_run(
    *,
    base_model: str,
    data_path: str,
    data_name: Optional[str],
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    quantize: bool,
    eval_step: int,
    save_step: int,
    device_arg: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[str],
    method: str,

    flora_activation: str,
    flora_flex_mode: str,
    flora_activation_kwargs_json: str,
    flora_gate_type: str,
    flora_gate_position: str,

    only_train_flora_params: bool,
    print_forward_mean_ms: bool,

    push_to_hub: bool,
    hub_model_id: str,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    device = pick_device(device_arg)
    print(f"\n=== RUN method={method} device={device} ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if quantize:
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or getattr(torch, "xpu", None) is not None:
            bnb_4bit_compute_dtype = torch.bfloat16
        else:
            bnb_4bit_compute_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            cache_dir=cache_dir,
        )

    model.to(device)
    model.config.use_cache = False


    # Target modules
    target_modules = (
        [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        if lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Parse flora_activation_kwargs
    flora_activation_kwargs = {}
    if flora_activation_kwargs_json and flora_activation_kwargs_json.strip():
        flora_activation_kwargs = json.loads(flora_activation_kwargs_json)

    # Build PEFT config
    peft_cfg = build_peft_config(
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        flora_activation=flora_activation,
        flora_flex_mode=flora_flex_mode,
        flora_activation_kwargs=flora_activation_kwargs,
        flora_gate_type=flora_gate_type,
        flora_gate_position=flora_gate_position,
        flora_debug=False,
    )

    # Wrap model
    model = get_peft_model(model, peft_cfg)
    model.to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Num trainable params:", len(trainable))
    print("First few trainables:", trainable[:20])

    # Lazy init activations: one forward with fixed length
    with torch.no_grad():
        enc = tokenizer("hello", return_tensors="pt")
        ids = enc["input_ids"]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if ids.shape[1] < cutoff_len:
            pad = torch.full((ids.shape[0], cutoff_len - ids.shape[1]), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        else:
            ids = ids[:, :cutoff_len]
        ids = ids.to(device)
        attn = (ids != pad_id).to(device)
        _ = model(input_ids=ids, attention_mask=attn)

    # Freeze everything except flora params if requested
    if only_train_flora_params and method.lower() == "flora":
        set_trainable_only_flora(model)

    total, trainable = count_trainable_params(model)
    print(f"Params: trainable={trainable:,} / total={total:,} ({100*trainable/total:.4f}%)")

    if print_forward_mean_ms:
        ms = mean_forward_ms(model, tokenizer, device=device, seq_len=min(128, cutoff_len), iters=20, warmup=5)
        print(f"Mean forward latency: {ms:.2f} ms (seq_len={min(128, cutoff_len)})")

    # Dataset (verbose)
    enable_dataset_download_verbose()

    # Many of these datasets require a config name (e.g., winogrande_xl, ARC-Easy, main).
    if data_name and data_name.strip():
        dataset = load_dataset(data_path, data_name.strip(), cache_dir=cache_dir)
    else:
        dataset = load_dataset(data_path, cache_dir=cache_dir)

    tokenized = tokenize_and_split(dataset, data_path=data_path, tokenizer=tokenizer, cutoff_len=cutoff_len, val_set_size=val_set_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        gradient_accumulation_steps=16,
        fp16=(device.type == "cuda"),
        bf16=False, #(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        learning_rate=learning_rate,
        remove_unused_columns=False,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
        hub_token=os.getenv("HF_TOKEN") if push_to_hub else None,
        eval_steps=eval_step,
        report_to="none",
    )

    optimizer = build_grouped_optimizer(
        model,
        base_lr=training_args.learning_rate,
        act_lr_mult=0.1,  # start conservative
        gate_lr_mult=0.5,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    DEBUG = False
    if DEBUG:
        tr_n = min(20, len(tokenized["train"]))
        ev_n = min(20, len(tokenized["test"]))

        train_dataset = tokenized["train"].select(range(tr_n))
        eval_dataset = tokenized["test"].select(range(ev_n))

    else:
        train_dataset = tokenized["train"]
        eval_dataset = tokenized["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,  # <-- add this
    )

    trainer.train()

    # -------------------------
    # Final evaluation on full test set
    # -------------------------
    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="test")
    print("\n=== Final test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Optional: persist metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        trainer.push_to_hub(commit_message=f"Fine-tuned with {method}")

    return output_dir



def build_optimizer_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    *,
    act_lr_mult: float = 0.1,     # activation params slower
    gate_lr_mult: float = 0.5,    # gates slower (often)
    act_weight_decay: float = 0.0,
    gate_weight_decay: float = 0.0,
    adapter_weight_decay: float = 0.01,
):
    """
    Creates optimizer param groups for:
      - adapters (A/B, lora_*): base_lr
      - gates: base_lr * gate_lr_mult
      - activations: base_lr * act_lr_mult
    """
    act_params, gate_params, adapter_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n = name.lower()

        # --- activation parameters ---
        # matches your Flex* modules: a, k, (fourier) a/w/p, (spline) knots_y, (poly) c
        if any(tok in n for tok in [
            ".act.", "flora_act", "knots_y",  # module containers / spline
        ]) or n.endswith((".a", ".k", ".w", ".p", ".c")):
            act_params.append(p)
            continue

        # --- gate parameters ---
        if "gate_after_a" in n or "gate_after_b" in n or ".gate." in n:
            gate_params.append(p)
            continue

        # --- adapter parameters (A/B or lora) ---
        if any(tok in n for tok in ["flora_a", "flora_b", ".a.", ".b.", "lora_", "lora"]):
            adapter_params.append(p)
            continue

        # fallback: if it is trainable but doesn't match, treat as adapter
        adapter_params.append(p)

    param_groups = []
    if adapter_params:
        param_groups.append(
            {"params": adapter_params, "lr": base_lr, "weight_decay": adapter_weight_decay}
        )
    if gate_params:
        param_groups.append(
            {"params": gate_params, "lr": base_lr * gate_lr_mult, "weight_decay": gate_weight_decay}
        )
    if act_params:
        param_groups.append(
            {"params": act_params, "lr": base_lr * act_lr_mult, "weight_decay": act_weight_decay}
        )

    # (optional) debug print sizes
    def _count(ps): return sum(p.numel() for p in ps)
    print(f"[opt groups] adapter={_count(adapter_params):,} gate={_count(gate_params):,} act={_count(act_params):,}")

    return param_groups



# -------------------------
# Multi-run driver: compare lora/dora/flora variants
# -------------------------
def run_experiments(
    base_model: str,
    data_path: str,
    data_name: Optional[str],
    output_dir: str,
    methods: str,
    flora_activations: str,
    flora_flex_mode: str,
    flora_activation_kwargs_json: str,
    flora_gate_type: str,
    flora_gate_position: str,
    **kwargs,
):
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    flora_act_list = [a.strip() for a in flora_activations.split(",") if a.strip()]

    runs = []

    for m in method_list:
        if m.lower() in ("lora", "dora"):
            out = os.path.join(output_dir, m.lower())
            train_one_run(
                base_model=base_model,
                data_path=data_path,
                data_name=data_name,
                output_dir=out,
                method=m.lower(),
                flora_activation="identity",
                flora_flex_mode=flora_flex_mode,
                flora_activation_kwargs_json=flora_activation_kwargs_json,
                flora_gate_type="none",
                flora_gate_position="after_b",
                **kwargs,
            )
            runs.append(out)

        elif m.lower() == "flora":
            for act in flora_act_list:
                tag = f"flora_{act}_{flora_flex_mode}_{flora_gate_type}_{flora_gate_position}"
                out = os.path.join(output_dir, tag)

                train_one_run(
                    base_model=base_model,
                    data_path=data_path,
                    data_name=data_name,
                    output_dir=out,
                    method="flora",
                    flora_activation=act,
                    flora_flex_mode=flora_flex_mode,
                    flora_activation_kwargs_json=flora_activation_kwargs_json,
                    flora_gate_type=flora_gate_type,
                    flora_gate_position=flora_gate_position,
                    **kwargs,
                )
                runs.append(out)

        else:
            raise ValueError(f"Unknown method in methods list: {m}")

    print("\n=== Finished runs ===")
    for r in runs:
        print("-", r)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LoRA vs DoRA vs FLoRA (with activation variants)")

    # IMPORTANT: your original default had "a | b" which is not a valid model id.
    parser.add_argument("--base_model", type=str, default="sshleifer/tiny-gpt2")

    # Use --data_path and optionally --data_name for configs like ARC-Easy, winogrande_xl, main, etc.
    parser.add_argument("--data_path", type=str, default="google/boolq")
    parser.add_argument("--data_name", type=str, default="", help="Dataset config name, e.g. winogrande_xl, ARC-Easy, main")

    parser.add_argument("--output_dir", type=str, default="./outputs_compare")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--val_set_size", type=int, default=500)
    parser.add_argument("--quantize", action="store_true")

    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--methods", type=str, default="lora,dora,flora",
                        help="Comma-separated: lora,dora,flora")

    parser.add_argument("--flora_activations", type=str, default="identity,gelu,fourier",
                        help="Comma-separated: identity,relu,gelu,fourier,spline,polynomial")
    parser.add_argument("--flora_flex_mode", type=str, default="channel",
                        help="global|spatial|channel|voxel")
    parser.add_argument("--flora_activation_kwargs_json", type=str, default="",
                        help='JSON string, e.g. {"n_terms":4,"max_h":512,"max_w":1}.')
    parser.add_argument("--flora_gate_type", type=str, default="none",
                        help="none|sigmoid|tanh|rezero")
    parser.add_argument("--flora_gate_position", type=str, default="after_b",
                        help="after_a|after_b|both")

    parser.add_argument("--only_train_flora_params", action="store_true")
    parser.add_argument("--print_forward_mean_ms", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="path/to/repo")

    args = parser.parse_args()

    run_experiments(
        base_model=args.base_model,
        data_path=args.data_path,
        data_name=args.data_name.strip() or None,
        output_dir=args.output_dir,
        methods=args.methods,
        flora_activations=args.flora_activations,
        flora_flex_mode=args.flora_flex_mode,
        flora_activation_kwargs_json=args.flora_activation_kwargs_json,
        flora_gate_type=args.flora_gate_type,
        flora_gate_position=args.flora_gate_position,

        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device_arg=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        only_train_flora_params=args.only_train_flora_params,
        print_forward_mean_ms=args.print_forward_mean_ms,
    )
