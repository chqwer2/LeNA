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
    TrainerCallback,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# If your fork exports LeNAConfig from peft
from peft import LeNAConfig
from peft.tuners.lena.layer import lena_gate_l1


# -------------------------
# Cache dir (your snippet)
# -------------------------
preferred_dir = "/Users/haochen/Documents/hf_models"
fallback_dir  = "/media/cbtil3/9feaf350-913e-4def-8114-f03573c04364"
fallback_dir2 = "/root/autodl-tmp/hf_home/hub"


print("os.path.isdir(fallback_dir2) = ", os.path.isdir(fallback_dir2))

# Prefer the HF_HOME cache if set (e.g. on Kelvin2: /mnt/scratch2/.../hf_home).
_hf_home = os.environ.get("HF_HOME")
if _hf_home:
    cache_dir = os.path.join(_hf_home, "hub")
elif os.path.isdir(preferred_dir):
    cache_dir = preferred_dir
elif os.path.isdir(fallback_dir2):
    cache_dir = fallback_dir2
elif os.path.isdir(fallback_dir):
    cache_dir = fallback_dir
else:
    cache_dir = fallback_dir2
    print("[WARN] No preferred cache dir found; using fallback:", cache_dir)


print("cache_dir =", cache_dir)
# Quick smoke mode: slice datasets to a few dozen samples (set LENA_DEBUG=1).
DEBUG = os.environ.get("LENA_DEBUG", "0") == "1"


class LeNAGradProbe(TrainerCallback):
    """Log the gradient norm of each LeNA parameter family separately.

    A single global grad_norm cannot say whether the noise lives in A/B, in the
    activation, or in the straight-through gate; this splits it out. Enable with
    LENA_GRAD_PROBE=1.
    """

    FAMILIES = (("A", "lora_A"), ("B", "lora_B"), ("act", ".act."),
                ("gate", ".gate."), ("mag", ".magnitude."))

    def __init__(self, every: int = 20):
        self.every = int(every)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every:
            return
        sums = {label: 0.0 for label, _ in self.FAMILIES}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float().pow(2).sum().item()
            for label, pat in self.FAMILIES:
                if pat in n:
                    sums[label] += g
                    break
        parts = " ".join(f"{k}={v ** 0.5:.4f}" for k, v in sums.items() if v > 0)
        print(f"[grad step={state.global_step}] {parts}", flush=True)


class LeNAGateL1Trainer(Trainer):
    """Trainer that adds `coef * mean_gate_openness` to the loss.

    This is the "learn where" regularizer: it pushes the selection gates closed
    (linear/LoRA) unless a location's nonlinearity earns its keep, yielding a sparse
    set of nonlinear locations (report it as "only X% of locations go nonlinear").
    """
    def __init__(self, *args, lena_gate_l1_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lena_gate_l1_coef = float(lena_gate_l1_coef)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.lena_gate_l1_coef > 0:
            pen = lena_gate_l1(model)
            if pen is not None:
                loss = loss + self.lena_gate_l1_coef * pen
        return (loss, outputs) if return_outputs else loss


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
    if dp in ("piqa", "ybisk/piqa"):
        raw = ex.get("label", None)

        # robust parse
        try:
            if raw is None:
                label = -1
            elif isinstance(raw, int):
                label = raw
            elif isinstance(raw, str):
                raw = raw.strip()
                label = int(raw) if raw != "" else -1
            else:
                label = int(raw)
        except Exception:
            label = -1

        # build text even if label missing
        solutions = [ex.get("sol1", ""), ex.get("sol2", "")]
        correct = solutions[label] if label in (0, 1) else ""
        answer_letter = ("A" if label == 0 else "B") if label in (0, 1) else ""

        return (
            "### Task: Choose the best solution.\n\n"
            f"Goal: {ex.get('goal', '')}\n"
            f"A. {solutions[0]}\n"
            f"B. {solutions[1]}\n\n"
            f"Answer: {answer_letter}\n"
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
        ctx = ex.get("ctx") or ((ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip())

        endings = ex.get("endings", [])
        # Some versions may store endings as tuple or other sequence; normalize
        if endings is None:
            endings = []
        else:
            endings = list(endings)

        raw = ex.get("label", None)

        # robust parse for label (can be int, "0", "", None)
        try:
            if raw is None:
                label = -1
            elif isinstance(raw, int):
                label = raw
            elif isinstance(raw, str):
                raw = raw.strip()
                label = int(raw) if raw != "" else -1
            else:
                label = int(raw)
        except Exception:
            label = -1

        choices = [(chr(ord("A") + i), endings[i]) for i in range(len(endings))]
        correct = endings[label] if 0 <= label < len(endings) else ""
        answer_letter = chr(ord("A") + label) if 0 <= label < len(endings) else ""

        return (
            "### Task: Pick the most plausible continuation.\n\n"
            f"Context: {ctx}\n\n"
            f"{_join_choices(choices)}\n\n"
            f"Answer: {answer_letter}\n"
            f"{correct}"
        )

    # ---- allenai/winogrande (winogrande_xl) ----
    # fields: sentence (with _), option1, option2, answer ("1"/"2")
    if dp.startswith("allenai/winogrande"):
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
        stem = ex.get("question", "")
        ch = ex.get("choices", {})  # dict with "label": [...], "text": [...]
        labels = ch.get("label", [])
        texts = ch.get("text", [])
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

    # ---- openai/gsm8k ----
    # Output space: a reasoning chain ending in a number. Incompatible with "emit a
    # letter": the adapter cannot serve both with one direction of update.
    if dp.startswith("openai/gsm8k") or dp == "gsm8k":
        q = ex.get("question", "")
        a = str(ex.get("answer", ""))
        return (
            "### Task: Solve the math problem. End with '#### <number>'.\n\n"
            f"Question: {q}\n\n"
            f"Answer: {a}"
        )

    # ---- rajpurkar/squad ----
    # Output space: a span copied out of the context.
    if dp.startswith("rajpurkar/squad") or dp == "squad":
        answers = ex.get("answers", {}) or {}
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        gold = texts[0] if texts else ""
        return (
            "### Task: Answer the question using a span from the passage.\n\n"
            f"Passage:\n{ex.get('context','')}\n\n"
            f"Question: {ex.get('question','')}\n\n"
            f"Answer: {gold}"
        )

    # ---- nyu-mll/glue (mnli / sst2) ----
    # Output space: a label from a DIFFERENT label set than the commonsense sets.
    if dp.startswith("nyu-mll/glue") or dp == "glue":
        if "premise" in ex and "hypothesis" in ex:          # MNLI
            names = {0: "entailment", 1: "neutral", 2: "contradiction"}
            lab = names.get(int(ex.get("label", -1)), "")
            return (
                "### Task: Does the premise entail the hypothesis?\n"
                "Reply with entailment, neutral or contradiction.\n\n"
                f"Premise: {ex['premise']}\n"
                f"Hypothesis: {ex['hypothesis']}\n\n"
                f"Answer: {lab}"
            )
        if "sentence" in ex:                                 # SST-2
            lab = {0: "negative", 1: "positive"}.get(int(ex.get("label", -1)), "")
            return (
                "### Task: Is the sentiment of the sentence positive or negative?\n\n"
                f"Sentence: {ex['sentence']}\n\n"
                f"Answer: {lab}"
            )

    # ---- fallback ----
    # If you pass a dataset that already has "text", use it
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"]

    ex = dict(ex)
    # Otherwise just dump something readable
    return "### Example\n" + json.dumps(ex, ensure_ascii=False, default=str)


def format_example_variant(data_path: str, ex: Dict[str, Any], variant: str = "letter") -> str:
    """Re-target the OUTPUT of an example without changing its content.

    All eight commonsense sets share one output mapping ("emit a letter"), so a single
    linear low-rank update serves every input and an input-conditional update has
    nothing to earn -- which is what we measured (the learned nonlinear amplitude
    collapses to ~0, and rank 4 vs 16 differ by 0.0045 answer accuracy). Mixing
    variants keeps the content and the difficulty fixed while forcing the adapter to
    implement several conflicting input->output maps at once, which is the regime the
    "input-conditional mixture of low-rank updates" claim is actually about.

      letter    : "Answer: A"                 (the standard protocol)
      text      : "Answer: <the option text>"  (free-form, no letter)
      textfirst : the option text, then the letter (inverted order)
    """
    full = format_example_to_text(data_path, ex)
    if variant == "letter":
        return full
    head, sep, tail = full.partition("Answer: ")
    if not sep:
        return full
    letter, _, gold_text = tail.partition("\n")
    gold_text = gold_text.strip()
    if not gold_text:
        gold_text = letter.strip()
    if variant == "text":
        return f"{head}Answer: {gold_text}"
    if variant == "textfirst":
        return f"{head}Answer: {gold_text}\nOption: {letter.strip()}"
    raise ValueError(f"unknown dataset variant {variant!r} (letter|text|textfirst)")


def add_text_column(dataset: DatasetDict, data_path: str, variant: str = "letter") -> DatasetDict:
    """
    Ensure every split has a 'text' column.
    """
    def _mapper(ex):
        return {"text": format_example_variant(data_path, ex, variant=variant)}

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


from datasets import concatenate_datasets

def parse_dataset_spec(spec: str) -> Tuple[str, Optional[str], str]:
    """
    spec: "path" | "path:config" | "path@variant" | "path:config@variant"
    returns: (path, config_or_None, display_name)

    The @variant suffix selects the OUTPUT FORMAT the model must produce for that
    copy of the dataset (see format_example_to_text). Mixing several variants of the
    same data creates genuine task conflict -- identical inputs demanding different
    output mappings -- with no difference in content or difficulty, which is the
    controlled way to ask whether an input-conditional update earns its keep.
    """
    spec = spec.strip()
    variant = "letter"
    if "@" in spec:
        spec, variant = spec.rsplit("@", 1)
        spec, variant = spec.strip(), variant.strip()
    if ":" in spec:
        path, cfg = spec.split(":", 1)
        path, cfg = path.strip(), cfg.strip()
        display = f"{path}:{cfg}" + (f"@{variant}" if variant != "letter" else "")
        return path, cfg, display
    else:
        path = spec
        display = path + (f"@{variant}" if variant != "letter" else "")
        return path, None, display


def load_tokenize_one(
    *,
    data_path: str,
    data_name: Optional[str],
    # dataset_specs: Optional[List[str]],  # NEW
    tokenizer,
    cutoff_len: int,
    val_set_size: int,
    cache_dir: str,
    seed: int = 42,
    variant: str = "letter",
) -> DatasetDict:

    # Load. Several of the commonsense sets (piqa, hellaswag, winogrande, social_i_qa,
    # ...) are still script-based on the Hub, so newer `datasets` refuses them without
    # an explicit opt-in to running the loading script. rajpurkar/squad's script is
    # incompatible with this datasets version, so fall back to its parquet export.
    def _load(path, name):
        try:
            return (load_dataset(path, name, cache_dir=cache_dir, trust_remote_code=True)
                    if name else load_dataset(path, cache_dir=cache_dir, trust_remote_code=True))
        except Exception:
            if path.startswith("rajpurkar/squad") or path == "squad":
                return load_dataset(path, revision="refs/convert/parquet", cache_dir=cache_dir)
            raise

    ds = _load(data_path, data_name.strip() if data_name else None)

    # Tokenize/split -> returns DatasetDict with train/test
    tok = tokenize_and_split(
        ds,
        data_path=data_path,   # important for formatting rules in format_example_to_text()
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        val_set_size=val_set_size,
        seed=seed,
        variant=variant,
    )
    return tok


# -------------------------
# Fixed-length tokenization + split
# -------------------------
def tokenize_and_split(dataset: DatasetDict, data_path: str, tokenizer, cutoff_len: int, val_set_size: int, seed: int = 42, variant: str = "letter"):
    dataset = normalize_splits(dataset)

    # build "text" if missing
    if "text" not in dataset["train"].column_names:
        dataset = add_text_column(dataset, data_path=data_path, variant=variant)

    def tokenize_function(examples):
        out = tokenizer(
            examples["text"],
            # No padding here: DataCollatorForLanguageModeling pads each batch to its
            # own longest sequence. Padding every example to the full 512 window made
            # most batches mostly padding -- pure wasted compute, since the pad
            # positions are masked to -100 and contribute nothing to the loss. The
            # optimisation is exactly loss-equivalent.
            truncation=True,
            max_length=cutoff_len,
        )
        # Deliberately no "labels" here. DataCollatorForLanguageModeling builds them
        # from input_ids AFTER padding and masks pad positions to -100; a pre-made
        # labels column cannot be padded by tokenizer.pad and breaks dynamic batching.
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
        split = tokenized["train"].train_test_split(test_size=val_size, seed=seed)
        tokenized = DatasetDict({"train": split["train"], "test": split["test"]})

    return tokenized


# -------------------------
# Trainable params control
# -------------------------
def set_trainable_only_lena(model):
    """
    Freezes everything EXCEPT FLoRA-related params:
      - A/B
      - activation module params
      - gate params
    Works if your injected modules are LeNALinear/LeNAConv1D with:
      A, B, act, gate_after_a, gate_after_b (ModuleDicts)
    """
    for p in model.parameters():
        p.requires_grad = False

    for _, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls not in ("LeNALinear", "LeNAConv1D"):
            continue

        for attr in ("A", "B", "act", "gate", "steer", "outgate", "ttt", "twopttt", "lora_Ag", "lora_Bg", "lora_A2", "iq_lam", "gate_after_a", "gate_after_b", "lena_A", "lena_B", "lena_act"):
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

        # LeNA-D only *is* DoRA if the magnitude vector adapts; leaving it frozen turns
        # the variant into a fixed rescale and makes the DoRA-decouple ablation
        # meaningless. It stays frozen (and unused) when use_dora is off, so the plain
        # LeNA path keeps its LoRA-matched parameter count.
        if getattr(mod, "use_dora", False):
            mag = getattr(mod, "magnitude", None)
            if mag is not None:
                for p in mag.parameters(recurse=True):
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
    lena_activation: str,
    lena_flex_mode: str,
    lena_activation_kwargs: Dict,
    lena_gate_type: str,
    lena_gate_position: str,
    lena_debug: bool = False,
    lena_gate_mode="global",
    gate_strength="hard",
    lena_use_dora: bool = False,
    lena_norm_before_act: bool = False,
    lena_norm_mode: str = "token",
    lena_gate_init: float = -2.0,
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

    if method == "lena":
        return LeNAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            lena_activation=lena_activation,
            lena_flex_mode=lena_flex_mode,
            lena_activation_kwargs=lena_activation_kwargs,
            lena_gate_type=lena_gate_type,
            lena_gate_position=lena_gate_position,
            lena_debug=lena_debug,
            lena_debug_verbose=False,
            lena_debug_forward=False,
            lena_debug_check_nan=True,
            lena_gate_mode=lena_gate_mode,
            gate_strength=gate_strength,
            lena_use_dora=lena_use_dora,
            lena_norm_before_act=lena_norm_before_act,
            lena_norm_mode=lena_norm_mode,
            lena_gate_init=lena_gate_init,
        )

    raise ValueError(f"Unknown method: {method}")


from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def build_grouped_optimizer(
    model,
    base_lr: float,
    act_lr_mult: float = 0.5,     # activation LR = base_lr * act_lr_mult
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
        # covers: LeNALinear.act.<key>.<param>, Flex* params, etc.
        return (".act." in n) or ("lena_act" in n) or (n.endswith(".knots_y")) or (n.endswith(".c"))

    def is_gate(n: str) -> bool:
        # The unified selection gate registers as "<module>.gate.<adapter>.param".
        # The old gate_after_a/gate_after_b names are gone, so matching only those
        # dropped the gate into the generic bucket -- full LR *and* weight decay,
        # which drags the logit toward 0 i.e. straight onto the hard gate's decision
        # boundary, where the straight-through estimator is at its noisiest.
        return (".gate." in n) or ("lena_gate" in n) or (".gate_after_" in n)

    def is_magnitude(n: str) -> bool:
        # DoRA magnitude is initialized to ||W|| (order 10s); decaying it shrinks the
        # effective base weight norm, which is not a regularizer but a bug.
        return ".magnitude." in n

    def is_adapter(n: str) -> bool:
        # LoRA/DoRA names vary; for your LeNALinear it might be A/B or lena_A/lena_B
        return (".A." in n) or (".B." in n) or ("lena_A" in n) or ("lena_B" in n) or ("lora_" in n)

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

        if ".steer." in n:
            groups["adapter_nodecay"].append(p)   # steer: base_lr, no weight-decay
        elif ".ttt." in n:
            groups["adapter_nodecay"].append(p)   # ttt: base_lr, no weight-decay
        elif is_act(n):
            groups["act_nodecay"].append(p)     # usually no WD for activation shaping
        elif is_magnitude(n):
            groups["adapter_nodecay"].append(p)
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

    # If you truly only train lena params, these "other" groups will be empty.
    add(groups["other_decay"],       base_lr, weight_decay)
    add(groups["other_nodecay"],     base_lr, 0.0)

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    return optimizer

import numpy as np


def preprocess_logits_for_metrics(logits, labels):
    # logits can be a tuple for some models
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)  # [bs, seq]

def compute_metrics(eval_pred):
    preds, labels = eval_pred  # preds: [bs, seq] int, labels: [bs, seq]

    # shift for next-token accuracy
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]

    mask = shift_labels != -100
    denom = mask.sum()
    if denom == 0:
        return {"token_acc": 0.0}

    correct = (shift_preds == shift_labels) & mask
    token_acc = correct.sum() / denom
    return {"token_acc": float(token_acc)}



def debug_trainables(model, method: str, topn_layers: int = 10):
    """
    Prints:
      - total/trainable params
      - per-category trainable params (A/B/act/gate for lena; lora_* for lora/dora)
      - how many layers have A/B/act/gate present + trainable
      - sanity checks (are A/B present at all?)
    """
    import re
    from collections import defaultdict

    def nparams(ps):
        return sum(p.numel() for p in ps)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DBG] Params: trainable={trainable:,} / total={total:,} ({100*trainable/total:.6f}%)")

    # ---- method-specific categorization by parameter name ----
    buckets = defaultdict(list)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()

        if method.lower() == "lena":
            # A/B in your LeNALinear are ModuleDict entries: "...A.<adapter>.weight" / "...B.<adapter>.weight"
            if re.search(r"\.a\.[^.]+\.(weight|bias)$", n) or "lena_a" in n:
                buckets["lena_A"].append(p)
            elif re.search(r"\.b\.[^.]+\.(weight|bias)$", n) or "lena_b" in n:
                buckets["lena_B"].append(p)
            elif ".act." in n or "lena_act" in n or n.endswith((".knots_y", ".c")):
                buckets["lena_act"].append(p)
            elif "gate_after_a" in n or "gate_after_b" in n or "lena_gate" in n:
                buckets["lena_gate"].append(p)
            else:
                buckets["lena_other"].append(p)

        else:
            # PEFT LoRA/DoRA commonly uses names: lora_A, lora_B, lora_magnitude_vector, etc.
            if "lora_a" in n:
                buckets["lora_A"].append(p)
            elif "lora_b" in n:
                buckets["lora_B"].append(p)
            elif "lora_magnitude" in n or "magnitude_vector" in n:
                buckets["dora_mag"].append(p)
            else:
                buckets["peft_other"].append(p)

    # Print bucket totals
    for k in sorted(buckets.keys()):
        print(f"[DBG] trainable bucket {k:>12s}: {nparams(buckets[k]):,} params in {len(buckets[k])} tensors")

    # ---- Layer/module counts (lena) ----
    if method.lower() == "lena":
        lena_layers = [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == "LeNALinear"]
        print(f"[DBG] Num LeNALinear modules: {len(lena_layers)}")
        print("[DBG] First few LeNALinear:", [n for n, _ in lena_layers[:topn_layers]])

        # Count how many LeNALinear have A/B/act/gates populated and how many of those params are trainable
        cnt = {
            "has_A": 0, "has_B": 0, "has_act": 0, "has_gateA": 0, "has_gateB": 0,
            "trainable_A_params": 0, "trainable_B_params": 0, "trainable_act_params": 0, "trainable_gate_params": 0,
        }

        for lname, layer in lena_layers:
            # present?
            hasA = hasattr(layer, "A") and len(getattr(layer, "A", {})) > 0
            hasB = hasattr(layer, "B") and len(getattr(layer, "B", {})) > 0
            hasAct = hasattr(layer, "act") and len(getattr(layer, "act", {})) > 0
            # single unified selection gate (the old gate_after_a/gate_after_b pair is gone)
            hasGA = hasattr(layer, "gate") and len(getattr(layer, "gate", {})) > 0
            hasGB = False

            cnt["has_A"] += int(hasA)
            cnt["has_B"] += int(hasB)
            cnt["has_act"] += int(hasAct)
            cnt["has_gateA"] += int(hasGA)
            cnt["has_gateB"] += int(hasGB)

            # trainable params inside those containers
            if hasA:
                for _, mod in layer.A.items():
                    for p in mod.parameters():
                        if p.requires_grad: cnt["trainable_A_params"] += p.numel()
            if hasB:
                for _, mod in layer.B.items():
                    for p in mod.parameters():
                        if p.requires_grad: cnt["trainable_B_params"] += p.numel()
            if hasAct:
                for _, mod in layer.act.items():
                    for p in mod.parameters():
                        if p.requires_grad: cnt["trainable_act_params"] += p.numel()
            if hasGA:
                for _, mod in layer.gate.items():
                    for p in getattr(mod, "parameters", lambda: [])():
                        if p.requires_grad: cnt["trainable_gate_params"] += p.numel()

        print("[DBG] LeNALinear container presence:",
              f"A:{cnt['has_A']}/{len(lena_layers)}",
              f"B:{cnt['has_B']}/{len(lena_layers)}",
              f"act:{cnt['has_act']}/{len(lena_layers)}",
              f"gateA:{cnt['has_gateA']}/{len(lena_layers)}",
              f"gateB:{cnt['has_gateB']}/{len(lena_layers)}")

        print("[DBG] Trainable params inside LeNALinear:",
              f"A={cnt['trainable_A_params']:,}",
              f"B={cnt['trainable_B_params']:,}",
              f"act={cnt['trainable_act_params']:,}",
              f"gate={cnt['trainable_gate_params']:,}")

        # Simple existence checks by *name*
        has_A_names = any(re.search(r"\.a\.[^.]+\.(weight|bias)$", n.lower()) for n, _ in model.named_parameters())
        has_B_names = any(re.search(r"\.b\.[^.]+\.(weight|bias)$", n.lower()) for n, _ in model.named_parameters())
        print("[DBG] Has A weights in named_parameters():", has_A_names,
              "| Has B weights in named_parameters():", has_B_names)


# -------------------------
# Main training function (single run)
# -------------------------
def train_one_run(
    *,
    base_model: str,
    dataset_specs: Optional[List[str]],
    # data_path: str,
    # data_name: Optional[str],
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    seed: int = 42,
    max_train_samples: int = 0,
    lena_norm_mode: str = "token",
    act_lr_mult: float = 1.0,
    gate_lr_mult: float = 1.0,
    quantize: bool = False,
    eval_step: int = 10,
    save_step: int,
    device_arg: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[str],
    method: str,

    lena_activation: str,
    lena_flex_mode: str,
    lena_activation_kwargs_json: str,
    lena_gate_type: str,
    lena_gate_position: str,

    only_train_lena_params: bool,
    print_forward_mean_ms: bool,

    push_to_hub: bool,
    hub_model_id: str,
    lena_gate_mode="global",
    gate_strength="hard",
    lena_use_dora: bool = False,
    lena_gate_l1_coef: float = 0.0,
    lena_norm_before_act: bool = False,
    lena_gate_init: float = -2.0,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    device = pick_device(device_arg)
    print(f"\n=== RUN method={method} device={device} ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              token=hf_token,
                                              cache_dir=cache_dir)
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
            torch_dtype=torch.float16,   # keep base in fp16 (fits 24-32G GPUs); adapters train via autocast
        )

    model.to(device)
    model.config.use_cache = False


    # Target modules
    target_modules = (
        [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        if lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Parse lena_activation_kwargs
    lena_activation_kwargs = {}
    if lena_activation_kwargs_json and lena_activation_kwargs_json.strip():
        lena_activation_kwargs = json.loads(lena_activation_kwargs_json)

    # Build PEFT config
    peft_cfg = build_peft_config(
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        lena_activation=lena_activation,
        lena_flex_mode=lena_flex_mode,
        lena_activation_kwargs=lena_activation_kwargs,
        lena_gate_type=lena_gate_type,
        lena_gate_position=lena_gate_position,
        lena_debug=False,
        lena_gate_mode=lena_gate_mode,
        gate_strength=gate_strength,
        lena_use_dora=lena_use_dora,
        lena_norm_before_act=lena_norm_before_act,
        lena_norm_mode=lena_norm_mode,
        lena_gate_init=lena_gate_init,
    )

    # Wrap model
    model = get_peft_model(model, peft_cfg)
    model.to(device)
    base_dtype = next(model.base_model.parameters()).dtype
    model = model.to(dtype=base_dtype)

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

    model.to(device)

    # Freeze everything except lena params if requested
    if method.lower() == "lena":
        set_trainable_only_lena(model)

    # Mixed precision: frozen base stays fp16 (memory); trainable adapter params -> fp32
    # for stable optimization (avoids fp16 master-weight underflow). Base is 13G, adapters tiny.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    total, trainable = count_trainable_params(model)
    print(f"Params: trainable={trainable:,} / total={total:,} ({100*trainable/total:.4f}%)")


    debug_trainables(model, method=method)


    if print_forward_mean_ms:
        ms = mean_forward_ms(model, tokenizer, device=device, seq_len=min(128, cutoff_len), iters=20, warmup=5)
        print(f"Mean forward latency: {ms:.2f} ms (seq_len={min(128, cutoff_len)})")

    # Dataset (verbose)
    enable_dataset_download_verbose()

    # Many of these datasets require a config name (e.g., winogrande_xl, ARC-Easy, main).
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # -------------------------
    # MULTI-DATASET LOADING
    # -------------------------
    # dataset_specs overrides single data_path/data_name if provided
    specs = dataset_specs[0].split()   # or []

    train_splits = []
    test_splits_by_name = {}
    print("specs = ", specs)

    for spec in specs:
        dp, dn, display = parse_dataset_spec(spec)
        print(f"\n[DATA] Loading: {display}, {dp}, {dn}")

        variant = display.split("@", 1)[1] if "@" in display else "letter"
        tok = load_tokenize_one(
            data_path=dp,
            data_name=dn,
            variant=variant,
            tokenizer=tokenizer,
            cutoff_len=cutoff_len,
            val_set_size=val_set_size,
            cache_dir=cache_dir,
            seed=seed,
        )

        # Optional debug slicing per dataset
        if DEBUG:
            tr_n = min(36, len(tok["train"]))
            ev_n = min(36, len(tok["test"]))
            tr_ds = tok["train"].select(range(tr_n))
            te_ds = tok["test"].select(range(ev_n))
        else:
            tr_ds = tok["train"]
            te_ds = tok["test"]

        print(f"[DATA] {display} train={len(tr_ds)} test={len(te_ds)}")

        train_splits.append(tr_ds)
        test_splits_by_name[display] = te_ds

    # Concatenate all train splits into one big training set
    if len(train_splits) == 1:
        train_dataset = train_splits[0]
    else:
        train_dataset = concatenate_datasets(train_splits).shuffle(seed=seed)

    # Pick ONE eval dataset for Trainer's periodic eval during training
    # (Final metrics are computed separately for each dataset after training.)
    first_name = list(test_splits_by_name.keys())[0]
    eval_dataset = test_splits_by_name[first_name]

    if max_train_samples and max_train_samples > 0 and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
        print(f"[DATA] Subsampled train set to {max_train_samples} (--max_train_samples)")

    print(f"\n[DATA] Combined training dataset size = {len(train_dataset)}")
    print(f"[DATA] During-training eval dataset   = {first_name} (size={len(eval_dataset)})")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        data_seed=seed,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        # Keep the effective batch at 16 while letting the GPU actually be used:
        # batch_size=1 x accum=16 runs sixteen serial forward/backward passes per
        # optimizer step, which is 4-6x slower in wall clock for identical math.
        gradient_accumulation_steps=max(1, 16 // max(1, batch_size)),
        eval_accumulation_steps=8,
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
        act_lr_mult=act_lr_mult,
        gate_lr_mult=gate_lr_mult,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # if DEBUG:
    #     train_dataset = train_dataset.select(range(min(200, len(train_dataset))))
    #     # keep eval datasets small too
    #     for k in list(test_splits_by_name.keys()):
    #         ds = test_splits_by_name[k]
    #         test_splits_by_name[k] = ds.select(range(min(200, len(ds))))
    #     first_name = list(test_splits_by_name.keys())[0]
    #     eval_dataset = test_splits_by_name[first_name]


    # Try to analysis model with Params whole, Params for training, and the percentage
    # and GFLOPs, inference time (avg for 200 cases)
    print("output_dir=", output_dir)

    # -------------------------
    # Params stats
    # -------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

    print("\n[ANALYZE]")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"% trainable      : {pct:.6f}%")

    # -------------------------
    # Avg inference time (200 samples)
    # -------------------------
    model.eval()
    n_eval = min(200, len(eval_dataset))
    warmup = min(10, n_eval)

    def _to_device(batch):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    # Timed inference
    t0 = time.perf_counter()
    times = []
    with torch.no_grad():
        for i in range(warmup):
            b = eval_dataset[i]
            _ = model(
                input_ids=torch.tensor(b["input_ids"], device=device).unsqueeze(0),
                attention_mask=torch.tensor(b["attention_mask"], device=device).unsqueeze(0),
            )
            device_sync(device)
            times.append(time.perf_counter() - t0)

    avg_ms = 1000.0 * (sum(times) / len(times))

    print(f"Avg inference time (200 samples): {avg_ms:.3f} ms")

    # -------------------------
    # GFLOPs per forward (single sample)
    # -------------------------
    gflops = None
    try:
        from torch.profiler import profile, ProfilerActivity

        b0 = eval_dataset[0]
        ids0 = torch.tensor(b0["input_ids"], device=device).unsqueeze(0)
        att0 = torch.tensor(b0["attention_mask"], device=device).unsqueeze(0)

        acts = [ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(ProfilerActivity.CUDA)

        with torch.no_grad():
            with profile(activities=acts, record_shapes=False, with_flops=True) as prof:
                _ = model(input_ids=ids0, attention_mask=att0)
                device_sync(device)

        total_flops = 0
        for evt in prof.key_averages():
            if getattr(evt, "flops", None):
                total_flops += evt.flops

        if total_flops > 0:
            gflops = total_flops / 1e9
            print(f"GFLOPs / forward (batch=1): {gflops:.3f}, total_flops={total_flops:,}")
        else:
            print("GFLOPs: profiler reported 0 (not supported for some ops/builds).")


    except Exception as e:
        print("GFLOPs: unavailable on this setup:", str(e))


    model.train()


    print(f"Training dataset={len(train_dataset)}")
    print(f"Evaluation dataset={len(eval_dataset)}")

    print("\n[START]")
    # Keep the frozen base in fp16 (memory); Trainer fp16 autocast handles mixed precision.
    # (Casting the whole 7B model to float32 OOMs a 32G GPU.)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # The grouped optimizer was built above and then never passed, so every
        # LR multiplier and the no-weight-decay rules for act/gate were dead code:
        # HF built its own optimizer and decayed the gate logits toward 0.
        # LENA_NO_GROUPED_OPT=1 restores that old behaviour, for A/B comparison.
        **({} if os.environ.get("LENA_NO_GROUPED_OPT", "0") == "1"
           else {"optimizers": (optimizer, None)}),
        compute_metrics=compute_metrics,  # <-- add this
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    if method.lower() == "lena" and lena_gate_l1_coef > 0:
        trainer = LeNAGateL1Trainer(lena_gate_l1_coef=lena_gate_l1_coef, **trainer_kwargs)
        print(f"[LeNA] gate-L1 'learn where' penalty enabled: coef={lena_gate_l1_coef}")
    else:
        trainer = Trainer(**trainer_kwargs)

    if os.environ.get("LENA_GRAD_PROBE", "0") == "1":
        trainer.add_callback(LeNAGradProbe(every=int(os.environ.get("LENA_GRAD_PROBE_EVERY", "20"))))
        print("[LeNA] per-family gradient probe enabled")


    # LENA_EVAL_ONLY=1 skips training entirely: with B zero-initialized the adapter
    # delta is exactly 0, so the final evaluation reports the FROZEN base model. That
    # number is what says whether any of the fine-tuned deltas are meaningful at all.
    if os.environ.get("LENA_EVAL_ONLY", "0") == "1":
        print("[EVAL-ONLY] skipping training; reporting the frozen base model")
    else:
        trainer.train()

    # -------------------------
    # Final evaluation on full test set
    # -------------------------

    torch.cuda.empty_cache()  # optional but helpful
    model.eval()

    print("\n=== Final test metrics ===")
    all_metrics = {}

    for name, ds_test in test_splits_by_name.items():
        # sanitize prefix (Trainer uses it in metric keys)
        prefix = "test_" + name.replace("/", "_").replace(":", "_").replace("-", "_")
        metrics = trainer.evaluate(eval_dataset=ds_test, metric_key_prefix=prefix)

        all_metrics[name] = metrics

        print(f"\n--- {name} ---")
        # print just the important ones in a stable order
        for k in sorted(metrics.keys()):
            print(f"{k}: {metrics[k]}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics_by_dataset.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ---- Export the selection-gate "where-map": per-module gate openness in [0,1].
    # Feed lena_gate_map.json to plot.py for the where-heatmap (Fig.1 panel B).
    if method.lower() == "lena":
        try:
            from peft.tuners.lena.layer import lena_gate_report
            gate_map = lena_gate_report(model)
            with open(os.path.join(output_dir, "lena_gate_map.json"), "w") as f:
                json.dump(gate_map, f, indent=2)
            if gate_map:
                vals = list(gate_map.values())
                frac = sum(v > 0.5 for v in vals) / len(vals)
                print(f"[LeNA where-map] {len(vals)} gated modules | mean openness={sum(vals)/len(vals):.3f} "
                      f"| fraction 'open' (>0.5) = {100*frac:.1f}%  (the 'only X% go nonlinear' number)")
        except Exception as e:
            print("[LeNA where-map] export skipped:", e)



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
            ".act.", "lena_act", "knots_y",  # module containers / spline
        ]) or n.endswith((".a", ".k", ".w", ".p", ".c")):
            act_params.append(p)
            continue

        # --- gate parameters ---
        if "gate_after_a" in n or "gate_after_b" in n or ".gate." in n:
            gate_params.append(p)
            continue

        # --- adapter parameters (A/B or lora) ---
        if any(tok in n for tok in ["lena_a", "lena_b", ".a.", ".b.", "lora_", "lora"]):
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
# Multi-run driver: compare lora/dora/lena variants
# -------------------------
def run_experiments(
    args,
    base_model: str,
    # data_path: str,
    dataset: Optional[str],
    output_dir: str,
    methods: str,
    lena_activations: str,
    lena_flex_mode: str,
    lena_activation_kwargs_json: str,
    lena_gate_type: str,
    lena_gate_position: str,
    lena_gate_mode="global",
    gate_strength="soft",
    **kwargs,
):
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    lena_act_list = [a.strip() for a in lena_activations.split(",") if a.strip()]

    runs = []

    for m in method_list:
        if m.lower() in ("lora", "dora"):
            out = os.path.join(output_dir, m.lower())
            train_one_run(
                base_model=base_model,
                dataset_specs=getattr(args, "dataset", None),  # NEW
                # data_path=data_path,
                # data_name=data_name,
                output_dir=out,
                method=m.lower(),
                lena_activation="identity",
                lena_flex_mode=lena_flex_mode,
                lena_activation_kwargs_json=lena_activation_kwargs_json,
                lena_gate_type="none",
                lena_gate_position="after_b",
                **kwargs,
            )
            runs.append(out)

        elif m.lower() == "lena":
            for act in lena_act_list:
                tag = f"lena_{act}_{lena_flex_mode}_{lena_gate_type}_{lena_gate_position}"
                if "gate_strength" in lena_activation_kwargs_json:
                    if "hard" in lena_activation_kwargs_json:
                        tag += "_hard"
                    else:
                        tag += "_soft"

                print("experiment tag:", tag)

                out = os.path.join(output_dir, tag)

                train_one_run(
                    base_model=base_model,
                    dataset_specs=dataset,
                    output_dir=out,
                    method="lena",
                    lena_activation=act,
                    lena_flex_mode=lena_flex_mode,
                    lena_activation_kwargs_json=lena_activation_kwargs_json,
                    lena_gate_type=lena_gate_type,
                    lena_gate_position=lena_gate_position,
                    lena_gate_mode=lena_gate_mode,
                    gate_strength=gate_strength,
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
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[],
        help="List of datasets. Each item is path or path:config "
             "(e.g., google/boolq or allenai/ai2_arc:ARC-Easy)."
    )

    # parser.add_argument("--data_name", type=str, default="", help="Dataset config name, e.g. winogrande_xl, ARC-Easy, main")
    parser.add_argument("--output_dir", type=str, default="./outputs_compare")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--lena_norm_mode", type=str, default="token",
                        choices=["token", "shared"],
                        help="'token': per-token LayerNorm before phi. 'shared': one running "
                             "scalar, which keeps per-token magnitude (see SharedRMS)")
    parser.add_argument("--act_lr_mult", type=float, default=1.0,
                        help="activation LR = learning_rate * this (measured act grads are ~1e3x "
                             "smaller than B's, so the old hardcoded 0.1 starved them)")
    parser.add_argument("--gate_lr_mult", type=float, default=1.0,
                        help="gate LR = learning_rate * this")
    parser.add_argument("--seed", type=int, default=42,
                        help="run seed; controls data split/shuffle and Trainer init (report >=3 seeds)")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="cap the combined train set (0 = use all); keeps sweeps affordable")
    parser.add_argument("--val_set_size", type=int, default=500)
    parser.add_argument("--quantize", action="store_true")

    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--methods", type=str, default="lora,dora,lena",
                        help="Comma-separated: lora,dora,lena")

    parser.add_argument("--lena_activations", type=str, default="identity,gelu,fourier",
                        help="Comma-separated: identity,relu,gelu,fourier,spline,polynomial")
    parser.add_argument("--lena_flex_mode", type=str, default="channel",
                        help="global|spatial|channel|voxel")
    parser.add_argument("--lena_activation_kwargs_json", type=str, default="",
                        help='JSON string, e.g. {"n_terms":4,"max_h":512,"max_w":1}.')
    parser.add_argument("--lena_gate_type", type=str, default="none",
                        help="none|sigmoid|tanh|rezero")
    parser.add_argument("--lena_gate_position", type=str, default="after_b",
                        help="after_a|after_b|both")
    parser.add_argument(
        "--lena_gate_mode",
        type=str,
        default="global",
        # "input" = g(z)=sigmoid(w.z+b), the input-conditional gate. spatial/voxel are
        # sized by sequence length and are rejected by Gate() as unregisterable.
        choices=["global", "channel", "input", "context"],
        help="Gate mode for position after_a",
    )
    parser.add_argument("--gate_strength", type=str, default="soft",)
    parser.add_argument("--lena_use_dora", action="store_true",
                        help="Use DoRA-style magnitude/direction decomposition (the 'LeNA-D' variant). "
                             "Off by default so the nonlinearity's gain is measured independently of DoRA.")
    parser.add_argument("--lena_gate_l1", type=float, default=0.0,
                        help="Coefficient for the gate-openness L1 penalty ('learn where' sparsity). "
                             "0 disables it; try 1e-3..1e-2.")
    parser.add_argument("--lena_norm_before_act", action="store_true",
                        help="LayerNorm the code z before the activation (input conditioning for phi; "
                             "keeps spline/poly in-range and init-scale insensitive).")
    parser.add_argument("--lena_gate_init", type=float, default=-2.0,
                        help="Pre-sigmoid gate init. NEGATIVE => start near-closed (start at LoRA). "
                             "POSITIVE (e.g. 2.0) => start OPEN; REQUIRED for the L1 'prune' sparsity "
                             "recipe (init-closed + L1 collapses / dead-gates).")

    parser.add_argument("--only_train_lena_params", action="store_true")
    parser.add_argument("--print_forward_mean_ms", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="path/to/repo")

    args = parser.parse_args()

    run_experiments(
        args=args,
        base_model=args.base_model,
        dataset=args.dataset,
        # data_name=args.data_name.strip() or None,
        output_dir=args.output_dir,
        methods=args.methods,
        lena_activations=args.lena_activations,
        lena_flex_mode=args.lena_flex_mode,
        lena_activation_kwargs_json=args.lena_activation_kwargs_json,
        lena_gate_type=args.lena_gate_type,
        lena_gate_position=args.lena_gate_position,

        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
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
        only_train_lena_params=args.only_train_lena_params,
        print_forward_mean_ms=args.print_forward_mean_ms,
        lena_gate_mode=args.lena_gate_mode,
        gate_strength=args.gate_strength,
        lena_use_dora=args.lena_use_dora,
        lena_gate_l1_coef=args.lena_gate_l1,
        lena_norm_before_act=args.lena_norm_before_act,
        lena_norm_mode=args.lena_norm_mode,
        act_lr_mult=args.act_lr_mult,
        gate_lr_mult=args.gate_lr_mult,
        lena_gate_init=args.lena_gate_init,
    )
