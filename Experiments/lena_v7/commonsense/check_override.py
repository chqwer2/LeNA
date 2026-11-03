"""Check that eval-time overrides fixed (s=0) and plain (s=1) give different scores (diag showed identical accuracy)."""
import json, os, sys, torch
sys.argv = [sys.argv[0]]
import diagnose_lena as D
from transformers import AutoModelForCausalLM
import lena_v7
run = "runs/llama3_lena"; cfg = json.load(open(os.path.join(run, "lena_config.json")))
tok = D.load_tokenizer("meta-llama/Meta-Llama-3-8B"); tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None); D.inject(model, cfg["targets"], adapter=cfg["adapter"], **kw)
sd = torch.load(os.path.join(run, "adapter.pt"), map_location="cpu")
for n_, p in model.named_parameters():
    if n_ in sd: p.data = sd[n_].to(p.device, torch.float32)
model.eval(); D.install_lena_hooks(model); S = D.Scorer(model, tok)
data = json.load(open("dataset/piqa/test.json"))[:4]
for comp in (False, True):
    lena_v7.set_compile(comp)
    for ex in data:
        pr = D.generate_prompt(ex["instruction"]); c = D.candidates("piqa", ex["instruction"])
        print("compile", comp, "gold", ex["answer"], {k: S.score(pr, c, **ov) for k, ov in (("learned", {}), ("fixed", dict(mode="fixed")), ("plain", dict(mode="plain")))}, flush=True)
