"""H15 launcher: run an unmodified training script with the lena_aff / lora_inorm adapter swapped in.

  python train_h15.py lena_aff   mri_sam_seg.py <args with --adapter lena> --out runs/X
  python train_h15.py lora_inorm clip_fewshot.py <args with --adapter lora> --out runs/Y
Writes <out>/h15_variant.json so the run is never mistaken for plain LeNA / LoRA (result.json still says the
script's --adapter value).
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import lena_aff  # noqa: E402

variant, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
need = {"lena_aff": "lena", "lora_inorm": "lora"}[variant]
assert args[args.index("--adapter") + 1] == need, f"{variant} needs --adapter {need}"
lena_aff.install(variant)
out = args[args.index("--out") + 1]
os.makedirs(out, exist_ok=True)
json.dump(dict(variant=variant, script=script, args=args), open(os.path.join(out, "h15_variant.json"), "w"), indent=1)
sys.argv = [script] + args
runpy.run_path(os.path.join(HERE, script), run_name="__main__")
