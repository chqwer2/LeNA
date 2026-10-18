# LeNA: Learning Nonlinear Adaptation

Low-rank adapters whose code map is a per-input activation curve.

## Current version (Pattern Recognition manuscript)

`Experiments/lena_v7/` — pure-PyTorch implementation of the current method: two linear reads `u = A1 x`,
`v = A2 x`; a per-channel piecewise-linear spline fitted from each input's `(u, v)` scatter by a tridiagonal
least-squares solve; a positive-part James–Stein step toward a learned prior shared across token positions;
causal prefix statistics for decoder LMs (carried through the KV cache and beam search). LoRA / DoRA baselines and
all ablation variants live in the same code base. See `Experiments/lena_v7/README.md` for usage, tests and the
commonsense-reasoning protocol (`Experiments/lena_v7/commonsense/`).

```sh
cd Experiments/lena_v7
python -m pytest -q test_lena_v7.py      # estimator definitions vs dense references
python test_llama_tiny.py                # tiny Llama: left padding, cached beam search == uncached
```

## Earlier code

- `Experiments/peft/`, `Experiments/Llama_Adaptation.py`, `Experiments/image_video_text_understanding/`,
  `Experiments/visual_instruction_tuning/` — conference (ICML) version of LeNA on a vendored `peft`.
- `Experiments/lena_probe/` — GPU-free probes for the conference version.
- `Nonlinearity/` — activation-family sketches.
- `archive/` — past experiment sessions, design notes and drafts; not maintained.
