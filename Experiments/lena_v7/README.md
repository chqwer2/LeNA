# LeNA v7 (PR version)

`lena_v7.py` implements the adapter of the PR manuscript (Sec. 3, Algorithm 1): two reads `u = A1 x`, `v = A2 x`; a
per-channel piecewise-linear spline (K = 16 knots on [-3, 3]) fitted from the input's (u, v) scatter by a tridiagonal
least-squares solve with LDLᵀ pivot retention (τ = 1); a positive-part James–Stein step toward a learned prior ρ;
causal prefix statistics for decoder LMs, carried through the KV cache and reordered with beam search.
It has no peft dependency; `inject(model, targets, adapter="lena"|"lora", ...)` swaps `nn.Linear` layers.

| Option | Paper |
|---|---|
| `mode=lena` (default), `learn_omega` | LeNA, learned ω |
| `learn_omega=False` | LeNA ω = 0 (James–Stein) |
| `mode=fixed` / `plain` / `plain, K=2` / `predictor` | fixed curve s = 0 / plain fit s = 1 / per-input affine / statistic predictor |
| `freeze_A2`, `prior=identity`, `prior=u_pred` | frozen random A2, identity prior, predictor prior on u only |
| `dora=True` | LeNA-D |
| `adapter="lora"` (± `dora`) | LoRA, LoRA† (rank r'), DoRA in the same code base |

Tests: `python -m pytest -q test_lena_v7.py` (definitions vs dense references: retention rule, solve, RSS identity,
step formula, causal = bidirectional on prefixes, padding, gradcheck, variants) and `python test_llama_tiny.py`
(tiny Llama: left padding; cached beam search equals uncached decoding; checked on transformers 4.52 and 5.4).

Commonsense (T1, T4, T7): `commonsense/` follows the LLM-Adapters / DoRA protocol (prompt copied verbatim, cutoff 256,
beam 4, answer extraction). `bash commonsense/run_llama3.sh lena 0`. LLaMA3-8B, r = 32: LoRA 0.705 %, LeNA 1.098 %,
LoRA† r' = 50 (1.102 %).
