# LeNA / IQ-LoRA / TTT-branch — designs & honest results

This folder records the *methods and writing* of the expressive-PEFT line, so the ideas survive independent of any
Overleaf project. Big artifacts (checkpoints, weights, envs) are intentionally excluded.

## The three innovations (and what is genuinely ours)
| name | one-line | novelty | honest result (GSM8K, Llama-2-7B, fair `first-####` eval) |
|---|---|---|---|
| **LeNA** | framework: learnable nonlinear activation φ in the low-rank code, `h = W0x + B·φ(g⊙ν(Ax))` | the *framework*; the AuroRA activation itself is **prior work (NeurIPS'25, 2505.18738)** | ≈ LoRA at every rank (r1..r32, Δ≤0.02) |
| **IQ-LoRA** | input-dependent update: `z + λ(z⊙A₂x)` (IQG = `z⊙(1+λ·tanh(A₂x))`) | input-conditional low-rank operator | ≈ LoRA (IQ2 r=2 = 0.230; null-but-stable) |
| **TTT-branch** | parallel test-time-training branch on the residual stream | TTT-as-PEFT (framing) | **negative** — see `ttt_branch_design.md` |

## The real, defensible contribution: a rigorous evaluation protocol
Every apparent "win" above evaporated under fair evaluation. The lasting result is *why*: three named, formalized
evaluation artifacts that flip PEFT conclusions on generative reasoning benchmarks. See `eval_protocol.md`. This is what the
Neurocomputing paper (`../neurocomputing_LeNA/`) is built on: **expressive low-rank variants match LoRA; prior gains were
artifacts.** Parity, correctly measured, is the point.

## Honest headline numbers
- LeNA(AuroRA) vs LoRA rank curve (fair): r1 .262/.269, r2 .285/.276, r4 .293/.283, r16 .314/.295, r32 .282/.294.
- The retracted "16× efficiency": LoRA r2 scored **.194 under `last-number`** but **.276 under `first-####`** — a pure
  extraction artifact (LoRA repetition-degenerates; `last-number` reads a spurious trailing token).

## Papers this maps to
- **Neurocomputing** (`../neurocomputing_LeNA/`): LeNA + IQ-LoRA + the evaluation protocol. Draft complete; experiments RED.
- **TTT-branch**: not submittable yet (negative results); needs the train/inference-mismatch fix first — future ICASSP.
