# Innovation round (2026-07-29): two new PEFT mechanisms + honest novelty positioning

This round designed and implemented **two genuinely new PEFT mechanisms**, each grounded in the
literature (novelty verified against 2025–2026 work) and each smoke-verified before launch.

Motivation / reframe from the prior rounds:
- Every prior "adapter trick" (LeNA/steer/OutGate/GLU/auroraf/TTT-SwiGLU/TTT-gram/two-path
  mean-split/isolated test-time adaptation) was **null on real accuracy** — see `RESULTS.md`.
- KEY correction: the null is likely a **measurement artifact**, not saturation. Our real-eval
  **noise floor is ±0.024 (eval_choice) / ±0.055 (gsm-gen @200)** — LARGER than the 1–3% gain a
  real PEFT innovation shows. So the testbed cannot detect a real mechanism win. → use
  **low-variance tasks + headroom + multi-seed**.
- The "ΔW parameterization" space is one of ML's most crowded (LoRA/DoRA/VeRA/PiSSA/OLoRA/EVA/
  LoRA-GA/AdaLoRA/SoRA/OFT/BOFT/HRA/FourierFT/KronA/SVFT/rsLoRA/LoRA+ …). Genuine novelty lives in
  the uncrowded corners: **input-dependence (not gating)** and **the objective/dynamics**.

---

## Method 1 — IQ-LoRA: Input-dependent Quadratic LoRA

**Formula** (code = linear LoRA + a *residual* second-order feature-interaction term):

    h(x) = W0 x + (α/r) · B ( A1 x  +  λ · (A1 x) ⊙ (A2 x) )

- `A1,A2 ∈ R^{r×d}` two down-projections, `B ∈ R^{d×r}` up-projection, `⊙` = Hadamard in the r-dim code.
- The code `c(x) = A1 x + λ (A1 x ⊙ A2 x)` is **quadratic in x** ⇒ the effective ΔW(x) **varies with the
  input** (captures feature-pair interactions x_i·x_j that no linear code can).
- **Safe init**: `B=0`, `A1` random (= standard LoRA), `A2` random, **`λ=0` ⇒ starts as EXACT LoRA**;
  λ grows only if the quadratic term helps. (λ init is env-controlled: `LENA_IQ_LAM`.)
- Params: 1.5× LoRA (adds `A2`). Fully parallel, no sequence dependence, no generation issues.

**Novelty (verified against 2025–2026 literature)** — the distinction is **functional vs structural**:

| Prior (crowded) | what it does | ΔW input-dependent? |
|---|---|---|
| PERA / LoHa / HiRA / BoHA / KRAdapter | Hadamard/polynomial/Khatri-Rao in **weight space** → higher *fixed* rank | **No** (fixed ΔW) |
| GLU adapters | `z ⊙ σ(Wz)` = a **gate** (attenuation) | Yes, but a gate |
| **IQ-LoRA (ours)** | Hadamard in **input-code space** → ΔW(x) **quadratic in x** | **Yes, not a gate** |

PERA states its own expansion is *"structural (in parameterization), NOT functional (varying with
input)"* — that line is exactly the white space IQ-LoRA occupies. Ties to our **fixed-subspace
ceiling** finding: LoRA & all weight-space variants confine Δh to a fixed subspace (linear in x);
IQ-LoRA makes Δh quadratic ⇒ the write subspace moves with input (functional escape).

**Status**: implemented (`code/patch_iq.py`, `LENA_IQ`), smoke-verified (λ=0 ≡ LoRA; λ≠0 activates
quadratic; all params train). Single-task MNLI (0.88, near-saturated) = null (+0.002, λ→0). Re-run on
multi-task (headroom on gsm/piqa) with λ∈{0, 0.5} — see RESULTS.

---

## Method 2 — Two-path TTT (ViT³-grounded, made CAUSAL)

Faithful port of **ViT³ (CVPR 2026 Oral, arXiv 2512.01643)** to a decoder-LM adapter. Verified against
the official code (`LeapLabTHU/ViTTT`): ViT³'s TTT is an **inner-loop-in-forward** mechanism (NOT
test-time adaptation on the test input), with **two parallel pathways**, trained by **standard CE**.

**Formula** (two parallel inner-TTT pathways, up-projected):

    Δ(x) = B_g · SwiGLU_{w*(x)}(q)     [global: causal SwiGLU inner-TTT, sequence mixing]
         + B_ℓ · Conv1d_{w*(x)}(q)     [local:  causal depthwise conv1d]

- **Global** = ViT³'s simplified-SwiGLU inner module: weights (w1,w2) fit on (k→v) via a **hand-derived
  one-step gradient** (closed-form, differentiable, no 2nd-order autograd), applied to q as
  `(q w1*) ⊙ silu(q w2*)`.
- **Local** = ViT³'s 3×3 depthwise-conv pathway → LM analog = causal depthwise conv1d.
- **KEY fix vs the earlier non-causal port**: the inner fit is made **CAUSAL** (cumulative prefix
  state, position i sees only j≤i). The non-causal port leaked future tokens and **collapsed
  autoregressive generation** (mnli-gen 0.87 → 0.31); the causal version fixes this by construction.
- **Training** = standard task CE + backprop (like ViT³), differentiating through the closed-form
  inner step. `B_t = 0` init ⇒ starts at base.

**Novelty positioning**: ViT³'s two-pathway TTT-as-a-layer is CVPR-2026; porting it to a **PEFT adapter
on a frozen LLM** + making it **causal** (which vision never needed) is the new twist. Distinct from
Akyürek-2024 / TTRL (those adapt at *test time* on the test input; ViT³'s TTT is a forward-pass layer).

**Status**: implemented (`code/patch_2pttt.py`, `LENA_2PTTT`), **causality smoke-verified** (perturbing
position 8 leaves outputs 0–7 exactly unchanged; positions ≥8 change). Multi-task training + eval — see
RESULTS. Critical check: mnli-gen must stay >0.8 (proving the causal fix vs the non-causal 0.31).

---

## Files
- `peft_lena/layer.py` — final LeNA layer with ALL methods (OutGate-gate/reflection, TTTHead
  SwiGLU + gram variants, TwoPathTTT causal, IQ quadratic, two-path mean-split). Env-gated:
  `LENA_OUTGATE / LENA_TTT / LENA_TWOPATH / LENA_IQ (LENA_IQ_LAM) / LENA_2PTTT`.
- `code/patch_*.py` — the exact patch scripts that produced each mechanism (reproducible).
- `code/eval_ttt.py` — isolated test-time-adaptation eval (K=0 vs K-step, paired).
- `slurm/run_*.sh` — all SLURM launch scripts.
- `logs/eval_results_key.txt` — extracted key result lines (no progress bars).
- `RESULTS.md` — consolidated results table + honest conclusions.
- `SESSION_FINDINGS.md` — full findings across all rounds.
