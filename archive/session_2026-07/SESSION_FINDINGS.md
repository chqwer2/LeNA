# LeNA / Low-Rank Nonlinear Adapter — Session Findings

**Goal:** invent a novel PEFT method that beats AuroRA/LoRA (ICLR-grade), on GSM8K reasoning,
Llama-2-7B, extreme-low-rank regime.

**Honest bottom line:** No original method beat the baselines. The genuine, robust contributions
are **analysis findings** (the fixed-subspace ceiling + the regime characterization + a hard
negative result). Everything that "looked promising" (steer-r2, SC-TTT, SC-aware) collapsed under
higher statistical power (small effects buried in ±0.03–0.15 eval noise). Reported here in full.

---

## 1. Baselines (GSM8K exact-match, Llama-2-7B, generative eval)

| method | r=1 | r=2 | r=4 | r=8 | r=16 | r=32 |
|---|---|---|---|---|---|---|
| LoRA (500-sample) | 0.186 | 0.194±.03 | 0.197 | 0.255 | 0.220 | 0.266 |
| AuroRA (500-sample) | 0.224 | **0.257±.04** | 0.176 | 0.235 | 0.261 | 0.256 |
| gap (aurora−lora) | +0.038 | **+0.063 (2.9σ)** | −0.021 | −0.020 | +0.041 | −0.010 |

Self-consistency voting (Wang 2022, existing, N=8): pushes any adapter to ~0.30–0.34
(greedy ~0.20 → vote ~0.32). Strong but not ours; costs N× inference.

## 2. Our designs (all implemented + tested)

| design | ours? | best (r=2) | verdict | note |
|---|---|---|---|---|
| **auroraf** (AuroRA + provable per-dim interp fallback) | yes | 0.233 | **ties AuroRA, zero cost** | only non-failing original design |
| aurorag (a=0 residual fallback) | yes | 0.193–0.211 | fail | bad init (starts at identity) |
| **steer** (input-steered subspace, random dir) | yes | 0.127–0.153 | fail | unstable; ws blows up |
| steer (SVD-init weight-aligned dir) | yes | 0.107 | fail (worst) | principled dirs didn't help |
| AuroGLU (AuroRA(z)⊙GLU-gate(z)) | yes | ~0.26 | plateau | same ceiling |
| GLU (multiplicative bottleneck) | first-use | 0.24 | plateau | >lora, <aurora |
| cross-token gate (causal cumulative mean) | yes | 0.167 | fail (worst gate) | novel axis, doesn't help |

## 3. Test-time methods (mechanism existing; applied by us)

| method | verdict | detail |
|---|---|---|
| per-sample TTT (backprop, LM-on-question) | **crashes** | lr=1e-3 → acc 0.00; lr=1e-5 → neutral. No lr sweet spot. |
| self-consistency TTT (per-sample) | fail | sc_ttt < sc_vote (adaptation hurts vs voting); more K = worse |
| global/transductive TTT | neutral | lora +0.025 / aurora −0.025 → ~0 (noise) |
| SC-aware / STaR training | mild/noisy | lora_r2 dgreedy +0.0875 but aurora_r2 −0.0125 (not replicated); dvote +0.03 but NOT low-rank-specific (r8 dvote +0.16 largest) |

## 4. "Once promising then died" (the key methodological lesson)

- **steer @ r=2**: first showed gap +0.10 (looked huge) → at 500 samples + more seeds = noise, failed.
- **SC-TTT**: at 40 samples ttt-vote +0.025 (looked like a win) → at 80 samples ttt-vote −0.06…−0.11.
- **SC-aware lora_r2**: greedy +0.0875 (looked like breakthrough) → aurora didn't replicate; vote baseline
  swung 0.30 vs 0.15 between runs (noise 0.15 >> claimed +0.03 signal).

**Lesson: small adapter tweaks × high-variance reasoning eval (±0.03–0.15 at 80–150 samples) = false signals.**

## 5. Genuine contributions (robust, novel — the paper backbone)

1. **Fixed-subspace ceiling (provable + measured).** LoRA and every nonlinear-bottleneck variant
   (AuroRA/CeRA/LeNA) confine the update to col(B), a fixed r-dim subspace; nonlinearity provably
   cannot escape it. Measured effective output rank ≤ r: **1.87 (r2), 3.63 (r4), 7.27 (r8)**.
2. **Regime characterization.** Nonlinearity helps LoRA **only at r=2** (+0.063, 2.9σ), neutral/negative
   at r≥4 — non-monotonic, peaked (corrects CeRA's implied monotone "nonlinearity substitutes for rank").
3. **Efficiency observation.** aurora-r2 ≈ lora-r8 (≈4× fewer params) — the one positive that held.
4. **Effective-rank probe** (measure_nonlin.py / ablate_linearize.py) — reusable adapter diagnostic.
5. **Hard negative result.** Breaking the fixed subspace (steer, 3 ways incl. SVD-init) fails — the
   "obvious fix" doesn't work; low-rank adaptation is fragile to added/perturbed directions.
6. **Instability diagnosis (in progress).** per-sample adaptation of the single tuned adapter corrupts
   it (no lr sweet spot) → motivates an isolated per-sample branch (two-branch design).

## 6. Design-space ideas proposed (NOT ours / not validated)
Norm-preserving nonlinearity (OFT×AuroRA), consistency-weighted training, spectral+norm-preserving
steer, two-branch adapter (global frozen + isolated per-sample local). Existing tricks worth trying:
PiSSA, LoRA+, rsLoRA, DoRA, OFT/BOFT, FourierFT, Kronecker/Monarch, MoLE.

## 7. Code inventory (session_pkg/)
- `code/` — eval/train/probe scripts: eval_generate, measure_nonlin, ablate_linearize, ttt_eval,
  ttt_sc, ttt_global, ttt_diag, sc_train(_iter), test_*, patch_* (activation/optimizer patches),
  Llama_Adaptation, analyze_runs
- `peft_lena/` — modified LeNA tuner (activations.py adds AuroRAF/AuroGLU/FlexGLU/SteerHead;
  layer.py adds steer branch + device fix; config.py Literals; model.py)
- `slurm/` — all SLURM run scripts
- `all_results.csv` — 92 adapter results (150 & 500 sample)
- `ttt_sc_results.txt` — all TTT/SC/diag log RESULT lines

## 8. Recommendation
Write the **analysis / efficiency paper**: fixed-subspace ceiling + regime + efficiency + negative
results. Real, evidence-backed, completable (needs: cross-task replication of r=2 peak, 500-sample
multi-seed to fight noise, ceiling theorem formalization). Honest tier: borderline-ICLR, but true and done.
