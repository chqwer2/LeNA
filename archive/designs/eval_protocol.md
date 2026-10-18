# Rigorous evaluation protocol for PEFT on generative reasoning (the real contribution)

Generative benchmarks score a decoded string, not a label. Three degrees of freedom in the pipeline can dominate the
method effect. Each is named, formalized, and fixed. This is the backbone of the Neurocomputing paper.

## Artifact 1 — extraction bias under trailing degeneration
GSM8K answers end with `#### n`. Two extractors:
- `E_last` (naive): last numeric token of the completion.
- `E_first` (ours): number in the **first** `####` match; fallback to last number.

**Definition (degeneration-invariance).** E is invariant if `E(y)=E(y‖s)` for any trailing string s appended after the
first terminal answer. `E_first` is invariant; `E_last` is not.

**Lemma (method-dependent bias).** If method M appends a further number after a correct answer with prob. p_M, then under
`E_last` the measured accuracy is `acc_M − p_M·Pr[correct]`; the downward bias `p_M·Pr[correct]` is method-dependent and can
be arbitrarily large. `E_first` removes it (p_M drops out). *Proof:* appending any token makes `E_last` wrong while
`E_first` stays correct (by the definition); take expectations over the p_M correct-but-continued completions. ∎

## Artifact 2 — baseline-specific degeneration (instantiates the Lemma)
LoRA checkpoints degenerate into repeated `#### n #### n …` (**15–36 blocks observed**) far more than the expressive
variants, so p_LoRA ≫ p_others and `E_last` deflates the baseline *specifically*, manufacturing a gap.
Evidence to plot: **nHASH histogram** (count of `####` blocks per completion), LoRA vs AuroRA.

## Artifact 3 — template collapse in perturbation benchmarks
GSM-Symbolic = 100 templates × 50 instances. Evaluating the first N<50 samples a **single template** — an estimate of one
perturbation family reported as the benchmark. Fix: **one-instance-per-template** subset. (Cost me a bogus 18× "win" once.)

## The concrete flip (real numbers)
At r=2, GSM8K: LoRA `.194` (E_last) vs `.276` (E_first); AuroRA `.257` vs `.285`. The naive protocol shows a `+.063`
"16× efficiency" gap; the fair protocol shows `+.008` (parity). **Same checkpoints.**

## Tooling
- `eval_fast.py` (cluster: FLoRA/Experiments): batched, left-padded, `E_first` extractor + diverse subsets. ~9× faster
  than per-example decoding. Caveat: left-pad batching corrupts cumulative-causal-state adapters (TTT) → those go bs=1.
- `reeval_all.sh`: re-score all saved adapters under the protocol (this is what corrected the whole rank curve).

## Rule (for any PEFT reasoning comparison)
Report the **extraction operator** and the **subset construction**. As the Lemma shows, the method effect can be smaller
than the protocol effect — so an unspecified protocol makes a PEFT comparison unfalsifiable.
