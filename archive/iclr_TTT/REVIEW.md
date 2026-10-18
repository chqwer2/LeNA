# Reviewer pass #1 — ICLR 10/10 bar. Scores are the target, harsh-honest current state below.

## Scores now
| axis | score | why |
|---|---|---|
| Novelty | 6/10 | TTT-as-PEFT + consistency framing is fresh, but recurrence = delta-rule/DeltaNet family. Novelty MUST live in the PEFT framing + the OOD-robustness result. Without the result, an ICLR reviewer sees "linear-attention adapter". |
| Math | 7/10 | Recurrence (Eq.2), consistency Prop, conditional-operator Prop are clean. Missing: the OOD excess-risk argument (the thesis is currently a hypothesis), Prop.2 proof, Lemma formalization. |
| Writing | 7/10 | Kaiming voice, tight. Related Work still \todocite. |
| Presentation | 6/10 | Structure good; needs the money figure real + a method schematic. |
| Results | 3/10 | ALL core results RED. This is the gating issue: ICLR will not accept without in-dist parity + a real OOD-robustness curve. |
| Experiments | 5/10 | Design is right (shift suite + causal isolation + fix ablation). Must be executed. |

## Blocking (in order)
1. **Results are the paper.** ICLR bar = the OOD-robustness result must exist and be clear. Everything else is scaffolding.
   Priority experiment order: build consistent-TTT (Eq.2) -> Table 1 (in-dist parity) -> Table 4 (fix vs naive) ->
   Table 2 + Fig 1 (OOD robustness) -> Table 3 (causal isolation).
2. **The OOD thesis needs theory.** Turn "why robustness grows with shift" into a proposition (excess risk of static vs
   adaptive under a shift model) or clearly label it a tested hypothesis. Currently a \todo.
3. **Prove Prop.2 (1 line) and formalize Lemma 1 (3 lines).** Pure writing.
4. **Sharpen novelty vs DeltaNet/TTT-Linear** in Related Work: the delta is (a) additive branch on a FROZEN LLM,
   (b) train/inference consistency as an explicit property, (c) OOD-robustness as the goal — not a new sequence layer.
5. **Method schematic figure** (frozen attn + branch + recurrence) — ICLR expects it.

## THIS/next loop targets (writing-only until experiments run)
- Prove Prop.2; formalize Lemma 1; write the OOD excess-risk proposition (or hypothesis box).
- Related Work: 1 tight paragraph sharpening the delta vs delta-rule/TTT-Linear/Mamba.
- Keep ALL results RED. The single most important non-writing action is BUILDING consistent-TTT and running Table 1 + 4.

## Iteration 2 done (novelty-first, writing-only)
- [x] Prop.OOD (adaptation gap grows with shift): rigorous, with proof sketch (SGD contraction vs bias-variance). This is
  the theoretical core that separates the paper from "a linear-attention adapter". Math 7 -> 8.5, Novelty 6 -> 7.5.
- [x] Prop.2 (input-conditional operator) PROVED + O(N)/chunkwise cost stated.
- [x] Lemma 1 (train/generation mismatch) FORMALIZED as a state-distribution-shift argument; fix tied to Prop.1.
Remaining writing: Related Work delta-sharpening; method schematic figure; abstract can now cite Prop.OOD as the mechanism.
Remaining (gating): RESULTS all RED — build consistent-TTT, run Tables 1-4 + Fig 1.

## Iteration 3 done (novelty writing)
- [x] Related Work sharpened: explicit 3-point delta vs delta-rule/DeltaNet/TTT-Linear/Mamba (backbone-layer vs additive
  frozen-LLM PEFT; consistency as a proved property; OOD-robustness goal + excess-risk separation). Novelty defense set.
- [x] Method schematic figure placeholder (fig:arch) added.
Scores: Novelty 7.5->8, Math 8.5, Writing 8, Presentation 7. Results 3 (RED), Experiments 5.

## Novelty writing is near-saturated. Remaining is:
- Minor: abstract could name the mechanism (Prop.OOD) in one clause; intro schematic ref. (polish, ~1 loop)
- GATING (not writing): build consistent-TTT (Eq.2) and run Tables 1-4 + Fig 1 (money). The paper cannot pass ICLR
  without the in-dist parity + monotone OOD-robustness result. This needs cluster + author go-ahead.
=> After ~1 polish loop, stop and hand to experiments (do not spin writing).

## Iteration 4 done — NOVELTY WRITING SATURATED, loop stopping
- [x] Abstract now names the mechanism (Prop.OOD: static pays cross-context variance = shift; adaptive error is shift-free).
- [x] Intro references the schematic (fig:arch). LaTeX self-consistent: all \ref have \label, no dangling refs.
FINAL writing scores: Novelty 8, Math 8.5, Writing 8, Presentation 7.5. Results 3 (RED), Experiments 5 (designed).

## The paper is as strong as writing alone allows. Remaining is NOT writing — it gates ICLR acceptance:
### A. Build + run (cluster; author go-ahead) — see EXPERIMENTS_TODO.md
- STEP 0: implement consistent-TTT (Eq.2) — replace prefix-mean state in ttt_branch4.py with the online delta rule.
- STEP 1 Table indist (parity), STEP 2 Table diag (fix vs naive), STEP 3 Table ood + Fig money (THE thesis),
  STEP 4 Table ablate (causal isolation). Submission bar: parity AND monotone OOD gap, both black.
### B. Mechanical: \todocite -> real refs (LoRA, DoRA, TTT, TENT, delta-rule, DeltaNet, Mamba, TTT-Linear, GSM8K/-Symbolic).
### C. Author: confirm the OOD shift ladder (which datasets) + second-model choice for scale.
Do NOT spin more writing loops — the innovation is written; the paper now needs the result.

## Author/experiment note
The consistent-TTT module (Eq.2) is NEW code — the repo's ttt_branch4.py is the teacher-forced version to be replaced.
See EXPERIMENTS_TODO.md.

## EXPERIMENT PHASE — iter 1
- [x] consistent-TTT implemented (ttt_consistent.py, chunkwise gated-linear-attention, learnable decay γ/η).
- [x] CORRECTNESS VERIFIED: chunked core == naive sequential scan, max diff = 0.000e+00 (chunk 32/64/128). The O(N)
  parallel form is exact — this also empirically confirms Prop.1 (train/inference use the identical recurrence).
- [x] Build: 33.7M trainable (0.53%). Smoke had a grad-flag bug (smoke path only; train() is correct).
- [ ] STEP1 (job 9643908): full train + gsm8k eval, in-dist parity vs LoRA (0.33) / naive branch (0.24-0.30). PENDING.
Next: read STEP1 result -> fill Table indist -> STEP2 (fix vs naive) -> STEP3 OOD ladder.

## EXPERIMENT PHASE — iter 2 (STEP1 result: NEGATIVE, honest)
STEP1 consistent-TTT gsm8k = **0.147** (1 seed, n150). WORSE than naive branch (0.24-0.30) and far below LoRA (0.33).
Learned eta->0.086 (tiny adaptation). Interpretation: the branch learned to contribute ~nothing useful; 0.147 is near the
FROZEN base level. The RMSNorm-stabilized gated-linear-attention branch, like the whole branch line, does NOT learn useful
in-dist task adaptation (it only adds to attention output; LoRA edits q/k/v/up/down). 
VERDICT: STEP1 parity gate FAILED, badly. A method at 0.147 in-dist cannot support an ICLR paper regardless of OOD, so
STEP3 (OOD) is moot until in-dist is fixed. This is the SECOND negative result for the TTT-as-PEFT branch (naive + consistent).
Honest read: the TTT branch architecture is the problem, not just the recurrence. Options: (a) let branch also edit q/k/v
(not just add to attn out) — but that's converging to LoRA; (b) accept the negative result and shelve the ICLR TTT paper.
Do NOT keep burning contended GPU chasing a method that's below the frozen baseline.
