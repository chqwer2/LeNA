# Papers portfolio — how to split the innovations, and how to do each

Three innovations (LeNA, IQ-LoRA, TTT-branch) + one real asset (the evaluation protocol). They are all ≈ parity /
negative on GSM8K, so the split is chosen to make each paper's contribution NOT depend on beating LoRA. Two papers, no
salami: distinct contributions, distinct experiments.

═══════════════════════════════════════════════════════════════════════════
## PAPER A — Neurocomputing (ready to build now)
**"Expressive low-rank adaptation and the evaluation artifacts that obscure it (LeNA + IQ-LoRA)"**

- **What goes in:** LeNA (nonlinear, framework; AuroRA = cited prior work) + IQ-LoRA (input-dependent) + the evaluation
  protocol. TTT appears only as a one-line out-of-scope caveat.
- **Venue & why:** Neurocomputing — broad ML scope, moderate bar, accepts parity + a clear angle. (KBS = stretch; Pattern
  Recognition = scope mismatch, skip.)
- **Contributions (do NOT depend on a win):**
  1. unified 1-DOF framework (LoRA / nonlinear / input-dependent differ by one term),
  2. a rigorous protocol with a *proved* Lemma removing 3 artifacts (extraction / degeneration / template),
  3. honest map: expressive variants match LoRA; prior gains were artifacts.
- **Status:** full draft written (`../neurocomputing_LeNA/main.tex`), math done, experiments RED.
- **HOW to do it:** follow `../neurocomputing_LeNA/PLAN.md`. Fastest path = run P1 (mostly EVAL-ONLY on existing
  checkpoints via `eval_fast.py`/`reeval_all.sh`): flip-table, nHASH fig, template table, breadth eval, money figure —
  these ARE the core evidence. Then fill bib + author decisions (LeNA novelty, retitle). P2/P3 (train DoRA/AdaLoRA/IA3,
  ablations) only if reviewers ask.
- **Effort:** ~1 eval sweep + writing polish. This is the near-term submission.

═══════════════════════════════════════════════════════════════════════════
## PAPER B — ICASSP / workshop (future; NOT ready — needs a positive result first)
**"Test-time adaptation as a parameter-efficient module: an OOD-robust branch for frozen LLMs"**

- **What goes in:** the TTT parallel-branch (its own paper — most novel *framing* of the three).
- **Venue & why:** ICASSP if a positive result exists; else NeurIPS-ENLSP / an efficiency workshop (more tolerant of
  preliminary results). Do NOT submit the current negative version anywhere.
- **The blocker (honest):** current branch is worse than LoRA everywhere (0.24/0.27/0.30). Root cause = **train/generation
  mismatch** (exposure bias amplified by test-time adaptation): teacher-forced training fits; autoregressive generation
  adapts to its own wrong prefix and amplifies errors. See `ttt_branch_design.md`.
- **HOW to make it viable (the real research, in order):**
  1. **Fix the mechanism:** replace the teacher-forced closed-form step + prefix-mean `/cnt` state with a **recurrent /
     stateful TTT** (Mamba / linear-attention style, decaying state) that is *consistent* between training and
     autoregressive generation. This also removes the O(N²) `use_cache=False` cost. Start from `ttt_branch4.py` (v4).
  2. **Target the home turf:** train on GSM8K, evaluate under **distribution shift** (GSM-Symbolic perturbations,
     numeric-range / length extrapolation). The claim to chase: branch degrades *slower* than LoRA as shift grows — a
     robustness curve. Even at ~equal in-dist, "in-dist parity + OOD more robust" is a genuine, publishable win.
  3. **Isolate causality:** TTT-on vs TTT-off (static branch) vs LoRA, to prove the robustness comes from *test-time
     adaptation*, not extra parameters.
  4. **Success bar:** in-dist ≥ ~LoRA (0.33) AND a monotone OOD-robustness advantage. Only then write it up.
- **Status:** negative result documented; needs step 1 before anything else. Months of work, not a deadline sprint.

═══════════════════════════════════════════════════════════════════════════
## Shared asset — the evaluation protocol
Belongs primarily to Paper A, but Paper B **reuses** it (cite A). This is what keeps the two honest and non-overlapping:
A = "expressive low-rank + how to measure it"; B = "test-time adaptation + OOD robustness". Different claims, different
experiments, one shared measurement standard.

## One-line decision guide
- Want a near-term publication → **build Paper A now** (run P1, write bib, submit to Neurocomputing).
- Believe in TTT → **do Paper B's step 1** (stateful TTT) as a research effort, then chase the OOD robustness result.
- Do NOT: submit TTT's negative version; claim AuroRA as ours; report parity as a win.
