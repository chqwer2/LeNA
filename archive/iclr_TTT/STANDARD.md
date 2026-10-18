# ICLR 10/10 bar for the TTT paper (fixed standard, held every loop)

**Honest status (do not hide).** The teacher-forced TTT branch is a NEGATIVE result: worse than LoRA in-dist (0.24-0.30 vs
0.33), from a train/generation MISMATCH (exposure bias amplified by test-time adaptation). This paper is written around the
FIXED method (a recurrent, train/inference-consistent fast-weight TTT) whose experiments are RED (to run). We never report a
number we do not have; RED tables mark exactly what must be produced. The paper is submittable ONLY once the RED core
(in-dist parity + an OOD-robustness advantage) turns black.

**Story (locked).** LoRA's update is FIXED at test time. A parameter-efficient branch that does test-time training adapts
its effective weights PER INPUT, so under distribution shift it can recover accuracy a static ΔW cannot. The prior TTT-as-
adapter attempt failed for a concrete, fixable reason (mismatch); the fix (an online delta-rule recurrence with learnable
decay, identical at train and generation, O(N)) removes it. Contribution: (i) TTT-as-PEFT for frozen LLMs with a
train/inference-consistent recurrence, (ii) the mismatch diagnosis + fix as the technical core, (iii) an OOD-robustness
result static PEFT structurally cannot match — IF the experiments confirm it.

**Reviewer bar (score each 1-10 every loop; ICLR target all ≥9):**
- Novelty: TTT-as-PEFT framing + the consistency fix must be crisp and clearly beyond DeltaNet/TTT-linear/LoRA.
- Math presentation: recurrence, its equivalence to an online gradient step, the input-conditional-operator view, and a
  proposition on why it helps under shift — all rigorous, every symbol defined once.
- Writing: Kaiming style. Load-bearing sentences, no hedging, no AI voice.
- Results: in-dist parity + OOD robustness curve + causal isolation (TTT-on/off) + the fix-vs-mismatch ablation. All real
  or RED. Parity stated as parity.
- Experiments: distribution-shift suite is the home turf; design must actually support the OOD claim.

**Hard rules.** Never claim an unproven win. Negative/pending → \textcolor{red}{} table + TODO with exact run spec.
Cite DeltaNet/TTT-linear/Mamba/LoRA honestly — the novelty is the PEFT framing + consistency, not the recurrence per se.
No AI voice. Math must be elegant and complete.
