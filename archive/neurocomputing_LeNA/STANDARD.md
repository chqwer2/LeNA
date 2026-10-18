# 10/10 Neurocomputing bar for this paper (the fixed standard we hold every loop)

**Story (locked, honest).** Static LoRA applies a *linear, input-independent* low-rank update. We ask: does adding
expressiveness along two orthogonal axes — (i) **nonlinearity** in the low-rank code (LeNA) and (ii) **input-dependence**
of the update (IQ-LoRA) — improve adaptation? Our answer is a *disciplined negative-leaning* result: under a **rigorous,
artifact-free evaluation protocol**, both expressive variants match well-tuned LoRA at matched parameter budget; the
apparent gains reported by naive protocols are **evaluation artifacts** (answer-extraction bias, repetition degeneration,
benchmark-template collapse). Contribution = (a) a unified formulation of expressive low-rank adaptation, (b) a rigorous
evaluation protocol that removes three named artifacts, (c) an honest empirical map showing parity + *why* prior "wins"
were illusory. This turns "parity" from a weakness into the point.

**Reviewer bar (score each 1-10 every loop; target all ≥9):**
- Writing: Kaiming style — every sentence load-bearing, logical, no filler, no AI hedging. Not verbose.
- Novelty: unified framework + the evaluation contribution must be crisp and defensible.
- Presentation: figures/tables self-contained; notation consistent; claims traceable to a number.
- Results: every number real or RED (undone). No number without a source. Parity stated as parity.
- Experiments: breadth (multi-dataset, multi-baseline, ablation) planned; gaps RED-tabled with designs.
- Math: complete, rigorous, elegant. Define every symbol once. No hand-waving.

**Hard rules.** Never write parity/retracted results as a win. Undone experiments -> \textcolor{red}{} tables + a TODO
with exact run spec. No fabricated citations (mark \todocite). No "AI voice".
