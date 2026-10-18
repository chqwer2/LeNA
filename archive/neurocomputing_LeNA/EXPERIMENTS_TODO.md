# Experiments to run — one row per RED table/figure in main.tex. Exact specs. Protocol = E_first, batched (eval_fast.py),
# diverse subsets. Training: Llama-2-7B, targets {q,k,v,up,down}, 3 epochs, bs2, lr3e-4, 7k train, >=3 seeds.
# Priority: P1 = reject-risk closer / cheap ; P2 = strengthens ; P3 = polish.

| ID  | Paper object            | What to run                                                                                          | Cost | Prio |
|-----|-------------------------|------------------------------------------------------------------------------------------------------|------|------|
| E0  | Table `tab:iqrank`      | Train IQ (λ=0.5) + IQG at r=2,8,16, 3 seeds; budget-matched LoRA r'=1.5r. Eval GSM8K E_first.         | med  | P1   |
| E3a | Table `tab:flip`        | Re-score EVERY existing checkpoint (all methods×datasets) under BOTH E_last and E_first. No training. | low  | P1   |
| E3b | Fig `fig:nhash`         | Count '####' blocks per completion (LoRA vs AuroRA), GSM8K test. From existing generations.           | low  | P1   |
| E3c | Table `tab:template`    | GSM-Symbolic first-40 vs diverse-40/80, mean±sd, LoRA & AuroRA. One eval pass.                        | low  | P1   |
| E1  | Table `tab:datasets`    | OOD eval of trained LoRA/AuroRA/IQ on GSM-Sym(div)/SVAMP/ASDiv/GSM-Plus, E_first. Eval-only (no train).| med  | P1   |
| E2  | Table `tab:baselines`   | Train DoRA, AdaLoRA, Houlsby-Adapter, IA3 at matched budget; eval GSM8K + GSM-Sym E_first.            | high | P2   |
| Fig3| Fig `fig:money`         | Bar chart from E3a: apparent gap under E_last vs E_first per method×dataset. Derived from E3a.         | low  | P1   |
| Fig1| Fig `fig:rankcurve`     | Plot Table 1 (+IQ from E0) with error bars. Derived, no new runs.                                     | low  | P2   |
| E4  | Table `tab:ablate-act`  | LeNA framework, φ∈{identity,GELU,spline,AuroRA} at r=2,8, GSM8K E_first.                              | med  | P2   |
| E5  | Table `tab:ablate-iq`   | IQ raw vs IQG, λ∈{0,.25,.5,1}, r=8, GSM8K E_first.                                                     | med  | P3   |

## Fastest path to a submittable draft
Do the P1 EVAL-ONLY items first — they reuse existing checkpoints, need almost no GPU, and produce the paper's core
evidence: E3a (flip table), E3b (nHASH), E3c (template), E1 (breadth eval), Fig3+Fig1 (derived). That fills most RED with
one eval sweep. Only E0/E2/E4/E5 need new training.

## Tooling (already on cluster)
- eval_fast.py = batched E_first extractor (Experiments/). reeval_all.sh = re-score all saved adapters.
- Existing checkpoints: gsmB_r{1,2,4,16,32}_{aurora,lora}_s*, r8_iq*, ttbranch* (branch out of scope), r32_lora*.
- Base activation defs: peft/tuners/lena/activations.py (AuroRA=CompAuroRA, IQ via LENA_IQ env in run scripts).

## Non-experiment blockers (author)
- LeNA novelty decision (framework vs a specific activation) — changes Method framing.
- Retitle to lead with the protocol? (recommended)
- Fill \todocite with real refs (bib file).
