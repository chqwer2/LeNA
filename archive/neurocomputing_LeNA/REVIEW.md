# Reviewer pass #2 (10/10 standard). CRITICAL finding at top.

## 🔴 BLOCKER #0 (CRITICAL, novelty/ethics) — found by reading the code
The "aurora" results (Table 1) come from `lena_activation='aurora'` = **AuroRA, a PUBLISHED activation
(NeurIPS'25, arXiv:2505.18738): sigma(Z)=tanh(H tanh Z)+w_s*spline(Z)**. It is NOT this project's contribution.
Code comment (activations.py, FlexRankMixC): *"AuroRA beats LeNA on GSM8K (0.287 vs 0.224)"* — AuroRA and LeNA are
DIFFERENT; the strong rank-curve numbers are AuroRA-in-LoRA, i.e. prior work applied to LoRA.
Also: `loran` = LoRAN (EMNLP'24), another published activation. `spline/fourier/...` are standard.

**Consequence.** Presenting "LeNA (ours)" with the AuroRA numbers = claiming prior work as novel. This must be fixed
before anything else. Two honest resolutions:
- (A) **Reframe around the evaluation protocol as the PRIMARY contribution** (this IS the user's + novel). Nonlinear
  low-rank instances — incl. the *published* AuroRA and the input-dependent IQ-LoRA — are CASE STUDIES showing that even a
  SOTA nonlinear activation matches LoRA under fair eval, and that prior gains were artifacts. Cite AuroRA/LoRAN properly.
  This is stronger and honest: "we debunk, with a protocol, apparent PEFT gains — including a NeurIPS'25 method."
- (B) If LeNA has a genuinely NOVEL activation of the user's own (candidate: `rankmixc` = compressed-bottleneck cross-rank
  nonlinearity, or `tanhres`, or the two-path idea), center THAT and report ITS numbers — but the code comment says
  rankmixc/LeNA underperform AuroRA and a->0 at r=8 (doesn't bind). Needs the user to state what is genuinely theirs.

**ACTION NEEDED FROM USER:** what is LeNA's own novel component? the framework? a specific activation (which)? Until
answered, the paper CANNOT claim the nonlinearity as a contribution. Default to resolution (A).

## Scores now
| axis | score | why |
|---|---|---|
| Writing | 6/10 | voice right; proofs still stubs |
| Novelty | 4/10 | dropped — the "nonlinear" novelty is largely prior work (AuroRA). Real novelty = the protocol + honest map. |
| Presentation | 6/10 | tables ok; needs figures |
| Results | 5/10 | real curve exists but it's AuroRA-in-LoRA; IQ missing under protocol |
| Experiments | 4/10 | single task/model, no baselines |
| Math | 7/10 | AuroRA now definable exactly (below); IQ operator clean; props still to prove |

## AuroRA exact form (for Method, cite arXiv:2505.18738)
phi(z)=tanh(H tanh(z)) + w_s ⊙ spline(z), H in R^{r~×r~} cross-rank, w_s per-channel, spline = learned knots. Not an exact
LoRA fallback (phi(0)!=0 in general). AuroRAG variant adds z + a[...] with a=0 => LoRA fallback.

## Revision targets THIS/next loop (continue, don't restart)
1. Reframe title/abstract/intro: PRIMARY = evaluation protocol + honest finding; nonlinearity (AuroRA) and input-dependence
   (IQ) are the two *studied* axes, AuroRA cited as prior work. (writing only)
2. Method: write AuroRA exactly with citation; write IQ operator; prove IQ-not-in-static-family (1 line).
3. Keep breadth/baselines/IQ-under-protocol RED (EXPERIMENTS_TODO).

## Iteration 3 done (pure writing, no restart)
- [x] AuroRA reframed as prior work (Eq.3 + \cite); Table1/Table2 relabelled; abstract fixed. (blocker #0 handled -> resolution A)
- [x] Prop (Input-dependence) stated + PROVED for IQ-LoRA. Math axis now ~8/10.
- [x] Parameter-accounting table added (budget-matched control defined).
- [x] Related Work drafted: static low-rank / expressive adapters / evaluation — prior work positioned, citations \todocite (not fabricated).
Updated scores: Writing 7, Novelty 5 (protocol carries it), Presentation 6, Results 5, Experiments 4, Math 8.

## Next loop targets (still writing-only; keep experiments RED)
1. Formalize Protocol Sec fully: definitions for degeneration-invariance done; add a short lemma that $E_{\text{last}}$ can
   err by an unbounded amount under repetition (1-line construction). Write Artifact-2/3 prose tightly.
2. Intro: one sentence "why measuring parity correctly is itself the contribution."
3. Abstract: ensure title still matches (title still says LeNA+IQ; consider retitle to lead with the protocol) — FLAG for author.
4. STILL BLOCKING on experiments: IQ-under-protocol numbers (E0), breadth (E1), baselines (E2), artifact figures (E3). RED.
5. STILL PENDING author input: what is genuinely novel in LeNA (framework vs a specific activation).

## Iteration 4 done (writing-only)
- [x] Protocol Sec now rigorous: Definition (degeneration-invariance) + Lemma 1 (E_last has method-dependent bias
  p_M*Pr[correct], unbounded; E_first removes it) WITH proof. This is the paper's mathematical backbone for the
  "evaluation-as-contribution" claim. Novelty of the protocol axis now defensible.
- [x] Artifact 2 rewritten to instantiate Lemma 1 (baseline-specific p_M); Artifact 3 tightened. Evidence stays RED (figures).
Updated scores: Writing 7.5, Novelty 6 (Lemma makes the protocol a real contribution), Presentation 6, Math 8.5.

## Iteration 5 done — WRITING SATURATED, loop stopped
- [x] Conclusion written (scope + Lemma-1 takeaway). [x] Protocol caveat finished (left-pad/causal-state).
- [x] Intro findings fixed: IQ no longer over-claimed (marked RED-to-run); AuroRA not called "ours"; parity-is-the-
  contribution sentence added.
FINAL scores (writing complete, experiments pending): Writing 8, Novelty 6 (protocol+Lemma carry it), Presentation 6.5,
Math 8.5, Results 5 (real curve + artifact table only), Experiments 4 (RED).

## Draft is as complete as writing alone allows. Remaining work is NOT writing:
### A. Experiments (cluster + author) — the reject-risk closers, see EXPERIMENTS_TODO.md
- E0 IQ-LoRA/IQG under E_first protocol (unblocks the "+IQ" in the title)
- E1 breadth: GSM-Symbolic(diverse)/SVAMP/ASDiv/+1  x {LoRA,AuroRA,IQ}
- E2 baselines: DoRA/AdaLoRA/Adapter/IA3 at matched budget
- E3 artifact figures: nHASH histogram (Lemma-1 evidence), first-N vs diverse-N table
### B. Author decisions (cannot proceed without)
- What is genuinely novel in LeNA? (framework vs a specific activation). Determines whether the paper is
  "study of prior expressive adapters + protocol" (safe, current default) or centers an original component.
- Retitle to lead with the protocol? (recommended given where the novelty actually sits)
### C. Mechanical (needs real refs, do with a bib file)
- Replace all \todocite with real citations (LoRA, DoRA, AdaLoRA, adapters, IA3, AuroRA=2505.18738, LoRAN, GSM8K, GSM-Symbolic).
Do NOT spin more writing-only loops; they add polish, not substance.
