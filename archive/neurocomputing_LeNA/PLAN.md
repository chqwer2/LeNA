# PLAN — how to write it, how to do it (Neurocomputing, LeNA + IQ-LoRA + evaluation protocol)

The single roadmap. STANDARD.md = the bar; REVIEW.md = current critique; EXPERIMENTS_TODO.md = run specs; this ties them
into an ordered path to submission. Kaiming style throughout: every sentence load-bearing; parity stated as parity.

## 0. The story (locked — do not drift)
Static LoRA is linear + input-independent. We study two orthogonal expressiveness axes — nonlinearity (LeNA framework,
instantiated with the *prior* AuroRA activation) and input-dependence (IQ-LoRA). **Finding:** at matched budget both match
LoRA; the literature's contrary "wins" are **evaluation artifacts**. Contribution = (i) unified 1-DOF framework,
(ii) a rigorous, proved protocol removing 3 artifacts, (iii) an honest map. *Parity, measured correctly, is the point.*

## 1. How to WRITE (section by section — status → what to do)
| § | status | what to do |
|---|---|---|
| Title/Abstract | draft | DECIDE retitle to lead with the protocol (recommended). Keep abstract honest (AuroRA = prior). |
| 1 Intro | done | Contributions + "parity is the contribution" in. Only trim on final pass. |
| 2 Related Work | draft, \todocite | Replace \todocite with real refs (bib). 3 paragraphs already structured. |
| 3 Method | done (math) | Unified Eqs 1–4; Prop (IQ) proved; AuroRA Eq.3 cited. IF author names a novel LeNA component, add it; else keep as "studied". |
| 4 Protocol | done (math) | Definition + Lemma 1 (proved). This is the backbone — do not weaken. Prose is tight. |
| 5 Experiments | design done, data RED | Fill RED tables/figures from §2 below. Every table caption already specifies its cells. |
| 6 Conclusion | done | scope + Lemma-1 takeaway in. |

Writing rules: define each symbol once; no number without a source; never call AuroRA "ours"; no AI hedging.

## 2. How to DO the experiments (ordered by ROI — see EXPERIMENTS_TODO.md for exact specs)
**Do P1 first — it is mostly EVAL-ONLY on existing checkpoints (almost no GPU) and produces the paper's core evidence.**
1. **E3a flip table** (`tab:flip`) — re-score every saved checkpoint under BOTH E_last and E_first. No training. *The punchline.*
2. **E3b nHASH figure** (`fig:nhash`) — count `####` blocks/completion, LoRA vs AuroRA. Evidence for Lemma 1.
3. **E3c template table** (`tab:template`) — GSM-Symbolic first-N vs diverse-N. One eval pass.
4. **E1 breadth** (`tab:datasets`) — OOD eval of trained LoRA/AuroRA/IQ on SVAMP/ASDiv/GSM-Sym(div)/GSM-Plus. Eval-only.
5. **Fig3 money figure** + **Fig1 rank curve** — derived from the above, no new runs.
Then, only if reviewers need more:
6. **E0** IQ under protocol (train), **E2** DoRA/AdaLoRA/Adapter/IA3 (train), **E4/E5** ablations (train).

Tooling (on cluster, FLoRA/Experiments; also mirrored in ../designs/code/): `eval_fast.py` (batched first-#### extractor),
`reeval_all.sh` (re-score all adapters). Branch adapters need bs=1; static adapters use the batched path.

## 3. Author decisions blocking finalization
- **LeNA novelty**: is anything in LeNA genuinely ours (framework? a specific activation)? Determines whether §3 claims a
  component or stays a "study of prior expressive adapters". Default = study (safe).
- **Retitle** to lead with the protocol? (recommended.)
- Real citations for \todocite (need a bib file).

## 4. Submission path (do in order)
1. Run P1 (steps 1–5) → fills all core RED. ~1 eval sweep.
2. Fill bib (\todocite → refs) + apply author decisions.
3. Final Kaiming pass (trim, consistency, figure polish) using REVIEW.md as the checklist.
4. Compile on Overleaf; check 4-column format + length for Neurocomputing.
5. Optional P2/P3 (E0/E2/E4/E5) if a reviewer asks for more breadth.
```
Fastest to a submittable draft = step 1 (P1 eval sweep) + step 2 (bib). Everything else is polish or on-demand.
```
