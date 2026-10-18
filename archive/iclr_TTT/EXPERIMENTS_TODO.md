# Experiments to run — TTT ICLR paper. The results ARE the paper; do these in order.

## STEP 0 (code, gating everything): build consistent-TTT
Implement Eq.(2) recurrence: S_t = (1-α)S_{t-1} + η(v_t - S_{t-1}k_t)k_t^T, learnable α (sigmoid), η; output o_t=S_t q_t,
up-proj B, per-channel gate γ. KEY property: SAME recurrence at train and generation (online, left-to-right), O(N),
chunk-parallelizable. Start from ../designs/code/ttt_branch4.py; REPLACE the prefix-mean /cnt closed-form state with this
recurrence. Smoke: α moves, γ moves, no deadlock (γ init 1, B init 0). Verify train==generation state on a toy sequence.

## STEP 1 — Table `tab:indist`: in-distribution parity (P1, gating)
Train consistent-TTT vs LoRA at matched budget, GSM8K, >=3 seeds. Eval first-#### (now batchable — recurrence is
state-consistent, but confirm padding handling). MUST reach >= ~LoRA (0.33) to clear the naive baseline (0.24-0.30).

## STEP 2 — Table `tab:diag`: fix vs naive (P1)
Same budget: naive prefix-fit branch (ttt_branch4) vs consistent recurrence, GSM8K + OOD-avg. Shows the mismatch fix is
what turns 0.24 -> parity. This is the technical-core evidence.

## STEP 3 — Table `tab:ood` + Fig `fig:money`: robustness vs shift (P1, THE thesis)
Train on GSM8K; eval on a shift ladder: GSM8K(none) < GSM-Symbolic(diverse) < GSM-Plus/perturbed < numeric-range/length
OOD. Report acc + LoRA->ours gap per level. CLAIM: gap grows monotonically with shift. Fig 1 = acc vs shift, two curves.

## STEP 4 — Table `tab:ablate`: causal isolation (P1)
consistent-TTT (η>0) vs state-frozen ablation (η=0, static branch, equal params) vs LoRA, on the OOD suite. Proves the
robustness comes from test-time ADAPTATION, not extra parameters.

## STEP 5 (P2, on reviewer request): scale + generality
Second model (Mistral-7B / Llama-3-8B), second task family, decoding-cost measurement (O(N) claim), chunkwise-parallel wall
clock.

## Tooling
Fair eval = ../designs/code/eval_fast.py (first-####, batched). Cluster: FLoRA/Experiments. Reuse GSM-Symbolic diverse
subset + GSM-Plus. Success bar for submission: STEP 1 (parity) AND STEP 3 (monotone OOD gap) both black.
