# Code (this session's TTT-branch + eval scripts)

Standalone scripts (frozen Llama-2-7B, PEFT branches; base data = GSM8K). See `../ttt_branch_design.md`,
`../eval_protocol.md` for the design and honest results.

| file | what |
|---|---|
| `ttt_branch.py`  | TTT parallel-branch, attn-only (baseline). TwoPathTTT = causal SwiGLU inner-TTT + local conv. |
| `ttt_branch2.py` | + WrappedMLP (`--targets attn,mlp`), batched fair eval option. |
| `ttt_branch3.py` | ablation: `--inner_lr` (TTT on/off), `--mid` (SwiGLU width) — fitting diagnostic. |
| `ttt_branch4.py` | **v4 strengthened**: learnable per-layer inner_lr + per-channel gate + enriched core (r→2r→r, zero-init). Best branch (0.30). |
| `ttt_lb.py`      | LoRA + branch joint (naive). |
| `ttt_c.py`       | **C design**: LoRA + do-no-harm gated branch (per-channel gate + L1). Result 0.24/0.20 (failed). |
| `eval_fast.py`   | batched, left-padded, `first-####` fair extractor + diverse subsets (~9× faster). THE trusted eval. |
| `eval_branch.py` | earlier per-sample branch eval (first-####). |
| `reeval_all.sh`  | re-score all saved adapters under the fair protocol (corrected the rank curve). |
| `run_rankfill.sh`, `run_lowdata.sh`, `rundiag.sh` | SLURM sweeps (rank curve fill, low-data, ablation diagnostic). |
| `b2full/b4full/cfull/lbfull*.sh`, `*smoke.sh` | SLURM launch + smoke scripts per variant. |

Eval caveat: branch (cumulative causal state) must be decoded `bs=1` — left-pad batching pollutes the state. Static
adapters (LoRA/LeNA/IQ) are unaffected and use the batched path.
