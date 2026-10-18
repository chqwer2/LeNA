# Claude session handoff — LeNA / FLoRA experiments

**Purpose:** resume this work in a fresh Claude Code session. Read this top-to-bottom, then
run the "Resume checklist". Everything below reflects state as of this session.

---

## 0. Goal (current)

**Run experiments on BOTH clusters (Kelvin2 primary + AutoDL 4090 secondary), on DIFFERENT tasks,
for multi-way empirical validation of the method.** The story/theory is settled enough; what decides
the paper's fate is whether the experiments show a **big, real gap** — not more narrative.

---

## 1. What this project is

FLoRA/**LeNA** = "Learned Nonlinear Adaptation": a LoRA variant that inserts a learnable nonlinearity
in the low-rank bottleneck. Paper draft: `~/Downloads/ICML___LeNA__Learning_Nonlinear_Adaptation.pdf`.
**It was REJECTED at ICML** (4 reviews: Reject/Weak-reject; Originality 2, Significance 1–2).

**Why rejected (the real problem):**
- Not novel: the pieces are each owned by 2024–25 papers the draft itself cites — **AFA-LoRA**
  (learnable activation), **AuroRA** (bottleneck nonlinearity, NeurIPS'25), **LoRAN** (sinusoid),
  **GainLoRA** (gate). None benchmarked against.
- Gains marginal (+0.5–1.5 avg), no seeds/std, **different param counts across methods** (comparison invalid).
- Code↔paper mismatch (paper's §3.5 gate is input-dependent `σ(θz)`; code is static `σ(θ)`).
- No insight into **when/why/where** nonlinearity helps (3 reviewers asked for the where-analysis).

## 2. The innovation pivot (what to build the paper on)

> **Rank and nonlinearity are two exchangeable currencies of adaptation, and nonlinearity is needed
> only at a sparse, learnable set of locations. LeNA learns this allocation with a provable LoRA fallback.**

Sharper mechanism framing: `Δ(x)=B·φ(Ax)` applies an **input-conditional** rank-r update — an
**implicit mixture of infinitely many LoRAs** (linear LoRA is forced to use ONE update for all inputs).
Metric = **`family_rank`** = effective rank of stacked per-input Jacobians (LoRA→1, nonlinear→>1).

**Honest bar for "attractive" (Significance ↑):** need a BIG param-efficiency gap (e.g. rank-2 LeNA ≈
rank-16 LoRA) on the **RIGHT battlefield** — NOT commonsense reasoning (marginal there), but a task
where linear low-rank genuinely struggles (**multi-task/multi-domain**, or **reasoning: GSM8K/MATH**).
If the gap won't open at low rank on real data → that's the ceiling; change task or stop.

## 3. Method changes already made (all local + rsynced to Kelvin2)

All in `Experiments/`:
- **`peft/tuners/lena/config.py`**: `lena_use_dora` (default False), `lena_norm_before_act` (False),
  `lena_gate_init` default −2.0.
- **`peft/tuners/lena/layer.py`**: DoRA decoupled to a flag; **unified gated interpolation**
  `h=(1-g)z+g·φ(z)` (one static selection gate, replaces old two-gate mess); `norm_before_act` wired
  (conditions φ input); dtype-consistency casts (φ/g→z.dtype, DoRA mag→y.dtype) for fp16 autocast;
  dropout wired; module-level **`lena_gate_l1()`** (sparsity penalty) + **`lena_gate_report()`** (where-map).
- **`gates.py`**: gate init honored (was hard-coded 1); added `Gate.value()`/`Gate.openness()`.
- **`activations.py`**: spline `use_gate="none"` now returns `y0+residual` (was missing y0); swish
  `init_t`→`init_gate`.
- **`Llama_Adaptation.py`**: `cache_dir` respects `HF_HOME`; `LENA_DEBUG=1` env → ~36-sample smoke;
  base loads **fp16** + dropped the whole-model float32 cast + trainable params→fp32 (fixed a 32G-GPU
  OOM); `LeNAGateL1Trainer` (adds L1 to loss); where-map JSON export after training; CLI flags
  `--lena_use_dora --lena_gate_l1 --lena_norm_before_act --lena_gate_init`.
- **`lena_probe/`** (GPU-free validation): `lena_core.py` (LoRAAdapter, LeNAAdapter, **BilinearAdapter**,
  `update_family_rank`, `make_target`, `fit`), `run_rank_substitution.py`, `run_mechanism_compare.py`,
  `run_mechanism_checks.py`, `plot.py`, `README.md`.

**Validated recipes (from probes, important):**
- Gate sparsity needs **hard gate + init OPEN (`--lena_gate_init 2.0`) + L1 prune**. `init-closed + L1`
  → dead-gate collapse (all gates shut). 
- `norm_before_act` keeps the code in spline range.
- **LeNA does NOT support 4-bit** (`LeNALinear` requires `nn.Linear`; quantized layers get skipped) →
  13B can't use `--quantize`; use A100 fp16. (7B fits 24G fp16 fine — 4090 is enough.)

## 4. CPU validation results (already run, on the mac)

`cd Experiments/lena_probe`:
- **`run_rank_substitution.py`**: linear LoRA plateaus at rel-MSE ≈0.52 at ALL ranks; LeNA-spline drops
  to ~0.02 → "nonlinearity substitutes for rank" holds in a controlled setting. (fig saved to
  `results/fig_rank_substitution.png`.)
- **`run_mechanism_compare.py`**: **LoRA family_rank=1** (one update for all inputs); **LeNA-spline
  family_rank up to 4 + near-perfect fit**; bilinear family_rank up to 10.5 but poor fit on the relu
  target. → **input-conditional / implicit-MoLoRA thesis VALIDATED; spline is the mechanism to push,
  bilinear is an ablation.**
- Caveat: synthetic targets are nonlinear-low-rank by construction → they prove the mechanism EXISTS,
  not that it MATTERS on real data. That's what the cluster experiments test.

## 5. Infrastructure

### Kelvin2 (QUB HPC — PRIMARY, free A100/V100)
- **Connect:** `ssh kelvin` needs a 2FA second factor → use a ControlMaster the USER opens once:
  `ssh -fN -o ControlMaster=yes -o ControlPersist=8h -o ControlPath=~/.ssh/cm-kelvin kelvin`
  then reuse: `ssh -o ControlPath=~/.ssh/cm-kelvin kelvin '<cmd>'`. **Network calls need the Bash
  sandbox disabled** (`dangerouslyDisableSandbox: true`).
- **Paths (home is 50G-capped → use scratch):** repo `/mnt/scratch2/users/hchen/FLoRA`; HF cache
  `/mnt/scratch2/users/hchen/hf_home` (**Llama-2-7b-hf 13G + all 8 datasets already cached**);
  logs `/mnt/scratch2/users/hchen/logs`; SLURM scripts `/mnt/scratch2/users/hchen/slurm`.
- **Env:** `source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh` → conda env `lena`
  (torch 2.5.1+cu121, transformers 5.14.1, datasets 2.20.0), HF_HOME, HF_TOKEN, **HF_HUB_DISABLE_XET=1**
  (Xet backend is broken here). Vendored `peft` imports fine from `Experiments/`.
- **SLURM:** batch GPU partitions = `k2-epsrc-gpu-a100`, `k2-epsrc-gpu-v100`, `k2-epsrc-gpu-a100mig`
  (interactive partition is **srun-only, blocks sbatch**). **Submit multi-partition** to schedule
  faster: `sbatch -p k2-epsrc-gpu-a100,k2-epsrc-gpu-v100,k2-epsrc-gpu-a100mig -t 00:20:00 --gres=gpu:1 --mem=48G <script>`.
  No `--account` needed. V100=32G, A100=40/80G, MIG slice=40G — all fit 7B fp16.

### AutoDL (RTX 4090 — SECONDARY, paid, GPU currently ON = billing)
- **Connect:** `bash ~/.claude/skills/autodl/scripts/ssh_run.sh '<cmd>'` (creds in
  `~/.config/autodl/ssh_env`; box = `ssh -p 34223 root@connect.westb.seetacloud.com`). Network calls
  need sandbox disabled.
- **State:** has **RTX 4090 24G** (enough for 7B fp16), `/root/autodl-tmp` 70G empty, conda `base` only.
  **NOT yet set up** — needs: env `lena`, repo, model. Tarball ready at `/tmp/flora_transfer.tgz`
  (1M, code with fixes) — transfer via scp/expect (no sshpass on mac; `expect` is at /usr/bin/expect).
- **API:** token in `~/.config/autodl/token`; `python ~/.claude/skills/autodl/scripts/autodl.py balance`
  works. **No instance on/off/release via API** (console only); the `autodl` skill (`~/.claude/skills/autodl/`)
  documents the cost model (无卡模式, auto-shutdown, 15-day rule). Balance ≈ ¥2143.
- **Cost note:** 4090 GPU is billing while on. `shutdown` via SSH stops GPU billing; power-on/release = console.

### Memory
Project memory (current machine): `~/.claude/projects/-media-cbtil3-WhiteSD-LeNA/memory/`.
Older mac-era memory lived at `/Users/haochen/.claude/projects/-Users-haochen-Documents-GitHub-FLoRA/memory/`.

### Access from the CB box (2026-07-22)
- **Kelvin2**: `ssh kelvin` via `~/.ssh/config` (port 55890, user `hchen`, `~/.ssh/id_ed25519`).
  Key auth passes but a **2FA second factor** remains, so the USER must open the ControlMaster once:
  `ssh -fN -o ControlMaster=yes -o ControlPersist=8h -o ControlPath=~/.ssh/cm-kelvin kelvin`.
- **AutoDL**: no sshpass/expect here, but `/usr/bin/python3` has **paramiko** -- drive it with a small
  wrapper. `ssh -p 34223 root@connect.westb.seetacloud.com`. The GPU is a **48G 4090**, not 24G.
- The `huggingface_hub` client **hangs** against hf-mirror on that box and crawls at ~1.3MB/s through
  `/etc/network_turbo`. Direct `curl` from `https://hf-mirror.com/<repo>/resolve/main/<file>` with the
  token as a bearer header gets 8-14MB/s -- pull into a plain dir and pass it as `--base_model`
  (`/root/autodl-tmp/llama2-7b`). Datasets still need `/etc/network_turbo` sourced.
- Two traps that each cost a cycle: `pkill -f <pat>` over SSH kills your own remote shell when the
  pattern appears in the command you sent; and SFTP-overwriting a *currently running* shell script
  makes bash execute garbage (it re-reads by byte offset).

## 6. Current job state  (updated 2026-07-22, working from the CB Linux box)

**Working machine changed.** Work now happens on `cbtil3-Precision-5820-Tower` at
`/media/cbtil3/WhiteSD/LeNA` (2x RTX A5000 24G, used ONLY for unit tests -- the user asked that
experiments stay on the two clusters). Access details and SSH/download traps are in the project
memory at `~/.claude/projects/-media-cbtil3-WhiteSD-LeNA/memory/`.

### Six bugs found and fixed (4 of them SILENT -- they produced numbers, just wrong ones)

1. **fp16/fp32 crash** in `layer.py`: the A projection's input was never cast. This is what killed
   every earlier smoke run.
2. **LayerNorm crash under autocast**: autocast returns Half from A's Linear even with fp32 weights
   and CUDA `layer_norm` does not promote, so the fp32 norm params blew up.
3. **The selection gate was never trained.** `Gate.param` was created lazily on the first forward,
   i.e. after the optimizer was built, so it stayed pinned at its init value and `--lena_gate_l1`
   penalized a frozen tensor. This voids the paper's central claim ("nonlinearity is needed only at
   a sparse *learnable* set of locations"). A stale diagnostic that still counted the long-removed
   `gate_after_a/gate_after_b` printed "gate=0" and hid it. Now `gate=160` on the real 7B.
4. **Trained adapters were never saved**: `adapter_model.safetensors` was 40 bytes / zero tensors,
   because PEFT selects by the registered prefix `"lena_"` and no LeNA parameter is named that way.
   Loading was broken too (`LeNAConfig` redeclared `peft_type` as an `init=False` field).
5. **LeNA-D was not DoRA**: `set_trainable_only_lena` never unfroze `magnitude`, so the DoRA
   magnitude stayed at init -- a fixed rescale. Exactly the variable E2 exists to measure.
6. **Dataset loading**: bare `piqa` no longer resolves and the commonsense sets are script-based;
   need `ybisk/piqa` + `trust_remote_code=True`. All 8 sets verified loadable (147,580 train rows).

**Consequence: every pre-2026-07-22 LeNA number is void** -- they were produced with a frozen gate
and a frozen magnitude. That plausibly explains the marginal +0.5-1.5 gains: the mechanism was not
learning.

### New capabilities
- `--seed` (split/shuffle/Trainer were all hardcoded to 42, so ">=3 seeds +/- std" was not even
  expressible) and `--max_train_samples` (caps the combined train set; the full 8-set is ~40h/config).
- `Experiments/tests/test_lena_dtype.py` -- 22 cases: fp16 base + fp32 trainables + autocast across
  every activation x DoRA x norm, plus "gate param must be in named_parameters() BEFORE any forward".
- `Experiments/tests/test_lena_save_load.py` -- full save -> from_pretrained round trip.
- `Experiments/analyze_runs.py` -- builds the comparison table: measured param counts (read from
  safetensors, so "matched params" is measured not claimed), per-dataset accuracy, mean +/- std
  across seeds, and gate %open.

**Params ARE matched**: at r=16 LeNA trains 28,090,528 vs LoRA's 28,049,408 -- **+0.15%**
(act 40,960 + gate 160). The reviewers' "different param counts => invalid comparison" is answerable.

### Running now
- **Kelvin2**: `lena_e1` (array 0-7: lora/lena x r in {4,8,16,32}, 8-set, `--max_train_samples 20000`,
  3 epochs, seed 1) and `lena_e3` (array 0-3: gate_l1 in {0,3e-4,1e-3,3e-3} at r=16). ~5.6h/job on a
  V100. Scripts: `/mnt/scratch2/users/hchen/slurm/lena_e1.sh`, `lena_e3.sh`.
- **AutoDL**: `chain5.sh` -> `run_e2.sh`, the DoRA-decouple ablation (LoRA / DoRA / LeNA / LeNA-D on
  boolq+piqa, r=16). LoRA done: boolq 0.7456, piqa 0.7524, **avg 0.7490**, 28,049,408 params.
  DoRA is ~3x slower per step than LoRA (it recomputes weight norms), so budget ~25h for all four.
- **Strategy**: one seed first to find WHERE the rank/param gap opens, then 3 seeds at that rank --
  not 24 jobs up front.

## 7. Experiment plan (the goal — both clusters, different tasks)

Full runner already on Kelvin2: `Experiments/run_lena_experiments.sh` (E1 rank-sweep, E2 DoRA-decouple,
E3 where-map/sparsity, E4 norm ablation, E5 activation ablation). To do, as SLURM jobs:
- **E1 rank-sweep iso-param** (LoRA vs LeNA r∈{4,8,16,32}) → param-accuracy Pareto + rank curve.
- **E2 DoRA decouple** (LoRA/DoRA/LeNA/LeNA-D) → isolate nonlinearity gain.
- **E3 where-map + sparsity** (hard gate, `--lena_gate_init 2.0`, `--lena_gate_l1` ∈ {0,3e-4,1e-3,3e-3})
  → "X% locations nonlinear" + `lena_gate_map.json` (plot with `lena_probe/plot.py --gate_map`).
- **E4 mechanism** spline vs bilinear vs linear on real data.
- **E5 MULTI-TASK gap** (the input-conditional battlefield): one LeNA vs one LoRA on a multi-task mix —
  gap should be BIG here (this is the thesis's home turf).
- **E6 different tasks for multi-way validation:** commonsense (8-set) on one cluster; a **reasoning
  task (GSM8K / MATH)** on the other — reasoning is where nonlinearity may actually bite (reviewers asked).
- **Everywhere: ≥3 seeds ± std; matched params.**
- **Deferred baselines** (reviewers named, must add for a real submission): AuroRA, AFA-LoRA, FourierFT.

**Cluster split idea:** Kelvin2 (A100/V100) = commonsense 8-set rank-sweep + where-map (heavy);
AutoDL 4090 = a different task (e.g. reasoning subset) in parallel → multi-way validation.

Example single run (commonsense, LeNA spline + sparsity, 7B):
```bash
ssh -o ControlPath=~/.ssh/cm-kelvin kelvin 'bash -l'
# then, in an sbatch script (partition/gres as §5), body:
source /mnt/scratch2/users/hchen/FLoRA/kelvin_env.sh
cd /mnt/scratch2/users/hchen/FLoRA/Experiments
python -u Llama_Adaptation.py --base_model meta-llama/Llama-2-7b-hf \
  --dataset "google/boolq piqa allenai/social_i_qa Rowan/hellaswag allenai/winogrande:winogrande_xl allenai/ai2_arc:ARC-Easy allenai/ai2_arc:ARC-Challenge allenai/openbookqa" \
  --methods lena --lena_activations spline --lena_flex_mode dim --lena_norm_before_act \
  --lena_gate_type sigmoid --gate_strength hard --lena_gate_init 2.0 --lena_gate_l1 1e-3 \
  --lora_r 16 --lora_alpha 32 --lora_target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --num_epochs 3 --batch_size 1 --cutoff_len 512 --device auto \
  --output_dir /mnt/scratch2/users/hchen/FLoRA/Experiments/runs/lena_r16
```

## 8. Resume checklist (do this first in a new session)

1. Ask the user to (re)open the Kelvin2 ControlMaster (§5) if `ssh -o ControlPath=~/.ssh/cm-kelvin kelvin true` fails.
2. Check smoke: `sacct -j 9442255` + tail its log (§6). If `SMOKE_EXIT=0` → pipeline works.
3. If smoke passed: launch **E1 (rank-sweep)** and **E3 (where-map)** as SLURM jobs on Kelvin2.
4. Set up **AutoDL 4090** as the 2nd worker (scp `/tmp/flora_transfer.tgz` or re-rsync; build env `lena`;
   download Llama-2-7b-hf to `/root/autodl-tmp/hf_home`; run a DIFFERENT task there). Remember it's billing.
5. Keep seeds ≥3, params matched. Export where-maps. Plot with `lena_probe/plot.py`.
6. **Decision gate:** if the rank/param gap doesn't open on real data → change battlefield to reasoning
   (GSM8K/MATH) or report honestly that the ceiling is reached.

## 9. Honest north star
The method is defensible but **not yet attractive**. Attractiveness = a big param-efficiency gap on the
right task + a clean where-map + the family_rank/implicit-MoLoRA insight. Chase the big number; don't
polish the mechanism. If the number doesn't come, that's the answer.
