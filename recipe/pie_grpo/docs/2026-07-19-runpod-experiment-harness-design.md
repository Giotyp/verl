# RunPod experiment harness — Pie-GRPO vs verl+vLLM-GRPO (design spec)

**Date:** 2026-07-19  ·  **Status:** design approved, spec for review

## Context & goal

The Pie⇄verl GRPO loop is validated at 1.5B on the local 2× RTX 4090 box (no P2P). We want to
run it on **RunPod NVLink hardware** and answer two questions with hard numbers:

1. **How fast / how well** does the Pie-backed GRPO train (time + held-out performance)?
2. How does it compare to a **verl-native GRPO baseline** (verl's own vLLM SPMD rollout, same task)?

Deliverable: a set of scripts + instrumentation, all committed under
`recipe/pie_grpo/experiments/` in the verl fork, so a single clone provisions and runs everything.

## Locked decisions

| Decision | Choice |
|---|---|
| Env provisioning | Base RunPod image + **thin layer that builds its own venv** (does NOT use base torch) |
| GPU target | **2× A40 (48 GB, Ampere SM 8.6), NVLink**, DP=2 (no TP — Stage C stays deferred) |
| Baseline | **verl-native GRPO** (`main_ppo.py`, `adv_estimator=grpo`, native vLLM rollout) + registered HumanEval reward |
| Stats | Per-run `metrics.json` (common schema) + `compare.py` (table + plot) |
| Datasets | **Vendored parquets** (reproducible; RL-Heval is not a git repo) + prepare scripts for regen |
| Baseline metrics | From verl's native validation + metrics dict (not console parsing) |

## RunPod / A40 specifics (the version-handling crux)

- **torch/vLLM pins:** the CUDA-IPC weight-sync requires **trainer torch == driver torch**. The
  working env is **`torch 2.11.0+cu129`, `vLLM 0.21.0`, CUDA 12.9**. RunPod A40 PyTorch base
  images ship torch 2.1–2.8, which is **incompatible** — so the setup script creates a **fresh
  venv** (no `--system-site-packages`) and installs the exact pins from a captured
  `requirements.lock`, ignoring the base image's torch. torch cu129 wheels bundle their own CUDA
  runtime; the base image only needs a **CUDA-12.9-capable NVIDIA driver** (≈550+), which RunPod
  A40 pods have.
- **Version guard:** after install, assert `torch.__version__ == 2.11.0+cu129` and
  `vllm.__version__ == 0.21.0`; **fail loud** on mismatch (same philosophy as the Stage C guard) —
  a silent torch mismatch produces a broken CUDA-IPC handle layout, the worst kind of failure.
- **A40 vs 4090:** 48 GB (vs 24 GB) removes the cuda:3 OOM-transient confound entirely; NVLink/P2P
  means the FSDP cross-GPU collectives are no longer host-staged (the speedup we're measuring).
  Design stays colocated DP=2 (each rank + its engine replica on one GPU) — unchanged from local.
- **Pie build:** built from `pie-gt` source (their `features/RL` changes aren't in any release).
  Open item for implementation: confirm whether the `vllm` driver path needs the `driver-cuda`
  cargo feature (→ needs `nvcc`/CUDA toolkit + ~22-min build) or whether
  `driver-portable,driver-dummy` suffices for the runtime + weight-sync + vLLM bridge (→ fast
  build, no toolkit). Default to matching the local feature set; optimize if the lean build serves
  the vllm driver.

## Components (all under `recipe/pie_grpo/experiments/`)

### 1. `publish_repos.sh` — repo accessibility (user runs; needs GitHub auth)
Push `pie-rl` → `Giotyp/verl`, `features/RL` → `Giotyp/pie`; `gh repo edit --visibility public` on
both. Idempotent; prints the clone URLs the setup script expects.

### 2. `setup_runpod.sh` — env bootstrap (thin layer)
Ordered, idempotent, fail-loud:
1. System deps (git, build-essential, Rust toolchain, `cmake`/`ninja` + CUDA toolkit **iff** the
   `driver-cuda` build is needed).
2. Clone both public forks; checkout `pie-rl` / `features/RL`.
3. Build `pie` from `pie-gt` → on PATH (`build_pie_dev.sh` or `cargo install --path server`).
4. **Fresh venv** + install `requirements.lock` (torch 2.11.0+cu129, vLLM 0.21.0); editable-install
   `pie_driver_vllm`, `pie_client` (pie-gt), `verl` (fork). **Version guard.**
5. Vendor datasets into `experiments/data/` (parquets committed; prepare scripts for regen).
6. Build the `grpo` inferlet (bakery) → install.
7. Template `qwen-rl-config.toml` + `config.yaml` for the pod's 2 GPU indices (default `cuda:0,1`).
8. Print the exact commands to launch each experiment.

### 3. `metrics.py` + hooks in `train_grpo.py` — instrumentation
`RunMetrics` (dependency-free): a `phase(name)` context manager timing **gen / logprob / advantage /
update / weight_sync / eval** per step; records the held-out heval+mbpp curve, per-step
reward_mean/pg_loss, `torch.cuda.max_memory_allocated` peak per rank, and throughput (gen tokens/s
from returned token counts, steps/hr). Writes `metrics.json` incrementally + final. `train_grpo`
gains `--metrics-out PATH` (rank 0 writes). No behavior change when the flag is absent.

### 4. `baseline/` — verl-native GRPO
- `humaneval_reward_verl.py`: verl-interface reward wrapping the vendored `compute_reward`, keyed on
  `reward_model.ground_truth` (same scorer as the Pie run → identical reward semantics).
- `grpo_baseline_config.yaml` + `run_baseline.sh`: `main_ppo.py`, `adv_estimator=grpo`, **same**
  model (Qwen2.5-Coder-1.5B), parquets, and hyperparams (n_samples=8, lr 3e-6, temp 0.8, max_tokens
  512), FSDP + native vLLM rollout, 2 GPUs, held-out validation on the same test parquets.
- Metrics bridge: emit the **same `metrics.json` schema** from verl's native validation + per-step
  metrics dict + a wall-clock/timing wrapper around the fit loop.

### 5. `compare.py` — comparison
Ingest two `metrics.json`; print a table (total time, steps/hr, gen tokens/s, phase breakdown where
available, peak mem, final + best heval/mbpp) and save a plot (heval-curve overlay + timing bars).

### 6. `run_all.sh` — orchestrator
One command: setup (if needed) → Pie run → baseline run → compare; or run each stage alone.

## Common `metrics.json` schema (the comparability contract)

```json
{
  "run": {"backend": "pie|verl-vllm", "model": "...", "n_gpus": 2, "gpu": "A40",
           "num_steps": 50, "hyperparams": {"n_samples": 8, "lr": 3e-6, "temperature": 0.8}},
  "timing": {"total_s": 0, "per_step_s": [],
             "phase_totals_s": {"gen": 0, "logprob": 0, "adv": 0, "update": 0,
                                "weight_sync": 0, "eval": 0}},
  "throughput": {"gen_tokens_total": 0, "gen_tokens_per_s": 0, "steps_per_hr": 0},
  "memory": {"peak_gpu_gb_per_rank": []},
  "eval": [{"step": 0, "heval": 0.0, "mbpp": 0.0}],
  "train": {"reward_mean_per_step": [], "pg_loss_per_step": []}
}
```

**Comparable across backends:** `total_s`, `steps_per_hr`, `gen_tokens_per_s`, `peak_gpu_gb`, and
the `eval` curves. **Backend-specific (best-effort):** the `phase_totals_s` breakdown — verl's loop
has no distinct Pie-style `weight_sync` phase, so its breakdown is informative, not 1:1. The
headline comparison rests on the common metrics.

## Risks & mitigations

- **torch/vLLM drift on the base image** → fresh venv with pinned lock + a hard version guard.
- **Pie build cost/toolkit** → confirm the lean (no-`driver-cuda`) build serves the vllm driver;
  fall back to the full build if not.
- **NVLink actually wired on the RunPod pod** → `setup_runpod.sh` runs `nvidia-smi topo -m` and
  reports the interconnect (NVLink vs PCIe/P2P vs none) so the result is interpreted correctly.
- **Baseline fairness** → identical model/data/reward/hyperparams; only the rollout+sync backend
  differs. Same held-out parquets + same `compute_reward` scorer on both sides.
- **RL-Heval not a repo** → vendor the parquets + prepare scripts into `experiments/data/`.

## Verification / success criteria

1. `setup_runpod.sh` on a fresh A40 pod → `pie serve` boots (dummy weights) + `torchrun` Pie run
   reaches the baseline eval green.
2. Pie run reproduces the expected held-out heval band (~0.515 → ~0.66) on 2× A40 — sanity that the
   port is faithful, plus its `metrics.json`.
3. Baseline run trains green + emits a schema-valid `metrics.json`.
4. `compare.py` renders the table + plot from the two runs.
5. `nvidia-smi topo -m` output captured so the speedup is attributed to the right interconnect.

## Sequencing (for the implementation plan)

Instrumentation (3) → Pie config/scripts (2 partial) → baseline (4) → compare (5) → setup end-to-end
(2 full) → orchestrator (6) → publish (1). Datasets vendored early (needed by both runs).
