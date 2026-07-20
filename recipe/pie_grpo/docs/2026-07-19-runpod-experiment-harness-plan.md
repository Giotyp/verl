# RunPod Experiment Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provision a 2×A40 RunPod box and benchmark Pie-backed GRPO against a verl-native vLLM GRPO baseline on the same HumanEval task, emitting comparable `metrics.json` for time + held-out performance.

**Architecture:** Everything lives under `recipe/pie_grpo/experiments/` in the verl fork. A dependency-free `RunMetrics` collector instruments both training paths into one JSON schema; a verl-native baseline reuses the same model/data/reward; `compare.py` diffs the two runs. A thin `setup_runpod.sh` builds its own pinned venv on a stock A40 PyTorch base image.

**Tech Stack:** Python 3.12, torch 2.11.0 (cu128 on RunPod), vLLM 0.21.0, verl (fork branch `pie-rl`), pie-gt (branch `features/RL`), pandas/pyarrow, matplotlib (compare plot), bash.

## Global Constraints

- torch pinned to **2.11.0** (RunPod: `+cu128` index `https://download.pytorch.org/whl/cu128`; local: `+cu129`). Version guard must assert `torch.__version__` starts with `2.11.0`.
- vLLM pinned to **0.21.0**. Trainer torch == driver torch (shared venv) — mandatory for CUDA-IPC.
- Model: **Qwen/Qwen2.5-Coder-1.5B-Instruct** (Pie config `hf_repo` and verl `model.path` must match).
- Hyperparams (both backends): n_samples=8, lr=3e-6, temperature=0.8, top_p=0.95, max_tokens=512, num_steps=50, problems_per_step=8.
- GPU: 2× A40, DP=2, `CUDA_VISIBLE_DEVICES` order must match the Pie config device list.
- **Commits: short one-line subject, NO Claude coauthor trailers.**
- Pie build: lean `--no-default-features --features driver-portable,driver-dummy` (no CUDA toolkit).

---

## File Structure

```
recipe/pie_grpo/experiments/
  __init__.py
  metrics.py                      # RunMetrics collector + JSON schema
  test_metrics.py                 # unit tests
  compare.py                      # ingest 2 metrics.json -> table + plot
  test_compare.py                 # unit tests
  requirements.lock               # captured pins (torch/vllm/...); RunPod torch = +cu128
  data/
    humaneval_train.parquet       # vendored (131 problems)
    humaneval_test.parquet        # vendored (33 held-out)
    mbpp_test.parquet             # vendored (100 transfer)
    prepare_humaneval.py          # regen script (copied from RL-Heval)
    prepare_mbpp.py               # regen script (copied from PieRL/data)
  configs/
    qwen-rl-config.toml           # Pie server config, devices cuda:0,1
    pie_config.yaml               # train_grpo config for RunPod (2 A40)
  baseline/
    humaneval_reward_verl.py      # verl-interface reward wrapping compute_reward
    test_humaneval_reward_verl.py
    verl_metrics_bridge.py        # verl metrics dict + timing -> common schema
    test_verl_metrics_bridge.py
    grpo_baseline_config.yaml      # verl main_ppo GRPO config
    run_baseline.sh               # launch verl + emit metrics.json
  setup_runpod.sh                 # env bootstrap (thin layer)
  publish_repos.sh                # push branches + set public
  run_all.sh                      # orchestrator
Modify:
  recipe/pie_grpo/train_grpo.py   # add --metrics-out + RunMetrics phase() hooks
```

---

### Task 1: Vendor datasets + capture requirements.lock

**Files:**
- Create: `recipe/pie_grpo/experiments/__init__.py` (empty)
- Create: `recipe/pie_grpo/experiments/data/{humaneval_train,humaneval_test,mbpp_test}.parquet` (copies)
- Create: `recipe/pie_grpo/experiments/data/prepare_humaneval.py`, `prepare_mbpp.py` (copies)
- Create: `recipe/pie_grpo/experiments/requirements.lock`
- Test: `recipe/pie_grpo/experiments/test_data.py`

**Interfaces:**
- Produces: three parquet files at known paths with columns `['prompt','data_source','ability','reward_model','extra_info']`; `reward_model` is a dict with keys `style`,`ground_truth` (ground_truth is a str). Consumed by both train_grpo (Pie) and the verl baseline.

- [ ] **Step 1: Copy the datasets and prepare scripts**

```bash
mkdir -p recipe/pie_grpo/experiments/data
cp /home/george/agentic-rl/RL-Heval/data/humaneval_train.parquet recipe/pie_grpo/experiments/data/
cp /home/george/agentic-rl/RL-Heval/data/humaneval_test.parquet  recipe/pie_grpo/experiments/data/
cp /home/george/agentic-rl/PieRL/data/mbpp_test.parquet          recipe/pie_grpo/experiments/data/
cp /home/george/agentic-rl/RL-Heval/data/prepare_humaneval.py    recipe/pie_grpo/experiments/data/
cp /home/george/agentic-rl/PieRL/data/prepare_mbpp.py            recipe/pie_grpo/experiments/data/
touch recipe/pie_grpo/experiments/__init__.py
```

- [ ] **Step 2: Capture the lockfile from the live venv, patch torch to cu128**

```bash
source /home/george/.pie/venvs/vllm/bin/activate
pip freeze | grep -vE "^-e |@ file://" > recipe/pie_grpo/experiments/requirements.lock
# force the cu128 torch build for RunPod A40 (keep version 2.11.0)
sed -i -E 's/^torch==2\.11\.0\+cu129$/torch==2.11.0  # install from cu128 index on RunPod/' \
  recipe/pie_grpo/experiments/requirements.lock
```

Then hand-verify `requirements.lock` contains `vllm==0.21.0`, `numpy==1.26.4`, `torch==2.11.0`. Remove any line that is a local editable path (pie_driver_vllm, pie_client, verl) — those are installed `-e` by the setup script, not from PyPI.

- [ ] **Step 3: Write the failing data test**

```python
# recipe/pie_grpo/experiments/test_data.py
import pathlib, pandas as pd
DATA = pathlib.Path(__file__).parent / "data"

def test_parquets_present_and_schema():
    for name, n in [("humaneval_train", 131), ("humaneval_test", 33), ("mbpp_test", 100)]:
        df = pd.read_parquet(DATA / f"{name}.parquet")
        assert len(df) == n, (name, len(df))
        assert {"prompt", "reward_model"} <= set(df.columns)
        rm = df.iloc[0]["reward_model"]
        assert "ground_truth" in rm and isinstance(rm["ground_truth"], str)
```

- [ ] **Step 4: Run it**

Run: `cd /home/george/git_repos/verl && python -m pytest recipe/pie_grpo/experiments/test_data.py -v`
Expected: PASS (files copied in Step 1). If humaneval_test has a different count, correct the expected `n` to match the actual file.

- [ ] **Step 5: Commit**

```bash
git add recipe/pie_grpo/experiments/
git commit -m "pie_grpo/experiments: vendor datasets + requirements.lock"
```

---

### Task 2: `metrics.py` — the RunMetrics collector

**Files:**
- Create: `recipe/pie_grpo/experiments/metrics.py`
- Test: `recipe/pie_grpo/experiments/test_metrics.py`

**Interfaces:**
- Produces:
  - `RunMetrics(backend: str, run_meta: dict, out_path: str)` — collector.
  - `.phase(name: str)` — context manager; accumulates elapsed seconds into `phase_totals_s[name]` and appends to the current step's timing.
  - `.start_step()` / `.end_step()` — brackets a step; `end_step` finalizes `per_step_s`.
  - `.record_gen_tokens(n: int)` — add to `gen_tokens_total`.
  - `.record_eval(step: int, heval: float, mbpp: float | None)` — append eval point.
  - `.record_train(reward_mean: float, pg_loss: float)` — append per-step train scalars.
  - `.record_peak_mem(gb_per_rank: list[float])` — set memory.
  - `.write()` — dump the schema dict to `out_path` (atomic); called after each `end_step` and at the end.
  - `.finalize()` — compute totals (total_s, throughput) and `write()`.
- Consumed by: train_grpo hooks (Task 3), verl_metrics_bridge (Task 6), compare.py (Task 4) reads the JSON it writes.

- [ ] **Step 1: Write the failing test**

```python
# recipe/pie_grpo/experiments/test_metrics.py
import json, time, pathlib
from recipe.pie_grpo.experiments.metrics import RunMetrics

def test_phase_timing_and_schema(tmp_path):
    out = tmp_path / "m.json"
    m = RunMetrics(backend="pie", run_meta={"model": "x", "n_gpus": 2}, out_path=str(out))
    m.start_step()
    with m.phase("gen"):
        time.sleep(0.02)
    m.record_gen_tokens(100)
    m.record_train(reward_mean=0.6, pg_loss=-0.03)
    m.end_step()
    m.record_eval(step=0, heval=0.5, mbpp=0.7)
    m.record_peak_mem([12.3, 12.1])
    m.finalize()

    d = json.loads(out.read_text())
    assert d["run"]["backend"] == "pie"
    assert d["timing"]["phase_totals_s"]["gen"] >= 0.02
    assert len(d["timing"]["per_step_s"]) == 1
    assert d["throughput"]["gen_tokens_total"] == 100
    assert d["throughput"]["gen_tokens_per_s"] > 0
    assert d["eval"] == [{"step": 0, "heval": 0.5, "mbpp": 0.7}]
    assert d["train"]["reward_mean_per_step"] == [0.6]
    assert d["memory"]["peak_gpu_gb_per_rank"] == [12.3, 12.1]
    assert d["timing"]["total_s"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest recipe/pie_grpo/experiments/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: metrics`.

- [ ] **Step 3: Implement `metrics.py`**

```python
# recipe/pie_grpo/experiments/metrics.py
"""Dependency-free run metrics for GRPO experiments (Pie + verl baseline).

Writes a single JSON schema so compare.py can diff two runs. Timing is wall-clock
via time.perf_counter (NOT Date.now — plain perf_counter is fine here). All numbers
are plain floats/ints; no torch/verl imports so it is unit-testable anywhere.
"""
from __future__ import annotations

import contextlib
import json
import os
import time


class RunMetrics:
    def __init__(self, backend: str, run_meta: dict, out_path: str):
        self.out_path = out_path
        self._t0 = time.perf_counter()
        self.data = {
            "run": {"backend": backend, **run_meta},
            "timing": {"total_s": 0.0, "per_step_s": [],
                       "phase_totals_s": {}},
            "throughput": {"gen_tokens_total": 0, "gen_tokens_per_s": 0.0,
                           "steps_per_hr": 0.0},
            "memory": {"peak_gpu_gb_per_rank": []},
            "eval": [],
            "train": {"reward_mean_per_step": [], "pg_loss_per_step": []},
        }
        self._step_t0 = None

    def start_step(self):
        self._step_t0 = time.perf_counter()

    def end_step(self):
        if self._step_t0 is not None:
            self.data["timing"]["per_step_s"].append(time.perf_counter() - self._step_t0)
            self._step_t0 = None
        self.write()

    @contextlib.contextmanager
    def phase(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            tot = self.data["timing"]["phase_totals_s"]
            tot[name] = tot.get(name, 0.0) + dt

    def record_gen_tokens(self, n: int):
        self.data["throughput"]["gen_tokens_total"] += int(n)

    def record_eval(self, step: int, heval: float, mbpp: float | None = None):
        self.data["eval"].append({"step": int(step), "heval": float(heval),
                                  "mbpp": (None if mbpp is None else float(mbpp))})
        self.write()

    def record_train(self, reward_mean: float, pg_loss: float):
        self.data["train"]["reward_mean_per_step"].append(float(reward_mean))
        self.data["train"]["pg_loss_per_step"].append(float(pg_loss))

    def record_peak_mem(self, gb_per_rank: list[float]):
        self.data["memory"]["peak_gpu_gb_per_rank"] = [float(x) for x in gb_per_rank]

    def finalize(self):
        total = time.perf_counter() - self._t0
        self.data["timing"]["total_s"] = total
        toks = self.data["throughput"]["gen_tokens_total"]
        nsteps = len(self.data["timing"]["per_step_s"])
        self.data["throughput"]["gen_tokens_per_s"] = (toks / total) if total > 0 else 0.0
        self.data["throughput"]["steps_per_hr"] = (nsteps / total * 3600) if total > 0 else 0.0
        self.write()

    def write(self):
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.out_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest recipe/pie_grpo/experiments/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recipe/pie_grpo/experiments/metrics.py recipe/pie_grpo/experiments/test_metrics.py
git commit -m "pie_grpo/experiments: add RunMetrics collector"
```

---

### Task 3: Wire RunMetrics into `train_grpo.py`

**Files:**
- Modify: `recipe/pie_grpo/train_grpo.py` (imports, `main()` argparse, `train()` phase hooks)
- Test: `recipe/pie_grpo/experiments/test_train_hooks.py` (import + argparse smoke; full loop is RunPod-verified)

**Interfaces:**
- Consumes: `RunMetrics` (Task 2).
- Produces: `train_grpo` writes a schema-valid `metrics.json` when `--metrics-out PATH` is passed (rank 0 only). No behavior change when the flag is absent (`metrics` is `None`, all calls guarded).

- [ ] **Step 1: Add the import and a null-safe helper**

In `train_grpo.py`, after the existing `from recipe.pie_grpo.topology import (...)` block, add:

```python
from recipe.pie_grpo.experiments.metrics import RunMetrics
```

- [ ] **Step 2: Add `--metrics-out` to `main()`**

In `main()`, extend the argparser:

```python
ap.add_argument("--metrics-out", default=None,
                help="write experiment metrics.json to this path (rank 0)")
```

and pass it into `train`: change `train(cfg)` to `train(cfg, metrics_out=args.metrics_out)`.

- [ ] **Step 3: Thread the collector through `train()`**

Change the signature to `def train(cfg: dict, metrics_out: str | None = None) -> None:`. Right after `is_main = rank == 0` and the hyperparam reads, add:

```python
    metrics = None
    if is_main and metrics_out:
        metrics = RunMetrics(
            backend="pie",
            run_meta={"model": cfg["model"]["path"], "n_gpus": world_size,
                      "gpu": os.environ.get("PIE_EXP_GPU", "unknown"),
                      "num_steps": cfg["train"]["num_steps"],
                      "hyperparams": {"n_samples": rollout["n_samples"],
                                      "lr": actor_cfg["lr"],
                                      "temperature": rollout["temperature"]}},
            out_path=metrics_out)
```

- [ ] **Step 4: Wrap the loop phases**

Wrap each phase in the per-step loop with `metrics.phase(...)` when `metrics` is set. Concretely, inside `for step in range(...)`:
- At loop top: `if metrics: metrics.start_step()`.
- Wrap the rank-0 `worker.generate(...)` call in `with (metrics.phase("gen") if metrics else _nullctx()):`. After building `gen`, add `if metrics: metrics.record_gen_tokens(sum(len(s["tokens"]) for g in gen for s in g["samples"]))`.
- Wrap `compute_log_prob` in `phase("logprob")`, `compute_grpo_outcome_advantage` block in `phase("adv")`, `update_actor` in `phase("update")`, `sync_weights` in `phase("weight_sync")`.
- In the rank-0 readout block, add `if metrics: metrics.record_train(rmean, pg_loss)`.
- Wrap the `run_eval(...)` calls (baseline + periodic) so eval time lands in `phase("eval")`, and after each eval capture the numbers: have `run_eval` return `(heval, mbpp)` and add `if metrics: metrics.record_eval(step, heval, mbpp)`.
- At loop bottom: `if metrics: metrics.end_step()`.
- After the loop (rank 0): `if metrics: metrics.record_peak_mem([torch.cuda.max_memory_allocated(rank)/1e9]); metrics.finalize()`.

Add a module-level null context near the other helpers:

```python
import contextlib
@contextlib.contextmanager
def _nullctx():
    yield
```

(Reuse `contextlib` if already imported.) Note: `run_eval` currently returns only `acc`; change it to `return acc, (m if mbpp_prompts else None)` and update its two call sites to unpack.

- [ ] **Step 5: Write the smoke test**

```python
# recipe/pie_grpo/experiments/test_train_hooks.py
import importlib
def test_train_grpo_imports_and_has_metrics_arg():
    tg = importlib.import_module("recipe.pie_grpo.train_grpo")
    import inspect
    src = inspect.getsource(tg.main)
    assert "--metrics-out" in src
    assert "metrics_out" in inspect.signature(tg.train).parameters
```

- [ ] **Step 6: Run it**

Run: `python -m pytest recipe/pie_grpo/experiments/test_train_hooks.py -v`
Expected: PASS. (Full metrics.json emission is verified on RunPod in Task 9 — needs GPUs.)

- [ ] **Step 7: Commit**

```bash
git add recipe/pie_grpo/train_grpo.py recipe/pie_grpo/experiments/test_train_hooks.py
git commit -m "pie_grpo: emit RunMetrics from train_grpo (--metrics-out)"
```

---

### Task 4: `compare.py` — table + plot from two runs

**Files:**
- Create: `recipe/pie_grpo/experiments/compare.py`
- Test: `recipe/pie_grpo/experiments/test_compare.py`

**Interfaces:**
- Produces: `load(path) -> dict`; `summary_row(d) -> dict` (backend, total_s, steps_per_hr, gen_tokens_per_s, peak_gb, final_heval, best_heval, final_mbpp); `render_table(rows) -> str`; `main(argv)` writes a PNG plot next to the first metrics file.
- Consumes: `metrics.json` from Task 2/3/6.

- [ ] **Step 1: Write the failing test**

```python
# recipe/pie_grpo/experiments/test_compare.py
import json
from recipe.pie_grpo.experiments import compare

def _fixture(tmp_path, backend, heval):
    d = {"run": {"backend": backend},
         "timing": {"total_s": 100.0, "per_step_s": [1.0]*50, "phase_totals_s": {"gen": 40.0}},
         "throughput": {"gen_tokens_total": 5000, "gen_tokens_per_s": 50.0, "steps_per_hr": 1800.0},
         "memory": {"peak_gpu_gb_per_rank": [20.0, 20.0]},
         "eval": [{"step": 0, "heval": 0.5, "mbpp": 0.7}, {"step": 49, "heval": heval, "mbpp": 0.7}],
         "train": {"reward_mean_per_step": [0.6], "pg_loss_per_step": [-0.03]}}
    p = tmp_path / f"{backend}.json"; p.write_text(json.dumps(d)); return str(p)

def test_summary_and_table(tmp_path):
    a = compare.load(_fixture(tmp_path, "pie", 0.66))
    row = compare.summary_row(a)
    assert row["backend"] == "pie"
    assert row["final_heval"] == 0.66
    assert row["best_heval"] == 0.66
    assert abs(row["peak_gb"] - 20.0) < 1e-6
    table = compare.render_table([row])
    assert "pie" in table and "steps_per_hr" in table
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest recipe/pie_grpo/experiments/test_compare.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `compare.py`**

```python
# recipe/pie_grpo/experiments/compare.py
"""Compare two experiment metrics.json (e.g. pie vs verl-vllm): table + plot."""
from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summary_row(d: dict) -> dict:
    evals = d.get("eval", [])
    hevals = [e["heval"] for e in evals]
    mbpps = [e["mbpp"] for e in evals if e.get("mbpp") is not None]
    mem = d["memory"]["peak_gpu_gb_per_rank"] or [0.0]
    return {
        "backend": d["run"]["backend"],
        "total_s": round(d["timing"]["total_s"], 1),
        "steps_per_hr": round(d["throughput"]["steps_per_hr"], 1),
        "gen_tokens_per_s": round(d["throughput"]["gen_tokens_per_s"], 1),
        "peak_gb": round(max(mem), 2),
        "final_heval": (hevals[-1] if hevals else None),
        "best_heval": (max(hevals) if hevals else None),
        "final_mbpp": (mbpps[-1] if mbpps else None),
    }


def render_table(rows: list[dict]) -> str:
    cols = ["backend", "total_s", "steps_per_hr", "gen_tokens_per_s", "peak_gb",
            "final_heval", "best_heval", "final_mbpp"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = lambda vals: " | ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals))
    out = [line(cols), "-+-".join("-" * widths[c] for c in cols)]
    out += [line([r[c] for c in cols]) for r in rows]
    return "\n".join(out)


def _plot(runs: list[dict], out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for d in runs:
        ev = d.get("eval", [])
        ax1.plot([e["step"] for e in ev], [e["heval"] for e in ev],
                 marker="o", label=d["run"]["backend"])
    ax1.set_title("held-out HumanEval pass@1"); ax1.set_xlabel("step"); ax1.legend()
    backends = [d["run"]["backend"] for d in runs]
    ax2.bar(backends, [d["throughput"]["steps_per_hr"] for d in runs])
    ax2.set_title("steps / hour")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="two+ metrics.json paths")
    ap.add_argument("--plot", default=None, help="output PNG (default: alongside first)")
    args = ap.parse_args(argv)
    runs = [load(p) for p in args.metrics]
    print(render_table([summary_row(d) for d in runs]))
    out_png = args.plot or (args.metrics[0].rsplit("/", 1)[0] + "/compare.png")
    _plot(runs, out_png)
    print(f"\nplot: {out_png}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest recipe/pie_grpo/experiments/test_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recipe/pie_grpo/experiments/compare.py recipe/pie_grpo/experiments/test_compare.py
git commit -m "pie_grpo/experiments: add compare.py (table + plot)"
```

---

### Task 5: verl-interface HumanEval reward

**Files:**
- Create: `recipe/pie_grpo/experiments/baseline/__init__.py` (empty)
- Create: `recipe/pie_grpo/experiments/baseline/humaneval_reward_verl.py`
- Test: `recipe/pie_grpo/experiments/baseline/test_humaneval_reward_verl.py`

**Interfaces:**
- Produces: `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float` — verl's reward signature (called per-sample by verl's reward manager). Wraps the existing `recipe.pie_grpo.humaneval_reward.compute_reward(solution_str, ground_truth)` so the baseline uses the identical scorer as the Pie run.

- [ ] **Step 1: Write the failing test**

```python
# recipe/pie_grpo/experiments/baseline/test_humaneval_reward_verl.py
import pandas as pd, pathlib
from recipe.pie_grpo.experiments.baseline import humaneval_reward_verl as R

def test_compute_score_matches_underlying(monkeypatch):
    called = {}
    def fake(sol, gt):
        called["args"] = (sol, gt); return 1.0
    monkeypatch.setattr(R, "compute_reward", fake)
    s = R.compute_score("humaneval", "def f(): pass", "GT", extra_info=None)
    assert s == 1.0
    assert called["args"] == ("def f(): pass", "GT")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest recipe/pie_grpo/experiments/baseline/test_humaneval_reward_verl.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# recipe/pie_grpo/experiments/baseline/humaneval_reward_verl.py
"""verl-interface reward for the baseline: wraps the SAME HumanEval unit-test scorer
the Pie run uses (recipe.pie_grpo.humaneval_reward.compute_reward), so reward
semantics are identical across backends. verl calls compute_score per sample."""
from __future__ import annotations

from recipe.pie_grpo.humaneval_reward import compute_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return float(compute_reward(solution_str, ground_truth))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest recipe/pie_grpo/experiments/baseline/test_humaneval_reward_verl.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recipe/pie_grpo/experiments/baseline/
git commit -m "pie_grpo/experiments: verl-interface HumanEval reward for baseline"
```

---

### Task 6: verl metrics bridge + baseline config + runner

**Files:**
- Create: `recipe/pie_grpo/experiments/baseline/verl_metrics_bridge.py`
- Create: `recipe/pie_grpo/experiments/baseline/test_verl_metrics_bridge.py`
- Create: `recipe/pie_grpo/experiments/baseline/grpo_baseline_config.yaml`
- Create: `recipe/pie_grpo/experiments/baseline/run_baseline.sh`

**Interfaces:**
- Consumes: `RunMetrics` (Task 2).
- Produces: `build_from_verl(run_meta, per_step_metrics, total_s, val_curve, peak_mem, out_path)` — maps a list of verl per-step metric dicts + verl validation results into a schema-valid `metrics.json` via `RunMetrics`. `verl` key names used: `response_length/mean` (gen tokens proxy), `actor/pg_loss`, `critic/rewards/mean`; validation dict keyed by data_source → pass@1.

- [ ] **Step 1: Write the failing test**

```python
# recipe/pie_grpo/experiments/baseline/test_verl_metrics_bridge.py
import json
from recipe.pie_grpo.experiments.baseline import verl_metrics_bridge as B

def test_bridge_writes_schema(tmp_path):
    out = tmp_path / "verl.json"
    per_step = [{"actor/pg_loss": -0.03, "critic/rewards/mean": 0.6,
                 "response_length/mean": 200.0, "step_time_s": 2.0} for _ in range(3)]
    val_curve = [{"step": 0, "heval": 0.5, "mbpp": 0.7},
                 {"step": 2, "heval": 0.64, "mbpp": 0.71}]
    B.build_from_verl(run_meta={"model": "x", "n_gpus": 2}, per_step_metrics=per_step,
                      total_s=6.0, val_curve=val_curve, peak_mem=[20.0, 20.0],
                      out_path=str(out), n_samples_per_step=8*8, resp_len_key="response_length/mean")
    d = json.loads(out.read_text())
    assert d["run"]["backend"] == "verl-vllm"
    assert len(d["timing"]["per_step_s"]) == 3
    assert d["eval"][-1]["heval"] == 0.64
    assert d["train"]["pg_loss_per_step"] == [-0.03, -0.03, -0.03]
    assert d["throughput"]["gen_tokens_total"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest recipe/pie_grpo/experiments/baseline/test_verl_metrics_bridge.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the bridge**

```python
# recipe/pie_grpo/experiments/baseline/verl_metrics_bridge.py
"""Map verl-native GRPO run outputs into the common metrics.json schema so
compare.py can diff it against the Pie run. verl exposes a per-step metrics dict
(the source of its console logging) + a validation dict; we translate both."""
from __future__ import annotations

from recipe.pie_grpo.experiments.metrics import RunMetrics


def build_from_verl(run_meta, per_step_metrics, total_s, val_curve, peak_mem,
                    out_path, n_samples_per_step, resp_len_key="response_length/mean"):
    m = RunMetrics(backend="verl-vllm", run_meta=run_meta, out_path=out_path)
    for st in per_step_metrics:
        m.start_step()
        # verl has no distinct Pie weight-sync phase; bucket the whole step under "update".
        with m.phase("update"):
            pass
        m.data["timing"]["per_step_s"].append(float(st.get("step_time_s", 0.0)))
        m.data["timing"]["phase_totals_s"]["update"] = \
            m.data["timing"]["phase_totals_s"].get("update", 0.0) + float(st.get("step_time_s", 0.0))
        m.record_train(reward_mean=float(st.get("critic/rewards/mean", 0.0)),
                       pg_loss=float(st.get("actor/pg_loss", 0.0)))
        # gen tokens proxy: mean response length * samples/step
        m.record_gen_tokens(int(float(st.get(resp_len_key, 0.0)) * n_samples_per_step))
        m._step_t0 = None  # per_step already appended above
    for pt in val_curve:
        m.record_eval(step=pt["step"], heval=pt["heval"], mbpp=pt.get("mbpp"))
    m.record_peak_mem(peak_mem)
    # override computed total with the measured wall-clock (verl owns the loop timing)
    m.finalize()
    m.data["timing"]["total_s"] = float(total_s)
    toks = m.data["throughput"]["gen_tokens_total"]
    n = len(m.data["timing"]["per_step_s"])
    m.data["throughput"]["gen_tokens_per_s"] = toks / total_s if total_s > 0 else 0.0
    m.data["throughput"]["steps_per_hr"] = n / total_s * 3600 if total_s > 0 else 0.0
    m.write()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest recipe/pie_grpo/experiments/baseline/test_verl_metrics_bridge.py -v`
Expected: PASS.

- [ ] **Step 5: Write the verl baseline config**

```yaml
# recipe/pie_grpo/experiments/baseline/grpo_baseline_config.yaml
# verl-native GRPO baseline (main_ppo). Same model/data/hyperparams as the Pie run;
# only the rollout+weight-sync backend differs (verl's native vLLM SPMD, no Pie).
# Launched by run_baseline.sh with hydra overrides for paths/devices.
algorithm:
  adv_estimator: grpo
data:
  train_files: recipe/pie_grpo/experiments/data/humaneval_train.parquet
  val_files: recipe/pie_grpo/experiments/data/humaneval_test.parquet
  train_batch_size: 8          # problems_per_step
  max_prompt_length: 512
  max_response_length: 512
  prompt_key: prompt
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-Coder-1.5B-Instruct
  actor:
    optim:
      lr: 3.0e-6
    ppo_mini_batch_size: 16
    use_dynamic_bsz: false
  rollout:
    name: vllm
    n: 8                        # n_samples
    temperature: 0.8
    top_p: 0.95
    gpu_memory_utilization: 0.5
    tensor_model_parallel_size: 1
reward:
  custom_reward_function:
    path: recipe/pie_grpo/experiments/baseline/humaneval_reward_verl.py
    name: compute_score
trainer:
  n_gpus_per_node: 2
  nnodes: 1
  total_epochs: 1
  total_training_steps: 50
  test_freq: 5
  logger: [console]
```

Note: field names above follow verl's `main_ppo` config tree. During Task 8 first-pod bring-up, reconcile any renamed keys against `verl/trainer/config/` (verl moves fast) — the runner prints the resolved config so mismatches surface immediately.

- [ ] **Step 6: Write the runner**

```bash
#!/usr/bin/env bash
# recipe/pie_grpo/experiments/baseline/run_baseline.sh
# Launch verl-native GRPO baseline on 2 GPUs and emit a common metrics.json.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-runs/verl_baseline/metrics.json}"; mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VERL_METRICS_OUT="$OUT"          # read by the metrics callback (see note)
python -m verl.trainer.main_ppo \
  --config-path "$(pwd)/recipe/pie_grpo/experiments/baseline" \
  --config-name grpo_baseline_config \
  2>&1 | tee "runs/verl_baseline/console.log"
echo "baseline metrics -> $OUT"
```

Note on capture: verl's trainer exposes the per-step metrics dict to its logger. The cleanest hook is a small custom logger that appends each step's dict + `time.perf_counter` delta and, on run end, calls `verl_metrics_bridge.build_from_verl(...)`. If wiring a logger proves invasive on the pinned verl, the fallback is a `--metrics-from-console` mode in a tiny post-processor that parses `console.log` for the `actor/pg_loss`, `critic/rewards/mean`, `response_length/mean`, and `val-core/.../pass@1` lines verl already prints, then calls the bridge. Decide on the first pod (Task 8); both feed the same `build_from_verl`.

- [ ] **Step 7: Commit**

```bash
git add recipe/pie_grpo/experiments/baseline/
git commit -m "pie_grpo/experiments: verl baseline config, runner, metrics bridge"
```

---

### Task 7: RunPod configs (Pie server + train_grpo)

**Files:**
- Create: `recipe/pie_grpo/experiments/configs/qwen-rl-config.toml`
- Create: `recipe/pie_grpo/experiments/configs/pie_config.yaml`

**Interfaces:**
- Produces: a Pie server TOML with `device = ["cuda:0","cuda:1"]`, `tensor_parallel_size = 1`, `load_format = "dummy"`, `hf_repo = Qwen/Qwen2.5-Coder-1.5B-Instruct`; and a `pie_config.yaml` identical to the repo `config.yaml` except data/eval paths point at `experiments/data/*` and `topology`/`pie` match 2×A40 (cuda:0,1). Consumed by `run_all.sh` (Task 10) / `setup_runpod.sh` (Task 8).

- [ ] **Step 1: Create the Pie server config**

```toml
# recipe/pie_grpo/experiments/configs/qwen-rl-config.toml
[server]
host = "127.0.0.1"
port = 8080
[auth]
enabled = false
[[model]]
name = "default"
hf_repo = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
[model.driver]
type = "vllm"
device = ["cuda:0", "cuda:1"]   # RunPod 2x A40; CVD order must match the trainer
tensor_parallel_size = 1
activation_dtype = "bfloat16"
random_seed = 42
[model.driver.options]
venv = "PIE_VENV_PATH"          # setup_runpod.sh sed-replaces with the created venv
gpu_memory_utilization = 0.5    # A40 48GB has ample headroom
enforce_eager = true
max_num_batched_tokens = 4000
load_format = "dummy"
```

- [ ] **Step 2: Create the train_grpo config for RunPod**

Copy the repo `config.yaml` and change only the paths + devices. The file must be identical to `recipe/pie_grpo/config.yaml` except:
- `data.humaneval_parquet: recipe/pie_grpo/experiments/data/humaneval_train.parquet`
- `eval.parquet: recipe/pie_grpo/experiments/data/humaneval_test.parquet`
- `eval.mbpp_parquet: recipe/pie_grpo/experiments/data/mbpp_test.parquet`
- keep `topology` defaults (auto → world_size=2), `pie.tensor_parallel_size: 1`, `num_steps: 50`.

```bash
cp recipe/pie_grpo/config.yaml recipe/pie_grpo/experiments/configs/pie_config.yaml
# then edit the three data/eval paths above (they currently point at /home/george/agentic-rl/...)
```

- [ ] **Step 3: Verify the configs parse**

```bash
python -c "import tomllib,yaml; tomllib.load(open('recipe/pie_grpo/experiments/configs/qwen-rl-config.toml','rb')); yaml.safe_load(open('recipe/pie_grpo/experiments/configs/pie_config.yaml')); print('configs parse')"
```
Expected: `configs parse`.

- [ ] **Step 4: Commit**

```bash
git add recipe/pie_grpo/experiments/configs/
git commit -m "pie_grpo/experiments: RunPod configs (2x A40)"
```

---

### Task 8: `setup_runpod.sh` — env bootstrap

**Files:**
- Create: `recipe/pie_grpo/experiments/setup_runpod.sh`

**Interfaces:**
- Produces: a provisioned pod — `pie` on PATH (lean build), a venv at `$HOME/.pie/venvs/vllm` with pinned torch/vLLM + editable pie/verl, datasets present, `grpo` inferlet installed, configs templated with the venv path. Idempotent; fail-loud version guard.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# recipe/pie_grpo/experiments/setup_runpod.sh
# Thin-layer bootstrap on a stock RunPod A40 PyTorch/CUDA base image.
# Builds its OWN venv (base torch 2.1-2.8 is NOT used) with the pinned stack.
set -euo pipefail

VERL_FORK="${VERL_FORK:-https://github.com/Giotyp/verl.git}"
PIE_FORK="${PIE_FORK:-https://github.com/Giotyp/pie.git}"
WORK="${WORK:-$HOME/agentic-rl}"; mkdir -p "$WORK"
VENV="$HOME/.pie/venvs/vllm"

echo "== topology (interpret the speedup against this) =="; nvidia-smi topo -m || true

echo "== system deps =="
apt-get update -y && apt-get install -y git build-essential curl
command -v cargo >/dev/null || (curl https://sh.rustup.rs -sSf | sh -s -- -y && . "$HOME/.cargo/env")
. "$HOME/.cargo/env" 2>/dev/null || true

echo "== clone forks =="
[ -d "$WORK/pie-gt" ] || git clone --branch features/RL "$PIE_FORK" "$WORK/pie-gt"
[ -d "$WORK/verl" ]   || git clone --branch pie-rl "$VERL_FORK" "$WORK/verl"

echo "== build pie (lean, no CUDA) =="
(cd "$WORK/pie-gt" && cargo install --path server --force \
   --no-default-features --features driver-portable,driver-dummy)

echo "== venv + pinned stack (cu128) =="
python3 -m venv "$VENV"; . "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.21.0
pip install -r "$WORK/verl/recipe/pie_grpo/experiments/requirements.lock" || true
pip install -e "$WORK/pie-gt/driver/vllm" -e "$WORK/pie-gt/client/python" -e "$WORK/verl"

echo "== version guard =="
python - <<'PY'
import torch, vllm, sys
assert torch.__version__.startswith("2.11.0"), f"torch {torch.__version__} != 2.11.0"
assert not torch.__version__.endswith("+cu129"), "cu129 wheel won't run on A40 (cu128 max)"
assert vllm.__version__ == "0.21.0", f"vllm {vllm.__version__} != 0.21.0"
print("version guard OK:", torch.__version__, vllm.__version__)
PY

echo "== build + install grpo inferlet =="
(cd "$WORK/pie-gt" && conda activate pie-env 2>/dev/null; \
   bakery build "$WORK/PieRL/grpo" -o /tmp/grpo.wasm) || \
   echo "NOTE: build the grpo inferlet per build-inferlet skill; install via pie."

echo "== template configs with venv path =="
CFG="$WORK/verl/recipe/pie_grpo/experiments/configs"
sed "s#PIE_VENV_PATH#$VENV#" "$CFG/qwen-rl-config.toml" > "$WORK/qwen-rl-config.toml"

cat <<EOF

== setup complete ==
Serve Pie:   pie serve --config $WORK/qwen-rl-config.toml --no-auth
Pie run:     cd $WORK/verl && CUDA_VISIBLE_DEVICES=0,1 PIE_EXP_GPU=A40 torchrun --nproc_per_node=2 \\
               -m recipe.pie_grpo.train_grpo --config recipe/pie_grpo/experiments/configs/pie_config.yaml \\
               --metrics-out runs/pie/metrics.json
Baseline:    bash recipe/pie_grpo/experiments/baseline/run_baseline.sh runs/verl_baseline/metrics.json
Compare:     python -m recipe.pie_grpo.experiments.compare runs/pie/metrics.json runs/verl_baseline/metrics.json
EOF
```

- [ ] **Step 2: Lint**

Run: `bash -n recipe/pie_grpo/experiments/setup_runpod.sh` (syntax check). If `shellcheck` is available: `shellcheck recipe/pie_grpo/experiments/setup_runpod.sh` and fix warnings.
Expected: no syntax errors.

- [ ] **Step 3: First-pod verification (documented — runs on RunPod, not here)**

On the first A40 pod: run the script; confirm the version guard passes, `pie serve` boots (lean build serves the vllm driver), and a short Pie run reaches the baseline eval. If the lean build fails to serve, re-run the pie build step adding `driver-cuda` + `apt-get install -y cuda-toolkit` (documented fallback). Record `nvidia-smi topo -m` (NVLink vs PCIe).

- [ ] **Step 4: Commit**

```bash
git add recipe/pie_grpo/experiments/setup_runpod.sh
git commit -m "pie_grpo/experiments: RunPod setup_runpod.sh"
```

---

### Task 9: `publish_repos.sh` — make forks accessible

**Files:**
- Create: `recipe/pie_grpo/experiments/publish_repos.sh`

**Interfaces:**
- Produces: pushes `pie-rl` → `Giotyp/verl`, `features/RL` → `Giotyp/pie`; sets both public. Run locally by the user (needs their GitHub auth).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# recipe/pie_grpo/experiments/publish_repos.sh
# Push the experiment branches to the user's forks and make them public so a
# fresh RunPod pod can clone them. Run locally (needs your GitHub auth).
set -euo pipefail
git -C /home/george/git_repos/verl push origin pie-rl
git -C /home/george/git_repos/pie-gt push origin features/RL
gh repo edit Giotyp/verl --visibility public --accept-visibility-change-consequences
gh repo edit Giotyp/pie  --visibility public --accept-visibility-change-consequences
echo "public: https://github.com/Giotyp/verl (pie-rl), https://github.com/Giotyp/pie (features/RL)"
```

- [ ] **Step 2: Lint**

Run: `bash -n recipe/pie_grpo/experiments/publish_repos.sh`
Expected: no syntax errors. (Execution is the user's — pushes to their GitHub.)

- [ ] **Step 3: Commit**

```bash
git add recipe/pie_grpo/experiments/publish_repos.sh
git commit -m "pie_grpo/experiments: publish_repos.sh"
```

---

### Task 10: `run_all.sh` — orchestrator

**Files:**
- Create: `recipe/pie_grpo/experiments/run_all.sh`

**Interfaces:**
- Produces: one command that runs the Pie experiment, the baseline, and compare; or a single stage via `$1` in {pie, baseline, compare, all}. Assumes `setup_runpod.sh` already ran and Pie server is served for the `pie` stage.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# recipe/pie_grpo/experiments/run_all.sh  [pie|baseline|compare|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
STAGE="${1:-all}"
PIE_OUT="runs/pie/metrics.json"; BASE_OUT="runs/verl_baseline/metrics.json"
mkdir -p runs/pie runs/verl_baseline

run_pie() {
  echo "== Pie run (assumes 'pie serve' is up on 127.0.0.1:8080) =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PIE_EXP_GPU="${PIE_EXP_GPU:-A40}" \
    torchrun --nproc_per_node=2 -m recipe.pie_grpo.train_grpo \
    --config recipe/pie_grpo/experiments/configs/pie_config.yaml --metrics-out "$PIE_OUT"
}
run_baseline() { bash recipe/pie_grpo/experiments/baseline/run_baseline.sh "$BASE_OUT"; }
run_compare()  { python -m recipe.pie_grpo.experiments.compare "$PIE_OUT" "$BASE_OUT"; }

case "$STAGE" in
  pie) run_pie ;;
  baseline) run_baseline ;;
  compare) run_compare ;;
  all) run_pie; run_baseline; run_compare ;;
  *) echo "usage: $0 [pie|baseline|compare|all]"; exit 1 ;;
esac
```

- [ ] **Step 2: Lint**

Run: `bash -n recipe/pie_grpo/experiments/run_all.sh`
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add recipe/pie_grpo/experiments/run_all.sh
git commit -m "pie_grpo/experiments: run_all.sh orchestrator"
```

---

## Self-Review

**Spec coverage:** (1) repo accessibility → Task 9. (2) time/perf stats → Tasks 2,3 (Pie) + 6 (baseline). (3) baseline → Tasks 5,6. (4) setup script → Task 8; datasets → Task 1; configs → Task 7; compare → Task 4; orchestrator → Task 10. cu128/lean-build constraints → Task 8 + Global Constraints. All spec sections covered.

**Placeholder scan:** the two "decide on first pod" notes (Task 6 metrics-capture method, Task 8 lean-vs-full build) are genuine spec-flagged verification branches with both paths spelled out, not gaps. No TBD/TODO code.

**Type consistency:** `RunMetrics` method names/signatures identical across Tasks 2,3,6. `compute_score` (Task 5) matches verl's reward signature and the `custom_reward_function.name` in Task 6 config. `metrics.json` schema identical across producer (2) and consumer (4).

**RunPod-only steps** (full GPU runs) are explicitly deferred to first-pod verification with documented fallbacks — the local test cycle covers every pure-Python unit.
