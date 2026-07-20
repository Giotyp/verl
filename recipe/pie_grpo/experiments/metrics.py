"""Dependency-free run metrics for GRPO experiments (Pie + verl baseline).

Writes a single JSON schema so compare.py can diff two runs. Timing is wall-clock
via time.perf_counter. All numbers are plain floats/ints; no torch/verl imports so
it is unit-testable anywhere.
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
            "timing": {"total_s": 0.0, "per_step_s": [], "phase_totals_s": {}},
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
