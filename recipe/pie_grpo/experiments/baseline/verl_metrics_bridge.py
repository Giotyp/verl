"""Map verl-native GRPO run outputs into the common metrics.json schema so
compare.py can diff it against the Pie run. verl exposes a per-step metrics dict
(the source of its console logging) + a validation dict; we translate both.

verl has no distinct Pie weight-sync phase, so each step's wall-clock is bucketed
under `update` — the comparable headline metrics are total_s / steps_per_hr /
gen_tokens_per_s / peak_gb / the eval curve.
"""
from __future__ import annotations

from recipe.pie_grpo.experiments.metrics import RunMetrics


def build_from_verl(run_meta, per_step_metrics, total_s, val_curve, peak_mem,
                    out_path, n_samples_per_step, resp_len_key="response_length/mean"):
    m = RunMetrics(backend="verl-vllm", run_meta=run_meta, out_path=out_path)
    for st in per_step_metrics:
        dt = float(st.get("step_time_s", 0.0))
        m.data["timing"]["per_step_s"].append(dt)
        m.data["timing"]["phase_totals_s"]["update"] = \
            m.data["timing"]["phase_totals_s"].get("update", 0.0) + dt
        m.record_train(reward_mean=float(st.get("critic/rewards/mean", 0.0)),
                       pg_loss=float(st.get("actor/pg_loss", 0.0)))
        # gen-tokens proxy: mean response length * samples per step
        m.record_gen_tokens(int(float(st.get(resp_len_key, 0.0)) * n_samples_per_step))
    for pt in val_curve:
        m.record_eval(step=pt["step"], heval=pt["heval"], mbpp=pt.get("mbpp"))
    m.record_peak_mem(peak_mem)
    # verl owns the loop timing; override the collector's perf_counter total.
    m.data["timing"]["total_s"] = float(total_s)
    toks = m.data["throughput"]["gen_tokens_total"]
    n = len(m.data["timing"]["per_step_s"])
    m.data["throughput"]["gen_tokens_per_s"] = toks / total_s if total_s > 0 else 0.0
    m.data["throughput"]["steps_per_hr"] = n / total_s * 3600 if total_s > 0 else 0.0
    m.write()
