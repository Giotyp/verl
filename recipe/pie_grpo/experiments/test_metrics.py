import json
import time

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


def test_mbpp_none_allowed(tmp_path):
    out = tmp_path / "m.json"
    m = RunMetrics(backend="verl-vllm", run_meta={}, out_path=str(out))
    m.record_eval(step=3, heval=0.6, mbpp=None)
    d = json.loads(out.read_text())
    assert d["eval"][0]["mbpp"] is None
