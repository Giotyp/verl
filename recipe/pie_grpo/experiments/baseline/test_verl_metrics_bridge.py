import json

from recipe.pie_grpo.experiments.baseline import verl_metrics_bridge as B


def test_bridge_writes_schema(tmp_path):
    out = tmp_path / "verl.json"
    per_step = [{"actor/pg_loss": -0.03, "critic/rewards/mean": 0.6,
                 "response_length/mean": 200.0, "step_time_s": 2.0} for _ in range(3)]
    val_curve = [{"step": -1, "heval": 0.5, "mbpp": 0.7},
                 {"step": 2, "heval": 0.64, "mbpp": 0.71}]
    B.build_from_verl(run_meta={"model": "x", "n_gpus": 2}, per_step_metrics=per_step,
                      total_s=6.0, val_curve=val_curve, peak_mem=[20.0, 20.0],
                      out_path=str(out), n_samples_per_step=8 * 8,
                      resp_len_key="response_length/mean")
    d = json.loads(out.read_text())
    assert d["run"]["backend"] == "verl-vllm"
    assert len(d["timing"]["per_step_s"]) == 3
    assert d["timing"]["total_s"] == 6.0
    assert d["eval"][-1]["heval"] == 0.64
    assert d["train"]["pg_loss_per_step"] == [-0.03, -0.03, -0.03]
    assert d["throughput"]["gen_tokens_total"] == 200 * 64 * 3
    assert d["throughput"]["steps_per_hr"] == 3 / 6.0 * 3600
