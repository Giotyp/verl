import json

from recipe.pie_grpo.experiments import compare


def _fixture(tmp_path, backend, heval):
    d = {"run": {"backend": backend},
         "timing": {"total_s": 100.0, "per_step_s": [1.0] * 50, "phase_totals_s": {"gen": 40.0}},
         "throughput": {"gen_tokens_total": 5000, "gen_tokens_per_s": 50.0, "steps_per_hr": 1800.0},
         "memory": {"peak_gpu_gb_per_rank": [20.0, 20.0]},
         "eval": [{"step": -1, "heval": 0.5, "mbpp": 0.7}, {"step": 49, "heval": heval, "mbpp": 0.7}],
         "train": {"reward_mean_per_step": [0.6], "pg_loss_per_step": [-0.03]}}
    p = tmp_path / f"{backend}.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_summary_and_table(tmp_path):
    a = compare.load(_fixture(tmp_path, "pie", 0.66))
    row = compare.summary_row(a)
    assert row["backend"] == "pie"
    assert row["final_heval"] == 0.66
    assert row["best_heval"] == 0.66
    assert abs(row["peak_gb"] - 20.0) < 1e-6
    table = compare.render_table([row])
    assert "pie" in table and "steps_per_hr" in table


def test_plot_written(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    p1 = _fixture(tmp_path, "pie", 0.66)
    p2 = _fixture(tmp_path, "verl-vllm", 0.60)
    out = tmp_path / "cmp.png"
    compare.main([p1, p2, "--plot", str(out)])
    assert out.exists() and out.stat().st_size > 0
