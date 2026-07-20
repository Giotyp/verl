from recipe.pie_grpo.experiments.baseline import humaneval_reward_verl as R


def test_compute_score_matches_underlying(monkeypatch):
    called = {}

    def fake(sol, gt):
        called["args"] = (sol, gt)
        return 1.0

    monkeypatch.setattr(R, "compute_reward", fake)
    s = R.compute_score("humaneval", "def f(): pass", "GT", extra_info=None)
    assert s == 1.0
    assert called["args"] == ("def f(): pass", "GT")
