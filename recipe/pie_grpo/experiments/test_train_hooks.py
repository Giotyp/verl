import ast
import importlib
import inspect
import pathlib


def test_train_grpo_parses():
    src = pathlib.Path(__file__).parents[1].joinpath("train_grpo.py").read_text()
    ast.parse(src)  # raises SyntaxError if the metrics hooks broke indentation


def test_train_grpo_has_metrics_wiring():
    tg = importlib.import_module("recipe.pie_grpo.train_grpo")
    assert "--metrics-out" in inspect.getsource(tg.main)
    assert "metrics_out" in inspect.signature(tg.train).parameters
    # the collector must not shadow the update_actor `metrics` local
    tsrc = inspect.getsource(tg.train)
    assert "run_metrics" in tsrc
    assert "metrics = actor.update_actor(" in tsrc  # untouched update return var
