import pathlib

import pandas as pd

DATA = pathlib.Path(__file__).parent / "data"


def test_parquets_present_and_schema():
    # mbpp_test holds all 257 sanitized problems; the eval caps to 100 at run time.
    for name, n in [("humaneval_train", 131), ("humaneval_test", 33), ("mbpp_test", 257)]:
        df = pd.read_parquet(DATA / f"{name}.parquet")
        assert len(df) == n, (name, len(df))
        assert {"prompt", "reward_model"} <= set(df.columns)
        rm = df.iloc[0]["reward_model"]
        assert "ground_truth" in rm and isinstance(rm["ground_truth"], str)


def test_requirements_lock_excludes_cuda_pins():
    # torch/vllm/nvidia are installed separately (cu128) by setup_runpod.sh. Check the
    # actual pin LINES (ignore the header comment, which mentions those names in prose).
    lock = (pathlib.Path(__file__).parent / "requirements.lock").read_text()
    pins = [ln.strip() for ln in lock.splitlines() if ln.strip() and not ln.startswith("#")]
    for p in pins:
        assert not p.startswith(("torch==", "torchvision==", "torchaudio==", "vllm==", "nvidia-")), p
    assert "transformers==5.12.1" in pins and "tensordict==0.10.0" in pins
