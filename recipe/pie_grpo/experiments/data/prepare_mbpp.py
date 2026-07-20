"""Prepare the MBPP transfer-eval parquet (sibling of RL-Heval's prepare_humaneval).

Writes ``data/mbpp_test.parquet`` (under PieRL) in the SAME schema the trainer's
``load_humaneval_fixed`` already reads (prompt as a chat turn, ground_truth a JSON
string), so no new loader is needed on the training side.

MBPP "sanitized" = 427 human-verified problems; the ``test`` split (~257) is the
standard held-out set and is DISJOINT from the HumanEval training data — that
disjointness is exactly what makes a rising MBPP pass@1 a real *transfer* signal.

Run once from the PieRL working dir (needs internet; caches under ~/.cache/huggingface):
    cd /home/george/agentic-rl/PieRL && python data/prepare_mbpp.py
"""

import json
from pathlib import Path

from datasets import load_dataset


def process_example(example, idx, split):
    tests = list(example["test_list"])
    # Standard zero-shot MBPP prompt: task text + the asserts it must satisfy.
    prompt_text = (
        "You are an expert Python programmer, and here is your task: "
        f"{example['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(tests)
        + "\n"
    )
    ground_truth = json.dumps(
        {
            "test_imports": list(example.get("test_imports", [])),
            "test_list": tests,
        }
    )
    return {
        "data_source": "google-research-datasets/mbpp",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "task_id": example["task_id"]},
    }


def main():
    Path("data").mkdir(exist_ok=True)
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    processed = ds.map(
        lambda ex, idx: process_example(ex, idx, "test"),
        with_indices=True,
        remove_columns=ds.column_names,
    )
    out = "data/mbpp_test.parquet"
    processed.to_parquet(out)
    print(f"Saved {len(processed)} → {out}")


if __name__ == "__main__":
    main()
