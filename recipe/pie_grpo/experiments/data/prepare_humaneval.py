import json
from pathlib import Path
from datasets import load_dataset

def process_example(example, idx, split):
    ground_truth = json.dumps({
        "prompt": example["prompt"],
        "test": example["test"],
        "entry_point": example["entry_point"],
    })
    return {
        "data_source": "openai/openai_humaneval",
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "task_id": example["task_id"]},
    }

def main():
    Path("data").mkdir(exist_ok=True)
    ds = load_dataset("openai/openai_humaneval", split="test")
    splits = ds.train_test_split(test_size=0.2, seed=42)
    for name, subset in [("train", splits["train"]), ("test", splits["test"])]:
        processed = subset.map(
            lambda ex, idx: process_example(ex, idx, name),
            with_indices=True,
            remove_columns=subset.column_names,
        )
        out = f"data/humaneval_{name}.parquet"
        processed.to_parquet(out)
        print(f"Saved {len(processed)} → {out}")

if __name__ == "__main__":
    main()
