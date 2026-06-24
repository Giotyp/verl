"""``train_grpo`` — the standalone GRPO loop (Phase 3).

Wires the verified Phase-1/2 pieces into one step (then a loop):

    sample prompts
      → PieRolloutWorker.generate          (Pie `grpo` inferlet, over WS)
      → pack_dataproto                      (rollout → padded verl DataProto)
      → PieActor.compute_log_prob           (old_log_probs, trainer-side)
      → reward_fn                           (scalar per sample → token-level)
      → compute_grpo_outcome_advantage      (group-relative advantage, by uid)
      → PieActor.update_actor               (GRPO/PPO optimizer step)
      → PieRolloutWorker.update_weights     (push policy → rollout engine, CUDA-IPC)

Run in Pie's ``vllm`` venv, on the SAME GPU as a live Pie server started with
``qwen-rl-config.toml`` (``load_format="dummy"``). See README / status-wsync.md.

    python -m recipe.pie_grpo.train_grpo --config recipe/pie_grpo/config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from recipe.pie_grpo.actor_bootstrap import PieActor
from recipe.pie_grpo.humaneval_reward import compute_reward
from recipe.pie_grpo.pie_rollout_worker import PieRolloutWorker
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.model import compute_position_id_with_mask


# --------------------------------------------------------------------------- #
# Manual #1 (active TODO) — rollout → DataProto                                #
# --------------------------------------------------------------------------- #
def pack_dataproto(
    prompts: list[str],
    gen: list[dict],
    tokenizer,
    *,
    system: str | None = None,
) -> DataProto:
    """Turns Pie's rollout output into a padded verl ``DataProto``."""

    def _prompt_ids(prompt):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return out["input_ids"] if hasattr(out, "keys") else out

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_input_ids = []
    all_attention_mask = []
    all_prompt_rows = []
    all_response_rows = []
    all_response_masks = []
    all_uids = []

    # Pass 1 — tokenize every row (prompt-major) and find the global maxes
    rows = []  # (p_ids, r_ids, uid_value)
    max_p = max_r = 0
    for p in range(len(prompts)):
        p_ids = _prompt_ids(prompts[p])
        for sample in gen[p]["samples"]:
            r_ids = sample["tokens"]
            rows.append((p_ids, r_ids, p))
            max_p = max(max_p, len(p_ids))
            max_r = max(max_r, len(r_ids))

    # Pass 2 — pad each row to the global maxes and accumulate
    for p_ids, r_ids, uid_value in rows:
        n_p, n_r = len(p_ids), len(r_ids)
        prompt_row = [pad_id] * (max_p - n_p) + p_ids  # LEFT pad
        response_row = r_ids + [pad_id] * (max_r - n_r)  # RIGHT pad
        prompt_mask = [0] * (max_p - n_p) + [1] * n_p
        resp_mask = [1] * n_r + [0] * (max_r - n_r)

        all_input_ids.append(prompt_row + response_row)
        all_attention_mask.append(prompt_mask + resp_mask)
        all_prompt_rows.append(prompt_row)  # bug #4 fix
        all_response_rows.append(response_row)
        all_response_masks.append(resp_mask)
        all_uids.append(uid_value)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    prompts_t = torch.tensor(all_prompt_rows, dtype=torch.long)
    responses_t = torch.tensor(all_response_rows, dtype=torch.long)
    response_mask = torch.tensor(all_response_masks, dtype=torch.bool)
    position_ids = compute_position_id_with_mask(attention_mask)  # derives from mask
    uid = np.array(all_uids, dtype=object)

    return DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_t,
            "responses": responses_t,
            "response_mask": response_mask,
            "uid": uid,
        }
    )


# --------------------------------------------------------------------------- #
# Reward (trivial for now — becomes Manual #2 once the loop runs end-to-end)   #
# --------------------------------------------------------------------------- #
def load_humaneval(
    parquet_path: str, n_problems: int, step: int
) -> tuple[list[str], list]:
    """Pull ``n_problems`` HumanEval problems for this step from the prepared parquet
    (RL-Heval's ``humaneval_train.parquet``). Cycles deterministically through the
    file so consecutive steps see different problems.

    Returns ``(prompts, ground_truths)`` aligned by index: ``prompts[p]`` is the
    user-turn problem text, ``ground_truths[p]`` its JSON ``{prompt,test,entry_point}``.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    start = (step * n_problems) % len(df)
    rows = [df.iloc[(start + i) % len(df)] for i in range(n_problems)]
    prompts = [r["prompt"][0]["content"] for r in rows]
    ground_truths = [r["reward_model"]["ground_truth"] for r in rows]
    return prompts, ground_truths


def load_humaneval_fixed(parquet_path: str, n_problems: int | None) -> tuple[list[str], list]:
    """Load a FIXED held-out set (the first ``n_problems``, or all if ``None``) — the
    same problems every eval, so the pass-rate curve reflects the policy, not which
    problems happened to be sampled. Use the *test* parquet (never trained on)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if n_problems is not None:
        df = df.iloc[:n_problems]
    prompts = [r["prompt"][0]["content"] for _, r in df.iterrows()]
    ground_truths = [r["reward_model"]["ground_truth"] for _, r in df.iterrows()]
    return prompts, ground_truths


def reward_fn(gen: list[dict], ground_truths: list) -> list[float]:
    """Per-sample HumanEval unit-test reward, **prompt-major** (must match
    pack_dataproto's row order): for prompt ``p``, score every completion against
    that problem's ground-truth via the sandboxed runner ``compute_reward``.

    ``gen`` is one dict per prompt (``{"samples": [{"text", "tokens"}, ...]}``);
    ``ground_truths[p]`` is the JSON ground-truth for prompt ``p`` (passed straight
    to ``compute_reward`` as its ``ground_truth`` arg).
    Returns a FLAT ``list[float]``of length ``sum(len(g["samples"]))`` in prompt-major
    order: all of prompt 0's samples, then prompt 1's, ... — the exact order pack_dataproto and the advantage
    grouping (``uid``) assume.
    """
    rew_list = []
    for p, per_prompt in enumerate(gen):
        for sample in per_prompt["samples"]:
            rew = compute_reward(sample["text"], ground_truths[p])
            rew_list.append(rew)
    return rew_list


def evaluate(
    worker, prompts, ground_truths, *, system, max_tokens, temperature, top_p
) -> float:
    """Held-out pass@1: one (greedy-ish) completion per FIXED problem, the fraction
    that passes the unit tests. No weight update — pure measurement on problems the
    policy never trains on, so a rising value across steps is unambiguous learning."""
    gen = worker.generate(
        prompts,
        n_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system=system,
    )
    rewards = reward_fn(gen, ground_truths)  # one per prompt (n_samples=1)
    return sum(rewards) / len(rewards)


def _scatter_rewards_to_tokens(
    rewards: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Scalar reward per sample → token-level ``(bsz, resp_len)`` with the reward
    on the last real response token, zeros elsewhere (the shape
    ``compute_grpo_outcome_advantage`` sums over)."""
    token_level = torch.zeros_like(response_mask, dtype=torch.float32)
    last_idx = response_mask.long().sum(dim=1) - 1  # last real token per row
    rows = torch.arange(response_mask.shape[0])
    token_level[rows, last_idx.clamp(min=0)] = rewards.to(torch.float32)
    return token_level


# --------------------------------------------------------------------------- #
# Loop                                                                         #
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> None:
    rollout, actor_cfg = cfg["rollout"], cfg["actor"]

    worker = PieRolloutWorker(
        pie_uri=cfg["pie"]["uri"],
        username=cfg["pie"]["username"],
        grpo_inferlet=cfg["pie"]["grpo_inferlet"],
    )
    actor = PieActor(
        model_path=cfg["model"]["path"],
        attn_implementation=cfg["model"].get("attn_implementation"),
        lr=actor_cfg["lr"],
        clip_ratio=actor_cfg["clip_ratio"],
        entropy_coeff=actor_cfg["entropy_coeff"],
        ppo_mini_batch_size=actor_cfg["ppo_mini_batch_size"],
        micro_batch_size_per_gpu=actor_cfg["micro_batch_size_per_gpu"],
        max_token_len_per_gpu=actor_cfg["max_token_len_per_gpu"],
        rollout_n=rollout["n_samples"],
        master_port=actor_cfg["master_port"],
        optimizer_offload=actor_cfg.get("optimizer_offload", False),
        param_offload=actor_cfg.get("param_offload", False),
        grad_offload=actor_cfg.get("grad_offload", False),
    )
    tokenizer = actor.tokenizer
    eval_cfg = cfg.get("eval", {})

    # Push the actor's real initial weights up front so Pie isn't on dummy weights
    # for the baseline eval / first rollout (also avoids a wasted garbage step 0).
    if cfg["train"]["sync_weights"]:
        worker.update_weights(PieRolloutWorker.extract_hf_state(actor.fsdp_module))

    # Fixed held-out set (same problems every eval → a clean learning curve).
    eval_prompts, eval_gts = load_humaneval_fixed(
        eval_cfg["parquet"], eval_cfg.get("n_problems")
    )

    def run_eval(tag: str) -> float:
        acc = evaluate(
            worker,
            eval_prompts,
            eval_gts,
            system=rollout.get("system"),
            max_tokens=rollout["max_tokens"],
            temperature=eval_cfg.get("temperature", 0.0),
            top_p=eval_cfg.get("top_p", 1.0),
        )
        print(f"[eval {tag}] held-out pass@1 = {acc:.3f}  (n={len(eval_prompts)})")
        return acc

    run_eval("baseline")  # base-model pass@1 before any training

    # Per-step (train) pass-rate — noisy because each step samples DIFFERENT problems;
    # the held-out eval above is the rigorous curve. run_mean smooths the train noise.
    reward_history: list[float] = []
    every = eval_cfg.get("every_steps", 5)
    for step in range(cfg["train"]["num_steps"]):
        # 0. this step's HumanEval problems (prompts + their ground-truths)
        prompts, ground_truths = load_humaneval(
            cfg["data"]["humaneval_parquet"], cfg["data"]["problems_per_step"], step
        )

        # 1. rollout — all n_samples per prompt come back batched
        gen = worker.generate(
            prompts,
            n_samples=rollout["n_samples"],
            max_tokens=rollout["max_tokens"],
            temperature=rollout["temperature"],
            top_p=rollout["top_p"],
            system=rollout.get("system"),
        )

        # 2. pack into a padded DataProto  (Manual #1)
        dp = pack_dataproto(prompts, gen, tokenizer, system=rollout.get("system"))

        # 3. old_log_probs (trainer-side, padded & shifted) — score at the rollout
        #    temperature so the ratio's behaviour policy is the one that sampled.
        dp.batch["old_log_probs"] = actor.compute_log_prob(
            dp, temperature=rollout["temperature"]
        )

        # 4. reward → token-level → group-relative advantage (by uid)
        rewards = torch.tensor(reward_fn(gen, ground_truths))
        response_mask = dp.batch["response_mask"].float()
        token_level = _scatter_rewards_to_tokens(rewards, response_mask)
        uid = dp.non_tensor_batch["uid"]
        advantages, _ = compute_grpo_outcome_advantage(token_level, response_mask, uid)
        dp.batch["advantages"] = advantages

        # 5. one GRPO/PPO optimizer step (same temperature as old_log_probs)
        metrics = actor.update_actor(
            dp, num_mini_batch=1, temperature=rollout["temperature"]
        )

        # 6. push updated policy → live rollout engine (CUDA-IPC, same GPU)
        if cfg["train"]["sync_weights"]:
            worker.update_weights(PieRolloutWorker.extract_hf_state(actor.fsdp_module))

        # --- verification readout ---
        # reward_std > 0 means the group has signal; adv|mean| > 0 means GRPO
        # produced a real gradient; the sample text shows garbage→coherent as the
        # pushed weights take effect on later steps.
        def _scalar(v):
            if isinstance(v, (list, tuple)):
                v = v[0] if v else float("nan")
            if hasattr(v, "value"):  # verl Metric
                v = v.value
            if hasattr(v, "item"):  # tensor
                v = v.item()
            return float(v)

        try:
            pg_loss = _scalar(metrics["metrics"].get("actor/pg_loss"))
        except Exception:
            pg_loss = float("nan")
        rmask_b = response_mask.bool()
        adv_abs = advantages[rmask_b].abs().mean().item()
        rmean = rewards.mean().item()
        reward_history.append(rmean)
        run_mean = sum(reward_history) / len(reward_history)  # cumulative pass-rate
        sample_txt = gen[0]["samples"][0]["text"].replace("\n", " ")[:100]
        print(
            f"[step {step:2d}] reward_mean={rmean:.3f} reward_std={rewards.std():.3f} "
            f"adv|mean|={adv_abs:.4f} pg_loss={pg_loss:+.5f}  run_mean={run_mean:.3f}\n"
            f"           sample: {sample_txt!r}"
        )

        if (step + 1) % every == 0 or step == cfg["train"]["num_steps"] - 1:
            run_eval(f"step {step}")

    # Learning-curve summary: first vs last third of steps (rough — each step uses
    # different problems; a held-out eval set would be the rigorous trend).
    if len(reward_history) >= 3:
        k = len(reward_history) // 3
        first = sum(reward_history[:k]) / k
        last = sum(reward_history[-k:]) / k
        overall = sum(reward_history) / len(reward_history)
        print(
            f"\n[trend] pass-rate first-{k}={first:.3f} -> last-{k}={last:.3f} "
            f"(Δ={last - first:+.3f}, overall {overall:.3f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="recipe/pie_grpo/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
