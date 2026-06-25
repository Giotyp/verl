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
import os

import numpy as np
import torch
import yaml

from recipe.pie_grpo.actor_bootstrap import PieActor
from recipe.pie_grpo.humaneval_reward import compute_reward
from recipe.pie_grpo.mbpp_reward import compute_mbpp_reward
from concurrent.futures import ProcessPoolExecutor
from recipe.pie_grpo.pie_rollout_worker import PieRolloutWorker
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.model import compute_position_id_with_mask


# --------------------------------------------------------------------------- #
# rollout → DataProto                                #
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


def load_humaneval_fixed(
    parquet_path: str, n_problems: int | None
) -> tuple[list[str], list]:
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


def _reward_for_prompt(job: tuple[list[dict], object]) -> list[float]:
    samples, ground_truth = job
    return [compute_reward(sample["text"], ground_truth) for sample in samples]


def reward_fn(gen: list[dict], ground_truths: list, workers: int = 1) -> list[float]:
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
    jobs = [
        ([sample for sample in per_prompt["samples"]], ground_truths[p])
        for p, per_prompt in enumerate(gen)
    ]
    if not jobs:
        return []

    max_workers = min(len(jobs), workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        prompt_rewards = list(executor.map(_reward_for_prompt, jobs))

    return [
        reward
        for prompt_rewards_per_prompt in prompt_rewards
        for reward in prompt_rewards_per_prompt
    ]


def _mbpp_reward_for_prompt(job: tuple[list[dict], object]) -> list[float]:
    samples, ground_truth = job
    return [compute_mbpp_reward(sample["text"], ground_truth) for sample in samples]


def mbpp_reward_fn(
    gen: list[dict], ground_truths: list, workers: int = 1
) -> list[float]:
    """MBPP transfer-eval reward — prompt-major, mirrors ``reward_fn`` but scores
    each completion with ``compute_mbpp_reward`` (the disjoint-benchmark signal)."""
    jobs = [
        ([sample for sample in per_prompt["samples"]], ground_truths[p])
        for p, per_prompt in enumerate(gen)
    ]
    if not jobs:
        return []

    max_workers = min(len(jobs), workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        prompt_rewards = list(executor.map(_mbpp_reward_for_prompt, jobs))

    return [
        reward
        for prompt_rewards_per_prompt in prompt_rewards
        for reward in prompt_rewards_per_prompt
    ]


def evaluate(
    worker,
    prompts,
    ground_truths,
    *,
    reward=reward_fn,
    max_workers,
    system,
    max_tokens,
    temperature,
    top_p,
) -> float:
    """Held-out pass@1: one (greedy-ish) completion per FIXED problem, the fraction
    that passes the unit tests. No weight update — pure measurement on problems the
    policy never trains on, so a rising value across steps is unambiguous learning.

    ``reward`` selects the scorer: ``reward_fn`` (HumanEval) or ``mbpp_reward_fn``."""
    gen = worker.generate(
        prompts,
        n_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system=system,
    )
    rewards = reward(
        gen, ground_truths, workers=max_workers
    )  # one per prompt (n_samples=1)
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
# Distributed helpers (SPMD under torchrun)                                    #
# --------------------------------------------------------------------------- #
def _dist_info() -> tuple[int, int]:
    """``(rank, world_size)`` from the launcher's env (torchrun) — or (0, 1)."""
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def sync_weights(worker, module, *, rank: int) -> None:
    """Push the actor's current policy into the live Pie rollout engine.

    ``extract_hf_state`` runs FSDP ``summon_full_params`` — an all-gather
    COLLECTIVE — so EVERY rank must drive it in lockstep or the run deadlocks.
    ``list(...)`` forces the generator to completion inside the summon context on
    all ranks (collective balanced), and the bf16 ``.clone()`` it does inside the
    context means the gathered tensors survive after the context exits (FSDP frees
    its flat-param shards, but our clones are independent). Only rank 0 shares a
    physical GPU with the engine, so only rank 0 packs the (same-GPU, zero-copy)
    CUDA-IPC handles and pushes them over the WebSocket.
    """
    named = list(PieRolloutWorker.extract_hf_state(module))  # collective on all ranks
    if rank == 0:
        worker.update_weights(named)
    del named


def broadcast_rollout(dp, rewards, *, world_size: int, src: int = 0):
    """Replicate rank-``src``'s rollout batch to EVERY FSDP rank.

    FSDP shards the *model + optimizer* across GPUs (the memory win that fits a
    >0.5B actor); we feed the *same data* to every rank so the forward/backward
    collectives line up and grads reduce-scatter consistently. This is
    memory-parallel, not data-parallel — a DP batch-split (≈2x throughput) is a
    later optimization, not needed to clear the 0.5B ceiling.

    Only rank ``src`` ran the Pie rollout, so it holds the packed ``DataProto``
    ``dp`` and the per-sample reward tensor ``rewards``; on every other rank they
    arrive as ``None`` and must be filled from the broadcast. Returns the
    rank-local ``(dp, rewards)``.
    """
    if world_size == 1:
        return dp, rewards
    # broadcast_object_list is COLLECTIVE: every rank calls it identically with the
    # same ``src``; the API sends from ``src`` and overwrites the list IN PLACE on
    # the others. So there's no sender/receiver branch — just capture the in-place
    # fill back out of the named list (returning the local names would hand back the
    # un-filled None on non-src ranks).
    obj = [dp, rewards]
    torch.distributed.broadcast_object_list(obj, src=src)
    dp, rewards = obj
    return dp, rewards


# --------------------------------------------------------------------------- #
# Loop                                                                         #
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> None:
    rank, world_size = _dist_info()
    is_main = rank == 0
    rollout, actor_cfg = cfg["rollout"], cfg["actor"]

    # Only rank 0 talks to Pie (generation, weight-push, eval). EVERY rank builds
    # the FSDP actor and runs the collectives (compute_log_prob / update_actor /
    # weight-extract) — those must stay in lockstep across ranks.
    worker = (
        PieRolloutWorker(
            pie_uri=cfg["pie"]["uri"],
            username=cfg["pie"]["username"],
            grpo_inferlet=cfg["pie"]["grpo_inferlet"],
        )
        if is_main
        else None
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
    # for the baseline eval / first rollout. Collective on all ranks; rank 0 pushes.
    if cfg["train"]["sync_weights"]:
        sync_weights(worker, actor.fsdp_module, rank=rank)

    # Fixed held-out sets + eval closure live on rank 0 (the only Pie talker).
    if is_main:
        eval_prompts, eval_gts = load_humaneval_fixed(
            eval_cfg["parquet"], eval_cfg.get("n_problems")
        )
        # Optional MBPP transfer eval — same parquet schema, so reuse the loader.
        mbpp_prompts, mbpp_gts = [], []
        mbpp_parquet = eval_cfg.get("mbpp_parquet")
        if mbpp_parquet and os.path.exists(mbpp_parquet):
            mbpp_prompts, mbpp_gts = load_humaneval_fixed(
                mbpp_parquet, eval_cfg.get("mbpp_n_problems")
            )

        def run_eval(tag: str) -> float:
            acc = evaluate(
                worker,
                eval_prompts,
                eval_gts,
                reward=reward_fn,
                max_workers=eval_cfg.get("max_workers", 1),
                system=rollout.get("system"),
                max_tokens=rollout["max_tokens"],
                temperature=eval_cfg.get("temperature", 0.0),
                top_p=eval_cfg.get("top_p", 1.0),
            )
            line = f"[eval {tag}] heval pass@1 = {acc:.3f}  (n={len(eval_prompts)})"
            if mbpp_prompts:
                m = evaluate(
                    worker,
                    mbpp_prompts,
                    mbpp_gts,
                    reward=mbpp_reward_fn,
                    max_workers=eval_cfg.get("max_workers", 1),
                    system=rollout.get("system"),
                    max_tokens=rollout["max_tokens"],
                    temperature=eval_cfg.get("temperature", 0.0),
                    top_p=eval_cfg.get("top_p", 1.0),
                )
                line += f"  |  mbpp pass@1 = {m:.3f}  (n={len(mbpp_prompts)})"
            print(line)
            return acc

        run_eval("baseline")  # base-model pass@1 before any training

    # Per-step (train) pass-rate — noisy because each step samples DIFFERENT problems;
    # the held-out eval above is the rigorous curve. run_mean smooths the train noise.
    reward_history: list[float] = []
    every = eval_cfg.get("every_steps", 5)
    for step in range(cfg["train"]["num_steps"]):
        # 0-1. rank 0 only: this step's problems → rollout → reward (all Pie I/O).
        if is_main:
            prompts, ground_truths = load_humaneval(
                cfg["data"]["humaneval_parquet"], cfg["data"]["problems_per_step"], step
            )
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
            rewards = torch.tensor(reward_fn(gen, ground_truths))
        else:
            gen = dp = rewards = None

        # 2b. replicate rank-0's batch to every FSDP rank (Learn-by-Doing).
        dp, rewards = broadcast_rollout(dp, rewards, world_size=world_size)

        # 3. old_log_probs — COLLECTIVE (all ranks), scored at the rollout
        #    temperature so the ratio's behaviour policy is the one that sampled.
        dp.batch["old_log_probs"] = actor.compute_log_prob(
            dp, temperature=rollout["temperature"]
        )

        # 4. reward → token-level → group-relative advantage (by uid). Deterministic
        #    and identical on every rank (same dp + rewards) → no extra broadcast.
        response_mask = dp.batch["response_mask"].float()
        token_level = _scatter_rewards_to_tokens(rewards, response_mask)
        uid = dp.non_tensor_batch["uid"]
        advantages, _ = compute_grpo_outcome_advantage(token_level, response_mask, uid)
        dp.batch["advantages"] = advantages

        # 5. one GRPO/PPO optimizer step — COLLECTIVE (all ranks).
        metrics = actor.update_actor(
            dp, num_mini_batch=1, temperature=rollout["temperature"]
        )

        # 6. push updated policy → live rollout engine. Barrier so rank 0 doesn't
        #    extract mid-step; collective extract on all ranks, rank 0 pushes.
        if world_size > 1:
            torch.distributed.barrier()
        if cfg["train"]["sync_weights"]:
            sync_weights(worker, actor.fsdp_module, rank=rank)

        # --- verification readout (rank 0 only) ---
        # reward_std > 0 means the group has signal; adv|mean| > 0 means GRPO
        # produced a real gradient; the sample text shows garbage→coherent as the
        # pushed weights take effect on later steps.
        if is_main:

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
            rmean = rewards.float().mean().item()
            reward_history.append(rmean)
            run_mean = sum(reward_history) / len(reward_history)  # cumulative pass-rate
            sample_txt = gen[0]["samples"][0]["text"].replace("\n", " ")[:100]
            print(
                f"[step {step:2d}] reward_mean={rmean:.3f} reward_std={rewards.float().std():.3f} "
                f"adv|mean|={adv_abs:.4f} pg_loss={pg_loss:+.5f}  run_mean={run_mean:.3f}\n"
                f"           sample: {sample_txt!r}"
            )

            if (step + 1) % every == 0 or step == cfg["train"]["num_steps"] - 1:
                run_eval(f"step {step}")

    # Learning-curve summary: first vs last third of steps (rough — each step uses
    # different problems; a held-out eval set would be the rigorous trend).
    if is_main and len(reward_history) >= 3:
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
