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
import contextlib
import os

import numpy as np
import torch
import yaml

from recipe.pie_grpo.actor_bootstrap import PieActor
from recipe.pie_grpo.humaneval_reward import compute_reward
from recipe.pie_grpo.mbpp_reward import compute_mbpp_reward
from concurrent.futures import ProcessPoolExecutor
from recipe.pie_grpo.pie_rollout_worker import PieRolloutWorker
from recipe.pie_grpo.topology import (
    check_tp_supported,
    device_idx_for_rank,
    resolve_fsdp_size,
    resolve_world_size,
)
from recipe.pie_grpo.experiments.metrics import RunMetrics


@contextlib.contextmanager
def _nullctx():
    yield


def _phase(rm, name):
    """Time `name` on the RunMetrics collector, or no-op when rm is None (non-rank-0)."""
    return rm.phase(name) if rm is not None else _nullctx()
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


def sync_weights(worker, module, *, device_idx: int) -> None:
    """Push this rank's policy weights into its colocated Pie DP replica.

    ``extract_hf_state`` runs FSDP ``summon_full_params`` — an all-gather
    COLLECTIVE — so the ``list(...)`` below must run on EVERY rank in lockstep (it
    does), or the run deadlocks. After it, every rank holds the full, identical
    model on its own GPU. Under DP each rank then pushes to ITS replica
    (``device_idx == rank``), a same-GPU zero-copy CUDA-IPC transfer — so both
    replicas end up identical. (Phase-1 single-engine code pushed only on rank 0.)
    """
    named = list(PieRolloutWorker.extract_hf_state(module))  # collective on all ranks
    worker.update_weights(named, device_idx=device_idx)


def assign_groups_to_ranks(uid, world_size: int) -> list[list[int]]:
    """Partition row indices into ``world_size`` disjoint lists along WHOLE uid-groups.

    Group-aware data-parallel training: each FSDP rank trains a DISJOINT set of
    prompt groups, so the actor-update compute is split (not duplicated) across
    ranks — the ~2x throughput win. ``uid`` is ``dp.non_tensor_batch["uid"]`` (a
    numpy object array, one entry per row) equal to the prompt index; a prompt's
    ``n_samples`` completions share a uid and are contiguous (see ``pack_dataproto``).

    Return a list of ``world_size`` lists of row indices (into the packed batch),
    one per rank — element ``i`` is the rows rank ``i`` will train on. Two invariants
    make the result CORRECT and DEADLOCK-FREE:

      1. WHOLE groups only. Every row of a given uid must land on the SAME rank —
         GRPO normalizes advantage within a uid group, so a split group silently
         corrupts its mean/std (``compute_grpo_outcome_advantage`` even special-cases
         a size-1 group to mean=0/std=1). Advantages are already computed on the full
         batch before this split, but keeping groups whole also keeps the per-rank
         token bookkeeping honest.
      2. EQUAL row count per rank. Static batching (``use_dynamic_bsz=False``) chunks
         each rank's rows with NO cross-rank count sync, so unequal rows ⇒ ranks issue
         different numbers of FSDP collectives ⇒ hang. Equal groups-per-rank ⇒ equal
         rows (every group is ``n_samples`` rows).

    TODO(human): implement the partition. Suggested approach — take the unique uids
    in first-seen order; assert ``len(groups) % world_size == 0`` (fail loud here
    rather than deadlock later inside the FSDP step); split the group list into
    ``world_size`` equal contiguous blocks; for each rank collect the row indices
    whose uid falls in that rank's block. (If group counts were ever uneven, the plan
    notes a dead-group padding fallback — but the assert is the right default now.)
    """
    # unique uids, in first-seen order
    groups: list = []

    for u in uid:
        if u not in groups:
            groups.append(u)

    assert len(groups) % world_size == 0, (
        f"{len(groups)} groups not divisible by world_size={world_size} — "
        "set problems_per_step to a multiple of world_size (or pad)"
    )
    per_rank = len(groups) // world_size
    row_lists: list[list[int]] = []
    for r in range(world_size):
        block = set(groups[r * per_rank : (r + 1) * per_rank])   # this rank's groups
        rows = [i for i, u in enumerate(uid) if u in block]      # positions
        row_lists.append(rows)
    return row_lists


def scatter_rollout(dp, *, world_size: int, rank: int, src: int = 0):
    """Scatter rank-``src``'s per-group DataProto shards to every FSDP rank.

    Replaces the old ``broadcast_rollout`` (which duplicated the FULL batch on every
    rank — memory-parallel, not data-parallel). Only rank ``src`` ran the Pie rollout
    and holds the packed ``dp`` with advantages already baked in; it partitions the
    batch into ``world_size`` whole-group shards (``assign_groups_to_ranks`` +
    ``DataProto.select_idxs``) and scatters them, so each rank returns ONLY its shard.
    Collective: every rank must call it in lockstep with the same ``src``.
    """
    if world_size == 1:
        return dp
    if rank == src:
        row_lists = assign_groups_to_ranks(dp.non_tensor_batch["uid"], world_size)
        shards = [dp.select_idxs(rows) for rows in row_lists]
    else:
        shards = None  # ignored on non-src ranks
    out: list = [None]
    torch.distributed.scatter_object_list(out, shards, src=src)
    return out[0]


# --------------------------------------------------------------------------- #
# Loop                                                                         #
# --------------------------------------------------------------------------- #
def train(cfg: dict, metrics_out: str | None = None) -> None:
    # Stage B: topology comes from the (optional) `topology:` config block; defaults
    # reproduce the launcher-driven DP=2 path (world/fsdp = WORLD_SIZE, identity map).
    topo = cfg.get("topology")
    rank, _ = _dist_info()
    world_size = resolve_world_size(topo)
    fsdp_size = resolve_fsdp_size(topo, world_size)
    # Stage C seam: tp_size=1 is the full-model-per-replica path (below unchanged);
    # tp_size>1 is refused loudly (not implemented + needs NVLink) — see check_tp_supported.
    tp_size = check_tp_supported(cfg.get("pie"))
    is_main = rank == 0
    rollout, actor_cfg = cfg["rollout"], cfg["actor"]

    # Experiment metrics (rank 0 only; None -> no-op everywhere via _phase()).
    run_metrics = None
    if is_main and metrics_out:
        run_metrics = RunMetrics(
            backend="pie",
            run_meta={"model": cfg["model"]["path"], "n_gpus": world_size,
                      "gpu": os.environ.get("PIE_EXP_GPU", "unknown"),
                      "num_steps": cfg["train"]["num_steps"],
                      "hyperparams": {"n_samples": rollout["n_samples"],
                                      "lr": actor_cfg["lr"],
                                      "temperature": rollout["temperature"]}},
            out_path=metrics_out)

    # Rank 0 drives generation + eval (Pie auto-balances the launches across both
    # DP replicas); EVERY rank builds the FSDP actor, runs the collectives
    # (compute_log_prob / update_actor / weight-extract) in lockstep, AND pushes its
    # weights to its replica via `device_idx_for_rank` (identity == rank → same-GPU
    # CUDA-IPC) so both replicas stay in sync.
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
        fsdp_size=fsdp_size,
        optimizer_offload=actor_cfg.get("optimizer_offload", False),
        param_offload=actor_cfg.get("param_offload", False),
        grad_offload=actor_cfg.get("grad_offload", False),
    )
    tokenizer = actor.tokenizer
    eval_cfg = cfg.get("eval", {})

    # Push the actor's real initial weights up front so Pie isn't on dummy weights
    # for the baseline eval / first rollout. Collective on all ranks; rank 0 pushes.
    if cfg["train"]["sync_weights"]:
        sync_weights(worker, actor.fsdp_module, device_idx=device_idx_for_rank(topo, rank))

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

        def run_eval(tag: str):
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
            mbpp_acc = None
            if mbpp_prompts:
                mbpp_acc = evaluate(
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
                line += f"  |  mbpp pass@1 = {mbpp_acc:.3f}  (n={len(mbpp_prompts)})"
            print(line)
            return acc, mbpp_acc

        with _phase(run_metrics, "eval"):
            heval0, mbpp0 = run_eval("baseline")  # base-model pass@1 before training
        if run_metrics:
            run_metrics.record_eval(-1, heval0, mbpp0)

    # Per-step (train) pass-rate — noisy because each step samples DIFFERENT problems;
    # the held-out eval above is the rigorous curve. run_mean smooths the train noise.
    reward_history: list[float] = []
    every = eval_cfg.get("every_steps", 5)
    for step in range(cfg["train"]["num_steps"]):
        if run_metrics:
            run_metrics.start_step()
        # 0-2. rank 0 only: problems → rollout → reward → FULL-batch advantages.
        #      Generation is already parallel (Pie DP=2 auto-balances the launches);
        #      the advantage math is model-free, so rank 0 computes it alone on the
        #      WHOLE batch BEFORE the group-aware scatter — splitting a uid group
        #      first would corrupt its group-relative mean/std.
        if is_main:
            prompts, ground_truths = load_humaneval(
                cfg["data"]["humaneval_parquet"], cfg["data"]["problems_per_step"], step
            )
            with _phase(run_metrics, "gen"):
                gen = worker.generate(
                    prompts,
                    n_samples=rollout["n_samples"],
                    max_tokens=rollout["max_tokens"],
                    temperature=rollout["temperature"],
                    top_p=rollout["top_p"],
                    system=rollout.get("system"),
                )
            if run_metrics:
                run_metrics.record_gen_tokens(
                    sum(len(s["tokens"]) for g in gen for s in g["samples"]))
            # pack into a padded DataProto  (Manual #1)
            dp = pack_dataproto(prompts, gen, tokenizer, system=rollout.get("system"))
            rewards = torch.tensor(reward_fn(gen, ground_truths))
            # reward → token-level → group-relative advantage (by uid), full batch.
            with _phase(run_metrics, "adv"):
                response_mask = dp.batch["response_mask"].float()
                token_level = _scatter_rewards_to_tokens(rewards, response_mask)
                uid = dp.non_tensor_batch["uid"]
                advantages, _ = compute_grpo_outcome_advantage(
                    token_level, response_mask, uid
                )
                dp.batch["advantages"] = advantages
        else:
            gen = dp = rewards = advantages = response_mask = None

        # 2b. scatter WHOLE uid-groups: each rank trains a DISJOINT shard (not the
        #     duplicated full batch). Advantages are already baked into dp on rank 0.
        local_dp = scatter_rollout(dp, world_size=world_size, rank=rank)

        # 3. old_log_probs on the LOCAL shard — COLLECTIVE (all ranks), scored at the
        #    rollout temperature so the ratio's behaviour policy is the one that sampled.
        with _phase(run_metrics, "logprob"):
            local_dp.batch["old_log_probs"] = actor.compute_log_prob(
                local_dp, temperature=rollout["temperature"]
            )

        # 4. one GRPO/PPO optimizer step on the LOCAL shard — COLLECTIVE. Grads
        #    reduce-scatter over WORLD and the engine all-reduces batch_num_tokens, so
        #    the gradient is the true global-token-mean over both shards (Stage A note).
        with _phase(run_metrics, "update"):
            metrics = actor.update_actor(
                local_dp, num_mini_batch=1, temperature=rollout["temperature"]
            )

        # 5. push updated policy → live rollout engine. Barrier so rank 0 doesn't
        #    extract mid-step; collective extract on all ranks, each pushes to its own
        #    replica. All ranks hold identical weights after the FSDP step regardless
        #    of which shard they trained on.
        if world_size > 1:
            torch.distributed.barrier()
        if cfg["train"]["sync_weights"]:
            with _phase(run_metrics, "weight_sync"):
                sync_weights(worker, actor.fsdp_module,
                             device_idx=device_idx_for_rank(topo, rank))

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
            if run_metrics:
                run_metrics.record_train(rmean, pg_loss)
            sample_txt = gen[0]["samples"][0]["text"].replace("\n", " ")[:100]
            print(
                f"[step {step:2d}] reward_mean={rmean:.3f} reward_std={rewards.float().std():.3f} "
                f"adv|mean|={adv_abs:.4f} pg_loss={pg_loss:+.5f}  run_mean={run_mean:.3f}\n"
                f"           sample: {sample_txt!r}"
            )

            if (step + 1) % every == 0 or step == cfg["train"]["num_steps"] - 1:
                with _phase(run_metrics, "eval"):
                    heval_s, mbpp_s = run_eval(f"step {step}")
                if run_metrics:
                    run_metrics.record_eval(step, heval_s, mbpp_s)

        if run_metrics:
            run_metrics.end_step()

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

    if run_metrics:
        run_metrics.record_peak_mem([torch.cuda.max_memory_allocated() / 1e9])
        run_metrics.finalize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="recipe/pie_grpo/config.yaml")
    ap.add_argument("--metrics-out", default=None,
                    help="write experiment metrics.json to this path (rank 0)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, metrics_out=args.metrics_out)


if __name__ == "__main__":
    main()
