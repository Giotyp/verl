"""``actor_bootstrap`` — stand up a verl FSDP actor *outside* Ray (Phase 2).

Phase 1 gave us :class:`~recipe.pie_grpo.pie_rollout_worker.PieRolloutWorker`
(rollout + weight-push against a live Pie server). Phase 2 builds the other half
of the loop — the *trainer-side policy* — and exposes the two methods the GRPO
step needs:

* :meth:`PieActor.compute_log_prob` → per-token log-probs of the responses
  (the ``old_log_prob`` term, and later the policy ratio).
* :meth:`PieActor.update_actor` → one GRPO/PPO optimizer step.

It also surfaces the FSDP-wrapped module (:attr:`PieActor.fsdp_module`) and the
engine (:attr:`PieActor.engine`) so ``extract_hf_state`` (Manual #3) can gather
the full, HF-named, bf16 weight stream that ``PieRolloutWorker.update_weights``
pushes into Pie over CUDA-IPC.

Design notes (see the plan, Phase 2):

* We drive verl's :class:`TrainingWorker` *directly* rather than
  ``ActorRolloutRefWorker``. The latter, even with ``role="actor"``, still
  constructs a checkpoint engine from ``config.rollout.checkpoint_engine``
  (``engine_workers.py:622``) — pure overhead here. ``TrainingWorker`` is the
  minimal FSDP-model + AdamW + loss-fn wrapper; its ``@register`` dispatch
  decorators are no-ops on a direct (non-RayWorkerGroup) call.
* ``TrainingWorker.__init__`` calls ``initialize_global_process_group_ray``,
  which despite the name just reads ``RANK``/``WORLD_SIZE``/``MASTER_ADDR`` from
  the environment and calls ``torch.distributed.init_process_group`` — no Ray.
  :func:`ensure_dist_ws1` sets those env vars for the single-process case.

**Run in Pie's ``vllm`` venv, on the SAME physical GPU as the driver.** The actor
shares the GPU with Pie's vLLM engine: the trainer's torch must equal the
driver's torch (CUDA-IPC ``reduce_tensor`` rebuild-tuple layout), and the two
must fit together in memory (Pie at ``gpu_memory_utilization=0.4``; this adds the
1.5B model + AdamW state ≈ 2x the model + FSDP overhead — watch OOM).
"""

from __future__ import annotations

import importlib.util
import os
from functools import partial

import torch

from verl.utils import tensordict_utils as tu
from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    FSDPActorConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# Side effect: route verl's unpad_input through transformers' pure-torch impl so
# the padded↔nested conversion works in the flash_attn-less Pie `vllm` venv.
from recipe.pie_grpo import _flash_attn_compat  # noqa: F401
from recipe.pie_grpo.topology import device_for_rank


def ensure_dist_ws1(master_addr: str = "127.0.0.1", master_port: int = 29512) -> None:
    """Seed the env for a single-process group AND bind this rank to its own GPU.

    ``TrainingWorker.__init__`` does the actual ``init_process_group`` (reading
    these vars); we only seed them, and only if torch.distributed isn't already
    up. ``setdefault`` means an outer launcher (torchrun, a real multi-GPU job)
    that already exported ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` wins — we never
    clobber a real distributed env, so the same path serves WS=1 and torchrun WS>1.

    NOTE: the init path ``TrainingWorker`` takes
    (``initialize_global_process_group_ray``) does NOT call ``set_device`` —
    unlike the non-Ray ``initialize_global_process_group``. Under torchrun every
    rank would then default to ``cuda:0`` and FSDP would pile all ranks onto one
    GPU (no sharding + the colocation OOM we're escaping). So bind
    ``LOCAL_RANK -> device`` here, before the process group inits.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(device_for_rank(None, local_rank))
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))


def _default_attn_implementation() -> str:
    """Pick a model attention backend that is available in this environment."""
    return (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "eager"
    )


def _resolve_attn_implementation(attn_implementation: str | None) -> str:
    """Resolve an explicit backend or fall back to the best available one."""
    if attn_implementation and attn_implementation != "auto":
        return attn_implementation
    return _default_attn_implementation()


def _build_configs(
    model_path: str,
    *,
    attn_implementation: str | None,
    lr: float,
    clip_ratio: float,
    entropy_coeff: float,
    ppo_mini_batch_size: int,
    micro_batch_size_per_gpu: int,
    max_token_len_per_gpu: int,
    rollout_n: int,
    fsdp_size: int | None = None,
    param_offload: bool = False,
    optimizer_offload: bool = False,
    grad_offload: bool = False,
) -> tuple[TrainingWorkerConfig, FSDPActorConfig]:
    """Assemble the verl dataclass configs for a WS=1 FSDP actor.

    Returns ``(training_worker_config, actor_config)``. The first builds the
    worker/engine; the second parameterizes ``ppo_loss`` (clip ratio, entropy,
    batch bookkeeping). The engine-config batch fields are assigned exactly as
    ``ActorRolloutRefWorker.init_model`` does (``engine_workers.py:563-574``) so
    we stay on verl's proven path instead of guessing field names.
    """
    # HFModelConfig.__post_init__ loads hf_config + tokenizer from `path`, so a
    # bare path is enough — no need to hand-build either.
    model_config = HFModelConfig(
        path=model_path,
        model_type="language_model",
        override_config={
            "attn_implementation": _resolve_attn_implementation(attn_implementation)
        },
    )

    engine_config = FSDPEngineConfig(
        strategy="fsdp",
        # Shard params+grads+optimizer across `fsdp_size` ranks (Stage B: resolved
        # from the topology config, falling back to the launcher's WORLD_SIZE).
        # WS=1 -> a formality (no real sharding); torchrun WS>1 -> the memory win
        # that lets a >0.5B actor fit alongside a colocated Pie engine.
        fsdp_size=(
            fsdp_size if fsdp_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
        ),
        model_dtype="bf16",  # rollout/compute dtype.
        # CPU-offload to fit the trainer alongside Pie's colocated vLLM engine.
        # AdamW states (~12GB for 1.5B fp32 m+v) + master copy dwarf the model;
        # parking them in host RAM is what makes single-GPU colocation feasible.
        param_offload=param_offload,
        optimizer_offload=optimizer_offload,
        grad_offload=grad_offload,
    )
    # Static (non-dynamic) batching keeps shapes predictable for the prototype.
    engine_config.use_dynamic_bsz = False
    engine_config.micro_batch_size_per_gpu = micro_batch_size_per_gpu
    engine_config.max_token_len_per_gpu = max_token_len_per_gpu
    engine_config.infer_micro_batch_size_per_gpu = micro_batch_size_per_gpu
    engine_config.infer_max_token_len_per_gpu = max_token_len_per_gpu
    engine_config.use_remove_padding = False
    # forward_only stays False — we need train_batch (the optimizer step).

    optimizer_config = FSDPOptimizerConfig(lr=lr)  # __post_init__ asserts lr set.
    checkpoint_config = CheckpointConfig()

    worker_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )

    # Config object handed to ppo_loss. rollout_n is MISSING by default and must
    # be set; global_batch_info defaults to {} (ppo_loss writes dp_size etc into
    # it at train time), loss_scale_factor defaults to None.
    actor_config = FSDPActorConfig(
        strategy="fsdp",
        clip_ratio=clip_ratio,
        entropy_coeff=entropy_coeff,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_dynamic_bsz=False,
        rollout_n=rollout_n,
    )
    actor_config.model_config = model_config
    return worker_config, actor_config


class PieActor:
    """Thin handle over a WS=1 verl ``TrainingWorker`` for the GRPO loop."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        *,
        attn_implementation: str | None = None,
        lr: float = 1e-6,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.0,
        ppo_mini_batch_size: int = 16,
        micro_batch_size_per_gpu: int = 4,
        max_token_len_per_gpu: int = 8192,
        rollout_n: int = 8,
        master_port: int = 29512,
        fsdp_size: int | None = None,
        param_offload: bool = False,
        optimizer_offload: bool = False,
        grad_offload: bool = False,
    ) -> None:
        ensure_dist_ws1(master_port=master_port)

        worker_config, actor_config = _build_configs(
            model_path,
            attn_implementation=attn_implementation,
            lr=lr,
            clip_ratio=clip_ratio,
            entropy_coeff=entropy_coeff,
            ppo_mini_batch_size=ppo_mini_batch_size,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            max_token_len_per_gpu=max_token_len_per_gpu,
            rollout_n=rollout_n,
            fsdp_size=fsdp_size,
            param_offload=param_offload,
            optimizer_offload=optimizer_offload,
            grad_offload=grad_offload,
        )
        self.actor_config = actor_config
        self.model_config = worker_config.model_config

        # Build (constructs the engine) then reset() (engine.initialize():
        # FSDP-wrap the HF model + create the AdamW optimizer).
        self.worker = TrainingWorker(worker_config)
        self.worker.reset()
        self.worker.set_loss_fn(partial(ppo_loss, config=actor_config))

    # ------------------------------------------------------------------ #
    # Handles for weight extraction (Manual #3 consumes these)           #
    # ------------------------------------------------------------------ #
    @property
    def engine(self):
        """The verl FSDPEngine. Exposes ``get_per_tensor_param()`` — verl's own
        full-param, HF-name-normalized, bf16 weight generator (the recommended
        backing for ``extract_hf_state``'s FSDP branch)."""
        return self.worker.engine

    @property
    def fsdp_module(self) -> torch.nn.Module:
        """The FSDP-wrapped policy module. Pass this to
        ``PieRolloutWorker.extract_hf_state`` for the manual ``summon_full_params``
        route (Manual #3)."""
        return self.worker.engine.module

    @property
    def tokenizer(self):
        return self.model_config.tokenizer

    # ------------------------------------------------------------------ #
    # The two methods the GRPO step calls                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nested_td(data):
        """Padded ``DataProto``/``TensorDict`` → the nested (jagged) TensorDict the
        engine consumes.

        We clone first because ``left_right_2_no_padding`` is *destructive* (it
        pops ``input_ids`` and reassigns keys); cloning keeps the caller's
        ``DataProto`` reusable across the two passes a GRPO step makes (one for
        ``compute_log_prob``, one for ``update_actor``). Mirrors the proven path
        in verl's ``tests/models/test_engine.py``.
        """
        batch = data.batch if hasattr(data, "batch") else data
        return left_right_2_no_padding(batch.clone())

    def compute_log_prob(self, data, *, temperature: float = 1.0):
        """Per-token log-probs of the responses (the ``old_log_prob`` term).

        Runs ``infer_batch`` with ``compute_loss=False`` (we want only
        ``model_output['log_probs']``), then ``no_padding_2_padding`` to recover a
        padded ``(bsz, max_response_len)`` tensor — already left-shifted by one for
        next-token alignment — so the loop can store it straight into the batch.

        ``temperature`` must match the rollout sampling temperature: the engine
        forward scores logits at ``logits / temperature`` (``transformer_impl.py``),
        so a mismatch makes ``old_log_probs`` describe the wrong distribution.
        """
        td = self._nested_td(data)
        tu.assign_non_tensor(td, compute_loss=False, temperature=float(temperature))
        out = self.worker.infer_batch(td)
        return no_padding_2_padding(tu.get(out, "log_probs"), td)

    @staticmethod
    def _global_batch_size(local_rows: int) -> int:
        """Sum the per-rank row count across the data-parallel world.

        Under group-aware data-parallel training each rank holds only its DISJOINT
        shard, so ``td.shape[0]`` is a LOCAL count. ``ppo_loss``'s seq-mean loss-agg
        modes normalize by ``global_batch_size``; feeding the local count would
        under-normalize. (The default ``token-mean`` mode uses the engine's all-reduced
        ``batch_num_tokens`` instead and is unaffected — this keeps the value correct
        if the mode ever changes.) Reduces to a no-op when not distributed.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return local_rows
        t = torch.tensor([local_rows], device=torch.cuda.current_device())
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return int(t.item())

    def update_actor(
        self,
        data,
        *,
        num_mini_batch: int = 1,
        epochs: int = 1,
        temperature: float = 1.0,
    ):
        """One GRPO/PPO optimizer step over the batch.

        The batch must already carry ``old_log_probs`` and ``advantages`` (padded,
        response-shaped). ``ppo_loss`` reads ``global_batch_size`` off the
        TensorDict for loss normalization (``losses.py:67``); ``dp_size`` and
        ``batch_num_tokens`` are injected by the engine. ``temperature`` must match
        the value used for ``compute_log_prob`` (and the rollout). We default to a
        single mini-batch (full-batch update) for the prototype.
        """
        td = self._nested_td(data)
        tu.assign_non_tensor(
            td,
            num_mini_batch=num_mini_batch,
            epochs=epochs,
            global_batch_size=self._global_batch_size(td.shape[0]),
            temperature=float(temperature),
        )
        return self.worker.train_mini_batch(td)
