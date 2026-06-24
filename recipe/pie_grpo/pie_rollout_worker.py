"""``PieRolloutWorker`` — verl ⇄ Pie rollout adapter (Phase 1).

This is the bridge between a standalone verl-based GRPO trainer and a *live* Pie
server colocated on the same GPU as the trainer:

* :meth:`generate` runs the ``grpo`` inferlet over Pie's WebSocket. One launched
  process per prompt returns **all** ``n_samples`` completions (Pie forks a
  shared system+user prefix in O(1) via a GPU D2D KV copy — the whole point of
  going through inferlets instead of verl's per-request HTTP rollout).
* :meth:`update_weights` pushes freshly-trained policy weights into the running
  rollout engine via CUDA-IPC handles (zero-copy, same GPU). The handle-packing
  logic is ported verbatim from the proven ``weight_sync/trainer_sync.py``.
* :meth:`extract_hf_state` turns the trainer's model into the HF-named, bf16,
  GPU-resident parameter stream that :meth:`update_weights` consumes. This is the
  one correctness-critical piece left as a guided ``TODO(human)``.

The Pie client is async; the trainer loop is sync. For the prototype we bridge
with ``asyncio.run`` per call (a single long-lived loop is the later
optimization noted in the plan).

Run this in Pie's ``vllm`` venv so the trainer's torch == the driver's torch:
the CUDA-IPC rebuild tuple (``reduce_tensor`` args, esp. the ``args[6]``
storage-device index) must match between sender and receiver.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from typing import Iterable

import torch
from torch.multiprocessing.reductions import reduce_tensor

from pie_client import PieClient


class PieRolloutWorker:
    """Drives generation and weight-sync against a colocated Pie server."""

    def __init__(
        self,
        pie_uri: str = "ws://127.0.0.1:8080",
        username: str = "rl-trainer",
        grpo_inferlet: str = "grpo@0.1.0",
    ) -> None:
        self.pie_uri = pie_uri
        self.username = username
        self.grpo_inferlet = grpo_inferlet

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #
    async def _generate_async(
        self,
        prompts: list[str],
        n_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        system: str | None,
    ) -> list[dict]:
        # One input dict per prompt; the grpo inferlet returns all n_samples
        # for that prompt from a single launched process.
        inputs: list[dict] = []
        for prompt in prompts:
            inp = {
                "prompt": prompt,
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if system is not None:
                inp["system"] = system
            inputs.append(inp)

        async with PieClient(self.pie_uri) as client:
            await client.authenticate(self.username)
            # run_processes launches one process per input and collects each
            # inferlet's Return payload in a single control-plane round-trip.
            payloads = await client.run_processes(self.grpo_inferlet, inputs)

        # Each payload is the grpo inferlet's JSON string:
        #   {"samples": [{"text": str, "tokens": [int, ...]}, ...]}
        return [json.loads(p) for p in payloads]

    def generate(
        self,
        prompts: list[str],
        n_samples: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        system: str | None = None,
    ) -> list[dict]:
        """Generate ``n_samples`` completions for each prompt.

        Returns one dict per prompt, ``{"samples": [{"text", "tokens"}, ...]}``,
        with samples kept batched (do not flatten here — the trainer needs the
        per-prompt grouping to assign GRPO ``uid``\\ s downstream).
        """
        return asyncio.run(
            self._generate_async(
                prompts, n_samples, max_tokens, temperature, top_p, system
            )
        )

    # ------------------------------------------------------------------ #
    # Weight sync                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_handles_from_named(
        named_tensors: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, bytes]], list[torch.Tensor]]:
        """Pack ``(name, cuda tensor)`` pairs into pickled CUDA-IPC handles.

        Ported from ``weight_sync/trainer_sync.py:build_handles`` but driven by
        an explicit iterable (the output of :meth:`extract_hf_state`) instead of
        ``model.named_parameters()``.

        Returns ``(handles, held)`` where ``handles`` is the wire payload and
        ``held`` keeps the source tensors alive until the (blocking) RPC
        returns — the IPC handle only references storage; if the tensor is GC'd
        before the receiver maps it, the import is undefined behaviour.
        """
        handles: list[tuple[str, bytes]] = []
        held: list[torch.Tensor] = []
        for name, tensor in named_tensors:
            t = tensor.detach().contiguous()
            held.append(t)
            handles.append((name, pickle.dumps(reduce_tensor(t))))
        return handles, held

    async def _update_weights_async(
        self, named_tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        handles, held = self.build_handles_from_named(named_tensors)
        async with PieClient(self.pie_uri) as client:
            await client.authenticate(self.username)
            ok, value = await client.update_weights(handles)
        del held  # sources are safe to drop once the RPC has returned
        if not ok:
            # A driver-side failure can surface opaquely (e.g. "missing field
            # 'value'"); check the Pie server log if `value` is uninformative.
            raise RuntimeError(f"update_weights failed: {value!r}")

    def update_weights(
        self, named_tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        """Push policy weights into the live rollout engine via CUDA IPC.

        ``named_tensors`` is the HF-named, bf16, GPU-resident stream produced by
        :meth:`extract_hf_state` (or, for the Phase-1 parity test, directly from
        a plain ``model.named_parameters()``).
        """
        asyncio.run(self._update_weights_async(named_tensors))

    # ------------------------------------------------------------------ #
    # Weight extraction                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_fsdp_wrapped(model: torch.nn.Module) -> bool:
        """True if ``model`` is (or contains) an FSDP-wrapped module.

        Covers FSDP1 (``FullyShardedDataParallel``), FSDP2 (``FSDPModule``), and
        the general case where only inner submodules are wrapped (detected by the
        ``_fsdp_wrapped_module`` marker FSDP injects).
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(model, FSDP):
                return True
        except Exception:
            pass
        try:
            from torch.distributed.fsdp import FSDPModule  # fsdp2

            if isinstance(model, FSDPModule):
                return True
        except Exception:
            pass
        return any(hasattr(m, "_fsdp_wrapped_module") for m in model.modules())

    @staticmethod
    def extract_hf_state(
        model: torch.nn.Module,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Produce the GPU-resident, HF-named, bf16 parameter stream to push.

        Yields ``(hf_name, cuda_tensor)`` for every parameter such that the name
        set is byte-for-byte identical to a plain ``model.named_parameters()`` on
        the HF reference model (the ~338-name reference). A silent name mismatch
        loads nothing and leaves the engine coherent-but-stale.

        Two paths:

        * **Plain module** (the Phase-1 parity test passes a raw
          ``AutoModelForCausalLM``) — VERIFIED, below: cast to bf16, ensure CUDA.
        * **FSDP-wrapped actor** (Phase 2: ``PieActor.fsdp_module``) — the active
          ``TODO(human)``.
        """
        if PieRolloutWorker._is_fsdp_wrapped(model):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(model, writeback=False):
                for name, param in model.named_parameters():
                    clean_name = name
                    for prefix in ("_fsdp_wrapped_module.", "module."):
                        clean_name = clean_name.replace(prefix, "")
                    yield (clean_name, param.detach().to(torch.bfloat16).clone())
            return

        # Plain-model path
        for name, param in model.named_parameters():
            yield (name, param.bfloat16().cuda())

