"""Topology resolution for the Pie⇄verl GRPO trainer.

Stage B — makes the 2-GPU / same-GPU-IPC assumptions **configurable** so an
N-GPU layout (and, later, an NVLink box where cross-GPU disaggregation becomes
possible) is a config change rather than a code rewrite. The three things that
were hardcoded become one seam each:

  * ``world_size``      — was ``int(os.environ["WORLD_SIZE"])`` scattered around
  * ``fsdp_size``       — was ``int(os.environ["WORLD_SIZE"])`` in the engine cfg
  * weight-sync target  — was the literal ``device_idx == rank``

**Defaults reproduce the current colocated DP=2 path exactly**: ``world_size`` and
``fsdp_size`` follow the launcher (torchrun ``WORLD_SIZE``), the device map is the
identity (``device_idx == rank`` → same-GPU zero-copy CUDA-IPC), and cross-GPU IPC
is disabled. Nothing here changes runtime behaviour on the 2× 4090 box; it only
un-freezes the knobs.

The config lives under a ``topology:`` block in ``config.yaml`` (all optional):

    topology:
      world_size: auto          # auto → launcher WORLD_SIZE; or an int
      fsdp_size: auto           # auto → world_size (pure ZeRO-3); int < ws → hybrid mesh
      weight_sync:
        device_map: identity    # identity → device_idx == rank (same-GPU IPC)
        allow_cross_gpu_ipc: false  # guard: cross-GPU IPC needs P2P/NVLink
"""

from __future__ import annotations

import os


def _auto_or_int(val, fallback: int) -> int:
    """``None`` / ``"auto"`` → ``fallback``; anything else → ``int(val)``."""
    if val is None or val == "auto":
        return int(fallback)
    return int(val)


def _env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def resolve_world_size(topo: dict | None) -> int:
    """Number of trainer ranks. ``auto`` (default) reads the launcher's
    ``WORLD_SIZE`` — the single source of truth under torchrun."""
    return _auto_or_int((topo or {}).get("world_size"), _env_world_size())


def resolve_fsdp_size(topo: dict | None, world_size: int) -> int:
    """FSDP shard degree. ``auto`` (default) == ``world_size`` → pure ZeRO-3 across
    every rank (today's behaviour). An int ``< world_size`` selects a hybrid
    (ddp × fsdp) device mesh, which verl's ``create_device_mesh`` already supports
    (``fsdp/utils.py``) — the knob for scaling wider without full-shard comms cost."""
    fsdp_size = _auto_or_int((topo or {}).get("fsdp_size"), world_size)
    if world_size % fsdp_size != 0:
        raise ValueError(
            f"fsdp_size={fsdp_size} must divide world_size={world_size}"
        )
    return fsdp_size


def device_for_rank(topo: dict | None, local_rank: int) -> int:
    """CUDA device index (into ``CUDA_VISIBLE_DEVICES``) this rank binds to.

    Identity today: rank *i* → visible device *i*; the ``CUDA_VISIBLE_DEVICES``
    ordering handles the physical map (e.g. ``5,3`` → visible 0,1). Kept as a seam
    so a future non-identity placement is a one-liner here, not a set_device edit."""
    return local_rank


def device_idx_for_rank(topo: dict | None, rank: int) -> int:
    """Which Pie DP replica this rank pushes its weights into.

    ``identity`` (default) → ``device_idx == rank``: the push lands on the replica
    colocated on this rank's own GPU, i.e. same-GPU zero-copy CUDA-IPC — the only
    path possible on a no-P2P box. A non-identity map would push cross-GPU, which
    needs P2P/NVLink; it is guarded behind ``allow_cross_gpu_ipc`` so it can never
    silently run on hardware that cannot map the IPC handle across GPUs."""
    ws = (topo or {}).get("weight_sync") or {}
    policy = ws.get("device_map", "identity")
    allow_cross = bool(ws.get("allow_cross_gpu_ipc", False))

    if policy == "identity":
        idx = rank
    else:
        raise ValueError(f"unknown weight_sync.device_map policy: {policy!r}")

    if idx != rank and not allow_cross:
        raise RuntimeError(
            f"weight_sync would push rank {rank} → device {idx} (cross-GPU), but "
            "allow_cross_gpu_ipc is false. Cross-GPU CUDA-IPC needs P2P/NVLink; "
            "the current box reports CNS (no P2P) for every GPU pair."
        )
    return idx
