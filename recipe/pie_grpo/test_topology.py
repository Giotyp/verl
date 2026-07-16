"""Unit tests for the Stage B topology resolvers (`topology.py`).

CPU-only, no GPUs / no verl imports — loads `topology.py` directly so it runs
anywhere. Run: `python recipe/pie_grpo/test_topology.py` (or under pytest).

Covers the two things Stage B must guarantee: (1) defaults reproduce the current
launcher-driven DP=2 behaviour, and (2) the new knobs (fsdp_size, device map) are
validated rather than silently mis-applied.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

_spec = importlib.util.spec_from_file_location(
    "pie_topology", pathlib.Path(__file__).with_name("topology.py")
)
topo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(topo)


def test_defaults_reproduce_dp2():
    os.environ["WORLD_SIZE"] = "2"  # simulate torchrun --nproc_per_node=2
    cfg = {
        "world_size": "auto",
        "fsdp_size": "auto",
        "weight_sync": {"device_map": "identity", "allow_cross_gpu_ipc": False},
    }
    assert topo.resolve_world_size(cfg) == 2           # auto -> launcher WORLD_SIZE
    assert topo.resolve_fsdp_size(cfg, 2) == 2         # auto -> world_size (pure ZeRO-3)
    assert [topo.device_idx_for_rank(cfg, r) for r in (0, 1)] == [0, 1]  # identity
    assert topo.device_for_rank(cfg, 1) == 1


def test_no_topology_block_is_backward_compatible():
    os.environ["WORLD_SIZE"] = "2"
    assert topo.resolve_world_size(None) == 2
    assert topo.resolve_fsdp_size(None, 2) == 2
    assert topo.device_idx_for_rank(None, 1) == 1


def test_fsdp_size_hybrid_and_divisibility():
    assert topo.resolve_fsdp_size({"fsdp_size": 2}, 4) == 2   # 4-GPU hybrid ddp x fsdp
    try:
        topo.resolve_fsdp_size({"fsdp_size": 3}, 4)           # 3 does not divide 4
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-divisible fsdp_size")


def test_explicit_world_size_override():
    os.environ["WORLD_SIZE"] = "2"
    assert topo.resolve_world_size({"world_size": 8}) == 8    # explicit int wins over env


def test_resolve_tp_size():
    assert topo.resolve_tp_size(None) == 1                       # default = full-model path
    assert topo.resolve_tp_size({}) == 1
    assert topo.resolve_tp_size({"tensor_parallel_size": 2}) == 2


def test_unknown_device_map_policy_rejected():
    try:
        topo.device_idx_for_rank({"weight_sync": {"device_map": "swap"}}, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown device_map policy")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all topology tests passed")
