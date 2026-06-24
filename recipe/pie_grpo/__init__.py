"""verl ⇄ Pie GRPO recipe.

Phase 1 ships ``PieRolloutWorker``: the adapter that drives a live Pie/vLLM
rollout engine directly — generation via the ``grpo`` inferlet over a WebSocket,
and policy weight-sync via CUDA-IPC handles — so a standalone verl-based trainer
can use Pie as its rollout backend without going through verl's HTTP rollout
abstraction. See ``.claude/plans/Impl-PieRolloutWorker.md``.
"""

from .actor_bootstrap import PieActor
from .pie_rollout_worker import PieRolloutWorker

__all__ = ["PieActor", "PieRolloutWorker"]
