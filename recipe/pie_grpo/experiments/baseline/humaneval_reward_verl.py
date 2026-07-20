"""verl-interface reward for the baseline: wraps the SAME HumanEval unit-test scorer
the Pie run uses (recipe.pie_grpo.humaneval_reward.compute_reward), so reward
semantics are identical across backends. verl calls compute_score per sample."""
from __future__ import annotations

from recipe.pie_grpo.humaneval_reward import compute_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return float(compute_reward(solution_str, ground_truth))
