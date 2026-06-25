"""MBPP unit-test reward — the HumanEval reward's sibling for the transfer eval.

Why MBPP: the policy trains on HumanEval, so evaluating on MBPP (a *disjoint*
benchmark, ~257 held-out problems) measures genuine generalization, not memorized
fit — and its larger size drops the eval noise floor well below the 33-problem
HumanEval set's ±3%.

Mechanism: the SAME sandboxed-subprocess runner as ``humaneval_reward`` (reused via
``_run_sandboxed`` — no duplicated resource-limit logic). MBPP differs only in how a
problem is specified: it gives a list of ``assert`` statements (``test_list``) plus
any ``test_imports``, and the function name lives inside the asserts (there is no
separate stub to splice, unlike HumanEval). We assemble and run:

    <test_imports>
    <model code>
    <assert test_list>

1.0 iff the process exits 0.
"""

from __future__ import annotations

import json

from recipe.pie_grpo.humaneval_reward import _extract_code, _run_sandboxed


def compute_mbpp_reward(solution_str: str, ground_truth) -> float:
    """1.0 if ``solution_str`` passes the MBPP asserts, else 0.0.

    ``ground_truth`` is the dict or a JSON string with keys
    ``{test_imports, test_list}`` (the parquet stores it JSON-encoded).
    """
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        code = _extract_code(solution_str)
        parts = list(gt.get("test_imports", [])) + [code] + list(gt["test_list"])
        full_code = "\n".join(parts) + "\n"
    except Exception:
        return 0.0
    return _run_sandboxed(full_code)
