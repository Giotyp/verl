"""HumanEval unit-test reward — vendored from RL-Heval (``reward_fn.py``).

Binary pass/fail: extract the model's code, splice it with the problem's stub
(only if the function isn't already defined — instruct models redeclare it,
completion models emit only the body), append the test harness, and run it in a
**sandboxed subprocess**. Returns 1.0 iff the test process exits 0.

Adapted for the Pie ``vllm`` venv: uses ``sys.executable`` (the venv has no bare
``python`` on PATH) instead of the literal ``"python"``.

SECURITY: this executes model-generated code every step. The subprocess is capped
(``RLIMIT_AS`` 256 MB, ``RLIMIT_CPU`` 10 s, 10 s wall timeout) — the standard
HumanEval harness — but it is still arbitrary code; run only on trusted hardware.
"""

from __future__ import annotations

import json
import re
import resource
import subprocess
import sys


def _extract_code(text: str) -> str:
    """Pull the Python body out of a completion: strip stray special-token markers
    (``<|im_end|>`` etc.), then take the first fenced ```python block (or any fenced
    block); fall back to the raw text."""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def compute_reward(solution_str: str, ground_truth) -> float:
    """1.0 if ``solution_str`` passes the HumanEval test, else 0.0.

    ``ground_truth`` is either the dict or a JSON string with keys
    ``{prompt, test, entry_point}`` (the parquet stores it JSON-encoded).
    """
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        code = _extract_code(solution_str)
        # Prepend the stub only when the function isn't already defined.
        if f"def {gt['entry_point']}" in code:
            body = code
        else:
            body = gt["prompt"] + code
        full_code = body + "\n\n" + gt["test"] + f"\ncheck({gt['entry_point']})\n"

        mem_bytes = 256 * 1024 * 1024  # 256 MB address space

        def _preexec():
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))  # 10 CPU-seconds

        result = subprocess.run(
            [sys.executable, "-c", full_code],
            timeout=10,
            capture_output=True,
            preexec_fn=_preexec,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0
