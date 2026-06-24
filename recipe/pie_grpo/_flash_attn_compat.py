"""Flash-attn-free ``unpad_input`` for the colocated Pie ``vllm`` venv.

verl's ``left_right_2_no_padding`` (padded → nested-tensor conversion) routes
through ``verl.utils.attention_utils.unpad_input``, which on CUDA hard-imports
``flash_attn.bert_padding`` — but the Pie ``vllm`` venv deliberately has **no
flash_attn** (the actor already runs ``attn_implementation="eager"``). The
dependency here is purely the padded↔nested *data reshape*, not any attention
kernel, so we satisfy it with ``transformers``' pure-torch ``_unpad_input`` /
``_pad_input`` (both present in transformers ≥4.x).

We patch verl's single provider, ``_get_attention_functions``, rather than
injecting a fake ``flash_attn`` package into ``sys.modules`` — the latter breaks
transformers' own ``importlib.util.find_spec("flash_attn")`` probe
(``flash_attn.__spec__ is None`` → ValueError) and could fool transformers into
trying to call kernels that don't exist. Patching verl leaves transformers'
detection (correctly: no flash) untouched.

Import this module for its side effect *before* the first weight/log-prob pass.
It is a no-op when flash_attn is actually installed.
"""

from __future__ import annotations

import verl.utils.attention_utils as _au


def install() -> bool:
    """Route verl's unpad/pad helpers through transformers when flash_attn is
    missing. Returns ``True`` if the shim was installed, ``False`` if real
    flash_attn is present (left as-is). Idempotent."""
    try:
        import flash_attn  # noqa: F401

        return False
    except ModuleNotFoundError:
        pass

    from einops import rearrange
    from transformers.modeling_flash_attention_utils import _pad_input, _unpad_input

    def index_first_axis(x, indices):
        # flash_attn's index_first_axis selects rows of a (b*s, ...) tensor; the
        # plain advanced-index is equivalent for our (no SP, no MoE) path.
        return x[indices]

    def _provider():
        _au._index_first_axis = index_first_axis
        _au._pad_input = _pad_input
        _au._rearrange = rearrange
        _au._unpad_input = _unpad_input
        return index_first_axis, _pad_input, rearrange, _unpad_input

    _au._get_attention_functions = _provider
    return True


_INSTALLED = install()
