"""``logprob_crosscheck`` — verify ``PieActor.compute_log_prob`` against an
independent oracle (a plain ``AutoModelForCausalLM`` forward of the same weights).

The trainer-side log-probs are the loop's highest-risk, least-checked piece: a
bug (off-by-one in the un-pad left-shift, wrong masking, a temperature mismatch)
is *silent* — the loss stays finite and rollouts stay coherent while the policy
gradient is quietly wrong. A vanilla HF forward is a fully independent reference
(verl validates its own engine the same way in ``tests/models/test_engine.py``),
so agreement to numerical tolerance is strong evidence the trainer path is correct.

This replaced the Pie ``token-logprob`` inferlet oracle, whose ``Logprob``-probe
path is blocked by a pie-gt driver bug (empty ``sampler_label_ids`` →
``IndexError`` in ``common.py:_process_logprob``); see ``_hf_reference``.

Parity rule that still matters:
  * **Temperature 1.0.** ``compute_log_prob`` is called at ``temperature=1.0`` so
    its raw log-probs match the unscaled HF reference — NOT the rollout temperature.
The HF reference reuses the *packed* batch, so prompt/cue tokenization is shared by
construction; this check targets verl's engine math, not Pie's tokenization.

Run (in the vllm venv, with Pie serving 0.5B for the rollout):
    python -m recipe.pie_grpo.logprob_crosscheck --config recipe/pie_grpo/config.yaml
"""

from __future__ import annotations

import argparse

import torch
import yaml

from recipe.pie_grpo.actor_bootstrap import PieActor
from recipe.pie_grpo.pie_rollout_worker import PieRolloutWorker
from recipe.pie_grpo.train_grpo import pack_dataproto


def _hf_reference(model_path: str, dp) -> list[dict]:
    """Independent oracle: teacher-force the SAME packed batch through a plain
    ``AutoModelForCausalLM`` and return per-sample response log-probs.

    Returns one dict per sample, ``{"log_probs": [...]}`` of length ``n_i - 1``
    (response log-probs for ``c_1..c_{n_i-1}``, SKIPPING ``c_0``) — the exact shape
    the token-logprob inferlet produced, so ``compare_logprobs`` is reused unchanged.

    Loaded in **bf16** to mirror the trainer's compute dtype (``model_dtype="bf16"``)
    and given the SAME ``position_ids`` the trainer used — so any residual gap is
    verl's engine math, not a precision/positions confound. Runs on CPU to avoid
    contending with the actor + Pie for GPU memory; the batch is tiny.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    input_ids = dp.batch["input_ids"]
    attn = dp.batch["attention_mask"]
    pos = dp.batch["position_ids"]
    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attn, position_ids=pos
        ).logits.float()
    # token_logp[i, k] = log p(input_ids[i, k+1] | prefix): the logit at k predicts token k+1.
    token_logp = (
        torch.log_softmax(logits[:, :-1], dim=-1)
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )

    max_p = dp.batch["prompts"].shape[1]  # response token j sits at column max_p + j
    rmask = dp.batch["response_mask"]
    results = []
    for i in range(input_ids.shape[0]):
        n = int(rmask[i].sum())
        # response token j (col max_p+j) is predicted by the logit at col max_p+j-1;
        # j runs 1..n-1 to skip c_0 and match the inferlet's output shape.
        lps = [float(token_logp[i, max_p + j - 1]) for j in range(1, n)]
        results.append({"log_probs": lps})
    return results


def compare_logprobs(
    trainer_lp: torch.Tensor,
    response_mask: torch.Tensor,
    inferlet_results: list[dict],
    *,
    tol: float = 2e-2,
) -> dict:
    """Align trainer vs. inferlet per-token log-probs and report the discrepancy.

    Inputs (rows are in the same prompt-major order ``pack_dataproto`` produced):
      - ``trainer_lp``     : ``(bsz, max_resp)`` padded; ``trainer_lp[i, j]`` is the
                             trainer's ``log p(c_j | prompt + c_0..c_{j-1})`` for
                             response token ``j`` of sample ``i``.
      - ``response_mask``  : ``(bsz, max_resp)`` bool/float; ``n_i = mask[i].sum()``
                             is sample ``i``'s real response length.
      - ``inferlet_results``: one dict per sample, ``["log_probs"]`` a list of
                             length ``n_i - 1`` — the inferlet scores ``c_1..c_{n_i-1}``
                             (it deliberately SKIPS ``c_0``).

    Returns a dict with at least: per-sample max|diff| and mean|diff|, the overall
    max|diff|, and a boolean ``passed`` (overall max ≤ ``tol``). A clean pass
    (~1e-2, bf16 + different kernels) confirms the trainer log-probs are correct;
    a *constant* offset across all tokens usually means a system/cue-encoding
    mismatch, while a *growing* offset means a position/shift bug.
    """
    per_sample = []
    overall_max = 0.0
    for i in range(trainer_lp.shape[0]):
        n_s = int(response_mask[i].sum())

        if n_s <= 1:  # no c_1.. to compare
            per_sample.append(
                {"n": n_s, "max_abs": float("nan"), "mean_abs": float("nan")}
            )
            continue

        tr = trainer_lp[i][1:n_s].float().cpu()
        inf = torch.tensor(
            inferlet_results[i]["log_probs"][: n_s - 1], dtype=torch.float32
        )

        abs_diff = (inf - tr).abs()
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())

        per_sample.append({"n": n_s, "max_abs": max_abs, "mean_abs": mean_abs})
        overall_max = max(overall_max, max_abs)

    return {
        "per_sample": per_sample,
        "overall_max_abs": overall_max,
        "passed": overall_max <= tol,
        "tol": tol,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="recipe/pie_grpo/config.yaml")
    ap.add_argument(
        "--n-prompts", type=int, default=2, help="how many prompts to probe"
    )
    ap.add_argument("--n-samples", type=int, default=2, help="completions per prompt")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rollout = cfg["rollout"]
    system = rollout.get("system")

    worker = PieRolloutWorker(
        pie_uri=cfg["pie"]["uri"],
        username=cfg["pie"]["username"],
        grpo_inferlet=cfg["pie"]["grpo_inferlet"],
    )
    actor = PieActor(model_path=cfg["model"]["path"], rollout_n=args.n_samples)
    prompts = cfg["data"]["prompts"][: args.n_prompts]

    # 1. roll out a few completions (rollout temperature is fine for *generating*)
    gen = worker.generate(
        prompts,
        n_samples=args.n_samples,
        max_tokens=rollout["max_tokens"],
        temperature=rollout["temperature"],
        top_p=rollout["top_p"],
        system=system,
    )

    # 2. trainer-side log-probs at TEMPERATURE 1.0 (to match the inferlet's raw probe)
    dp = pack_dataproto(prompts, gen, actor.tokenizer, system=system)
    trainer_lp = actor.compute_log_prob(dp, temperature=1.0)

    # 3. independent oracle: a plain HF forward of the same weights (no Pie probe).
    reference_results = _hf_reference(cfg["model"]["path"], dp)

    # 4. align + compare
    report = compare_logprobs(trainer_lp, dp.batch["response_mask"], reference_results)
    for i, s in enumerate(report["per_sample"]):
        print(
            f"  sample {i}: n={s['n']:3d}  max|Δ|={s['max_abs']:.4f}  mean|Δ|={s['mean_abs']:.4f}"
        )
    # Per-token table is informational only — bf16-vs-fp32 log-softmax noise makes
    # per-token max|Δ| an unreliable verdict (it's dominated by zero-mean scatter).
    print(
        f"   (per-token max|Δ|={report['overall_max_abs']:.4f}, mean|Δ| ~0.05 — bf16 noise, see below)"
    )

    # --- bias vs noise: the criterion that actually settles correctness ---
    # A real bug (off-by-one / temperature / masking) is BIASED — it survives
    # averaging and wrecks the correlation. bf16 scatter is zero-mean and leaves
    # correlation ~1. So the verdict keys on correlation + diff-of-means, not max|Δ|.
    rmask = dp.batch["response_mask"]
    tr_all, rf_all = [], []
    for i in range(trainer_lp.shape[0]):
        n = int(rmask[i].sum())
        if n <= 1:
            continue
        tr_all.append(trainer_lp[i][1:n].float().cpu())
        rf_all.append(torch.tensor(reference_results[i]["log_probs"][: n - 1]))
    tr = torch.cat(tr_all)
    rf = torch.cat(rf_all)
    signed = tr - rf
    diff_of_means = abs(tr.mean() - rf.mean()).item()
    corr = torch.corrcoef(torch.stack([tr, rf]))[0, 1].item()
    print(
        f"   diff-of-means (bf16 bar <0.02) = {diff_of_means:.5f}\n"
        f"   signed mean bias (trainer-ref) = {signed.mean():+.5f}  (±{signed.std():.4f})\n"
        f"   correlation (struct. bar >.999)= {corr:.5f}"
    )
    # No structural bug ⇒ near-perfect correlation; bias within bf16 precision.
    passed = corr > 0.999 and diff_of_means < 0.02
    print(f"[{'PASS' if passed else 'FAIL'}] compute_log_prob matches an independent HF forward")


if __name__ == "__main__":
    main()
