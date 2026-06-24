"""Phase-1 parity test: garbage → coherent via ``PieRolloutWorker``.

Exercises verification steps 3 & 4 of the plan against a *live* Pie server:

  1. generate() BEFORE sync           -> garbage text (engine booted load_format="dummy")
  2. diff name set                    -> extract_hf_state names == model.named_parameters() (~338)
  3. update_weights(extract_hf_state) -> ok=True
  4. generate() AFTER sync            -> coherent text

MUST run on the SAME physical GPU as Pie's vLLM driver (CUDA-IPC requirement):
pin both with the same CUDA_VISIBLE_DEVICES, and run inside Pie's `vllm` venv so
the trainer's torch == the driver's torch (the reduce_tensor rebuild-tuple
layout, esp. the args[6] storage-device index, must match).

Setup (separate terminal), per qwen-rl-config.toml:
    pie serve --config /home/george/agentic-rl/PieRL/qwen-rl-config.toml
  with the `grpo` inferlet installed (build + install from ~/agentic-rl/PieRL/grpo).

Usage:
    cd ~/git_repos/verl
    CUDA_VISIBLE_DEVICES=5 /home/george/.pie/venvs/vllm/bin/python \
        -m recipe.pie_grpo.parity_test \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --pie-uri ws://127.0.0.1:8080
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from recipe.pie_grpo import PieRolloutWorker


def _show(label: str, gen: list[dict]) -> None:
    print(f"\n===== {label} =====")
    for pi, per_prompt in enumerate(gen):
        for si, sample in enumerate(per_prompt["samples"]):
            text = sample["text"].replace("\n", "\\n")
            ntok = len(sample["tokens"])
            print(f"  prompt {pi} sample {si} ({ntok} tok): {text[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--pie-uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--username", default="rl-trainer")
    ap.add_argument("--grpo-inferlet", default="grpo@0.1.0")
    ap.add_argument("--prompt", default="Write a Python function that returns the nth Fibonacci number.")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--skip-before", action="store_true",
                    help="skip the pre-sync generate (use if Pie was booted with real weights)")
    ap.add_argument("--use-actor", action="store_true",
                    help="extract weights from the FSDP-wrapped verl PieActor (Phase-2 "
                         "path — exercises extract_hf_state's FSDP branch) instead of a "
                         "plain AutoModelForCausalLM (Phase-1 path)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible — set CUDA_VISIBLE_DEVICES to Pie's GPU.")

    worker = PieRolloutWorker(args.pie_uri, args.username, args.grpo_inferlet)
    prompts = [args.prompt]

    # 1. Rollout BEFORE weight sync (garbage if Pie booted load_format="dummy").
    if not args.skip_before:
        before = worker.generate(prompts, n_samples=args.n_samples, max_tokens=args.max_tokens)
        _show("BEFORE sync (expect garbage under load_format=dummy)", before)

    # Build the weight SOURCE on the same physical GPU as the driver. Two paths:
    #   --use-actor : the FSDP-wrapped verl PieActor (Phase-2 — exercises
    #                 extract_hf_state's FSDP branch: summon + name-strip + bf16).
    #   default     : a plain AutoModelForCausalLM (Phase-1 — the verified path).
    # `ref_names` is the HF-name reference the extracted set must match exactly.
    if args.use_actor:
        from recipe.pie_grpo import PieActor

        print(f"\n[parity] building PieActor({args.model}) — FSDP, world_size=1 ...")
        actor = PieActor(model_path=args.model, rollout_n=args.n_samples)
        weight_source = actor.fsdp_module

        # Reference HF names WITHOUT a second full model resident: instantiate the
        # architecture on the `meta` device (zero real storage) just to enumerate
        # the parameter names vLLM expects.
        with torch.device("meta"):
            ref_model = AutoModelForCausalLM.from_config(actor.model_config.hf_config)
        ref_names = {n for n, _ in ref_model.named_parameters()}
    else:
        print(f"\n[parity] loading {args.model} (bf16, cuda)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        ).cuda()
        model.eval()
        weight_source = model
        ref_names = {n for n, _ in model.named_parameters()}

    # 2. Name-set parity check — the silent-failure guard. extract_hf_state must
    #    reproduce exactly the names a plain named_parameters() yields.
    got_names = {n for n, _ in PieRolloutWorker.extract_hf_state(weight_source)}
    missing = ref_names - got_names
    extra = got_names - ref_names
    print(f"[parity] reference params: {len(ref_names)} | extracted: {len(got_names)}")
    if missing or extra:
        print(f"[parity] !! NAME MISMATCH  missing={sorted(missing)[:5]}... extra={sorted(extra)[:5]}...")
        raise SystemExit("Name set differs from reference — push would silently load nothing.")
    print("[parity] ✓ name sets identical")

    # 3. Push weights into the live engine (zero-copy CUDA IPC).
    print("[parity] pushing weights via update_weights...")
    worker.update_weights(PieRolloutWorker.extract_hf_state(weight_source))
    print("[parity] ✓ update_weights ok")

    # 4. Rollout AFTER sync — should now be coherent.
    after = worker.generate(prompts, n_samples=args.n_samples, max_tokens=args.max_tokens)
    _show("AFTER sync (expect coherent code)", after)
    print("\n[parity] done — eyeball BEFORE vs AFTER for the garbage→coherent signal.")


if __name__ == "__main__":
    main()
