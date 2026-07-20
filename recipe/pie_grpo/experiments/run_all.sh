#!/usr/bin/env bash
# Orchestrator: run_all.sh [pie|baseline|compare|all]
# Assumes setup_runpod.sh has run. The `pie` stage assumes `pie serve` is already up
# on 127.0.0.1:8080 (start it in the background first).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
STAGE="${1:-all}"
PIE_OUT="runs/pie/metrics.json"
BASE_OUT="runs/verl_baseline/metrics.json"
mkdir -p runs/pie runs/verl_baseline

run_pie() {
  echo "== register grpo inferlet on the live Pie server =="
  python -m recipe.pie_grpo.experiments.install_grpo \
    --uri ws://127.0.0.1:8080 --username rl-trainer
  echo "== Pie GRPO run (assumes 'pie serve' is up) =="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PIE_EXP_GPU="${PIE_EXP_GPU:-A40}" \
    torchrun --nproc_per_node=2 -m recipe.pie_grpo.train_grpo \
    --config recipe/pie_grpo/experiments/configs/pie_config.yaml --metrics-out "$PIE_OUT"
}
run_baseline() { bash recipe/pie_grpo/experiments/baseline/run_baseline.sh "$BASE_OUT"; }
run_compare()  { python -m recipe.pie_grpo.experiments.compare "$PIE_OUT" "$BASE_OUT"; }

case "$STAGE" in
  pie) run_pie ;;
  baseline) run_baseline ;;
  compare) run_compare ;;
  all) run_pie; run_baseline; run_compare ;;
  *) echo "usage: $0 [pie|baseline|compare|all]"; exit 1 ;;
esac
