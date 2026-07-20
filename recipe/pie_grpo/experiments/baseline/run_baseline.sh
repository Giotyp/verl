#!/usr/bin/env bash
# Launch the verl-native GRPO baseline on 2 GPUs and emit a common metrics.json.
#
# verl owns its training loop; to fill metrics.json we collect its per-step metrics
# dict + step wall-clock + validation (heval/mbpp) and hand them to
# verl_metrics_bridge.build_from_verl. Two capture paths (decide on the first pod;
# both feed the same bridge):
#   (a) a small custom verl logger that appends each step's dict + perf_counter delta
#       and calls build_from_verl on run end (cleanest);
#   (b) parse console.log (this script tees it) for actor/pg_loss, critic/rewards/mean,
#       response_length/mean, val-core/.../pass@1 + step times, then call the bridge.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-runs/verl_baseline/metrics.json}"
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VERL_METRICS_OUT="$OUT"

python -m verl.trainer.main_ppo \
  --config-path "$(pwd)/recipe/pie_grpo/experiments/baseline" \
  --config-name grpo_baseline_config \
  2>&1 | tee "$(dirname "$OUT")/console.log"

echo "verl baseline done. Metrics collection: see the two capture paths in this script's"
echo "header; both call recipe/pie_grpo/experiments/baseline/verl_metrics_bridge.build_from_verl"
echo "to write $OUT (finalize the capture path on the first pod against verl's live output)."
