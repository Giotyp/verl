#!/usr/bin/env bash
# Thin-layer bootstrap on a stock RunPod A40 PyTorch/CUDA base image.
# Builds its OWN venv (base torch 2.1-2.8 is NOT used) with the pinned stack, so the
# CUDA-IPC weight sync sees a consistent torch across trainer + Pie vLLM driver.
set -euo pipefail

VERL_FORK="${VERL_FORK:-https://github.com/Giotyp/verl.git}"
PIE_FORK="${PIE_FORK:-https://github.com/Giotyp/pie.git}"
WORK="${WORK:-$HOME/agentic-rl}"; mkdir -p "$WORK"
VENV="$HOME/.pie/venvs/vllm"

echo "== topology (interpret the speedup against this) =="; nvidia-smi topo -m || true

echo "== system deps =="
apt-get update -y && apt-get install -y git build-essential curl
command -v cargo >/dev/null || { curl https://sh.rustup.rs -sSf | sh -s -- -y; }
. "$HOME/.cargo/env" 2>/dev/null || true

echo "== clone forks =="
[ -d "$WORK/pie-gt" ] || git clone --branch features/RL "$PIE_FORK" "$WORK/pie-gt"
[ -d "$WORK/verl" ]   || git clone --branch pie-rl "$VERL_FORK" "$WORK/verl"

echo "== build pie (lean, no CUDA toolkit) =="
(cd "$WORK/pie-gt" && cargo install --path server --force \
   --no-default-features --features driver-portable,driver-dummy)

echo "== fresh venv + pinned stack (cu128 for A40) =="
python3 -m venv "$VENV"; . "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.21.0
pip install -r "$WORK/verl/recipe/pie_grpo/experiments/requirements.lock"
pip install -e "$WORK/pie-gt/driver/vllm" -e "$WORK/pie-gt/client/python" -e "$WORK/verl"

echo "== version guard =="
python - <<'PY'
import torch, vllm
assert torch.__version__.startswith("2.11.0"), f"torch {torch.__version__} != 2.11.0"
assert not torch.__version__.endswith("+cu129"), "cu129 wheel won't run on A40 (cu128 max)"
assert vllm.__version__.startswith("0.21.0"), f"vllm {vllm.__version__} != 0.21.0"
print("version guard OK:", torch.__version__, vllm.__version__)
PY

echo "== template Pie server config with the venv path =="
CFG="$WORK/verl/recipe/pie_grpo/experiments/configs"
sed "s#PIE_VENV_PATH#$VENV#" "$CFG/qwen-rl-config.toml" > "$WORK/qwen-rl-config.toml"

# The grpo inferlet is VENDORED (prebuilt wasm under experiments/inferlets/grpo/) and
# installed at run time by run_all.sh via install_grpo.py — no wasm build toolchain here.

cat <<EOF

== setup complete ==
1) Serve Pie:   pie serve --config $WORK/qwen-rl-config.toml --no-auth   # (background)
2) Run all:     cd $WORK/verl && CUDA_VISIBLE_DEVICES=0,1 PIE_EXP_GPU=A40 \\
                  bash recipe/pie_grpo/experiments/run_all.sh all
   (run_all installs the grpo inferlet, runs the Pie experiment + verl baseline, compares)
EOF
