#!/usr/bin/env bash
# Push the experiment branches to the user's forks and make them public so a fresh
# RunPod pod can clone them. Run LOCALLY (needs your GitHub auth). Idempotent.
set -euo pipefail
git -C "$HOME/git_repos/verl" push origin pie-rl
git -C "$HOME/git_repos/pie-gt" push origin features/RL
gh repo edit Giotyp/verl --visibility public --accept-visibility-change-consequences
gh repo edit Giotyp/pie  --visibility public --accept-visibility-change-consequences
echo "public: https://github.com/Giotyp/verl (pie-rl) + https://github.com/Giotyp/pie (features/RL)"
