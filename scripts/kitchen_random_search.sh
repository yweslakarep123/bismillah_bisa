#!/bin/bash
# Random hyperparameter search only (run baseline first via kitchen_run_experiment.sh).
# Seeds: train & eval [0, 42, 101]; early stopping enabled per trial.
# See flowpolicy_hyperparameter_finetuning.md
#
# Usage:
#   bash scripts/kitchen_random_search.sh 10 0        # 10 trials on GPU 0
#   bash scripts/kitchen_random_search.sh 1 0 5       # single trial id=5
#   bash scripts/kitchen_run_experiment.sh 10 0       # baseline + search + plots

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_TRIALS=${1:-1}
GPU_ID=${2:-0}
TRIAL_ID=${3:-}
EXTRA_ARGS=("${@:4}")

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"

CMD=(
  python scripts/kitchen_hparam_search.py
  --phase search
  --gpu "${GPU_ID}"
  --train-seeds 0 42 101
  --eval-seeds 0 42 101
)

if [[ -n "${TRIAL_ID}" && "${TRIAL_ID}" != "--"* ]]; then
  CMD+=(--trial-id "${TRIAL_ID}")
else
  CMD+=(--n-trials "${N_TRIALS}" --start-trial 0)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
