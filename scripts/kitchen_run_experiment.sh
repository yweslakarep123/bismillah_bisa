#!/bin/bash
# Full Franka Kitchen experiment:
#   1) Baseline training (early stopping) × seeds [0, 42, 101]
#   2) Random hyperparameter search (early stopping) × same seeds
#   3) Plot success rate vs latency, trade-off vs val loss, subtask SR, etc.
#
# Usage:
#   bash scripts/kitchen_run_experiment.sh [n_search_trials] [gpu_id]
#   bash scripts/kitchen_run_experiment.sh 10 0
#   bash scripts/kitchen_run_experiment.sh 0 0   # baseline + plots only (no search)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_TRIALS=${1:-10}
GPU_ID=${2:-0}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"

PHASE="all"
if [[ "${N_TRIALS}" -eq 0 ]]; then
  PHASE="baseline"
fi

ARGS=(
  python scripts/kitchen_hparam_search.py
  --phase "${PHASE}"
  --gpu "${GPU_ID}"
  --train-seeds 0 42 101
  --eval-seeds 0 42 101
  --eval-episodes 50
  --plot-after
)

if [[ "${N_TRIALS}" -gt 0 ]]; then
  ARGS+=(--n-trials "${N_TRIALS}")
fi

echo "Running: ${ARGS[*]}"
"${ARGS[@]}"

echo ""
echo "Plots saved under FlowPolicy/data/hparam_search/plots/"
echo "Results: baseline_results.jsonl, results.jsonl, best_config.json"
