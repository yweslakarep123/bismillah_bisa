#!/bin/bash
# Eval checkpoint: 50 episodes x 3 seeds (§5 flowpolicy_hyperparameter_finetuning.md).
# Usage:
#   bash scripts/kitchen_eval_multiseed.sh <checkpoint.ckpt> [gpu_id]
#   bash scripts/kitchen_eval_multiseed.sh <run_output_dir> [gpu_id]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=${1:?Provide checkpoint .ckpt or run output directory}
GPU_ID=${2:-0}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"

CKPT=""
RUN_DIR=""
if [[ "${TARGET}" == *.ckpt ]]; then
  CKPT="${TARGET}"
  RUN_DIR="$(dirname "$(dirname "${CKPT}")")"
else
  RUN_DIR="${TARGET}"
fi

ARGS=(
  python scripts/kitchen_hparam_search.py
  --trial-id 0
  --skip-train
  --eval-episodes 50
  --eval-seeds 0 42 101
  --gpu "${GPU_ID}"
  --wandb-mode offline
  --results-csv data/hparam_search/eval_only_results.csv
)

if [[ -n "${CKPT}" ]]; then
  ARGS+=(--ckpt-path "${CKPT}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
fi

echo "Running: ${ARGS[*]}"
"${ARGS[@]}"
