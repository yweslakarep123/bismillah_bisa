#!/bin/bash
# Single baseline run with early stopping (one seed).
# Full protocol (3 seeds + search + plots): bash scripts/kitchen_run_experiment.sh
#
# Usage: bash scripts/train_kitchen.sh [seed] [gpu_id] [debug]
# Example: bash scripts/train_kitchen.sh 0 0 false

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED=${1:-0}
GPU_ID=${2:-0}
DEBUG=${3:-false}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

ADDITION=${4:-baseline}
EXP_NAME="kitchen_complete-flowpolicy-${ADDITION}"
RUN_DIR="data/outputs/${EXP_NAME}_seed${SEED}"

if [ "${DEBUG}" = "true" ]; then
  WANDB_MODE=offline
else
  WANDB_MODE=online
fi

python train.py --config-name=flowpolicy_kitchen.yaml \
  hydra.run.dir="${RUN_DIR}" \
  training.debug="${DEBUG}" \
  training.seed="${SEED}" \
  training.device="cuda:0" \
  training.use_early_stopping=true \
  training.run_validation=true \
  exp_name="${EXP_NAME}" \
  logging.mode="${WANDB_MODE}" \
  checkpoint.save_ckpt=true
