#!/bin/bash
# Usage: bash scripts/eval_kitchen.sh <checkpoint_path> [gpu_id]
# For full hparam protocol (50 ep x 3 seeds, mean±std): use kitchen_eval_multiseed.sh
# Example: bash scripts/kitchen_eval_multiseed.sh path/to/best_val_loss_epoch1000.ckpt 0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH=${1:?Provide checkpoint path}
GPU_ID=${2:-0}
SEED=${3:-3}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -c "
import pathlib
import sys
import torch
import dill
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path('.').resolve()))
from train import TrainFlowPolicyWorkspace

ckpt = pathlib.Path('${CKPT_PATH}')
payload = torch.load(ckpt.open('rb'), pickle_module=dill, map_location='cpu')
cfg = payload['cfg']
cfg.training.seed = ${SEED}
ws = TrainFlowPolicyWorkspace(cfg)
ws.load_payload(payload)
ws.epoch = payload['pickles'].get('epoch', 0) if 'pickles' in payload else 0
ws.eval()
"
