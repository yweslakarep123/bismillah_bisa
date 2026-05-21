#!/bin/bash
# Generate plots from baseline + random-search JSONL results.
# Usage: bash scripts/kitchen_plot_results.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

cd "${REPO_ROOT}/FlowPolicy"
python scripts/kitchen_plot_results.py "$@"

echo "Plots: ${REPO_ROOT}/FlowPolicy/data/hparam_search/plots/"
