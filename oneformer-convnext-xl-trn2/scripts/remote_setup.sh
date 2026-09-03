#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p \
    agent_artifacts/data/pip_cache \
    agent_artifacts/data/hf_cache \
    agent_artifacts/results \
    agent_artifacts/traces \
    agent_artifacts/tmp

source scripts/remote_env.sh

echo "Using Neuron environment: $VIRTUAL_ENV"
python -c \
    'import torch, torch_neuronx; print("torch", torch.__version__); print("torch_neuronx", getattr(torch_neuronx, "__version__", "unknown")); print("trace", hasattr(torch_neuronx, "trace"))'

rm -rf "$PROJECT_DIR/.deps"
mkdir -p .deps
python -m pip install --target .deps --upgrade -r requirements.txt
python -m pip check

df -h "$PROJECT_DIR"
neuron-ls
python scripts/check_environment.py
python scripts/test_grid_sample.py
python scripts/smoke_wrapper.py
