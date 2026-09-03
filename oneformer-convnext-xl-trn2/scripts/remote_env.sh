#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for candidate in \
    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference \
    /opt/aws_neuronx_venv_pytorch_inference \
    /opt/aws_neuronx_venv_pytorch_2_9 \
    /opt/aws_neuronx_venv_pytorch; do
    if [[ -f "$candidate/bin/activate" ]]; then
        source "$candidate/bin/activate"
        break
    fi
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Unable to locate the Neuron PyTorch virtual environment." >&2
    exit 3
fi

export PYTHONPATH="$PROJECT_DIR/.deps:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/opt/aws/neuron/bin:$PATH"
export PIP_CACHE_DIR="$PROJECT_DIR/agent_artifacts/data/pip_cache"
export HF_HOME="$PROJECT_DIR/agent_artifacts/data/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES=0-3
