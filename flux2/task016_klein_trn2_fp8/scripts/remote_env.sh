#!/usr/bin/env bash

set -euo pipefail

for candidate in \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference \
  /opt/aws_neuronx_venv_pytorch_2_9 \
  /opt/aws_neuronx_venv_pytorch; do
  if [[ -f "${candidate}/bin/activate" ]]; then
    source "${candidate}/bin/activate"
    break
  fi
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Unable to locate the Neuron PyTorch virtual environment." >&2
  exit 3
fi

export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_VISIBLE_CORES=0-3
export NEURON_RT_EXEC_TIMEOUT=1800
export NEURON_RT_INSPECT_ENABLE=0
export NEURON_COMPILED_ARTIFACTS=/mnt/nvme/flux2-klein/neuron-cache
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/nvme/flux2-klein/hf-cache
export TRANSFORMERS_CACHE=/mnt/nvme/flux2-klein/hf-cache
