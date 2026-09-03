#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source scripts/remote_env.sh

mkdir -p agent_artifacts/results agent_artifacts/traces agent_artifacts/tmp
exec > >(tee agent_artifacts/results/remote_run_custom.log) 2>&1

export TMPDIR="$PROJECT_DIR/agent_artifacts/tmp"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_LOG_LEVEL=INFO

python scripts/test_grid_sample.py --compile --implementation custom
python scripts/analyze_model.py \
    --custom-grid-sample \
    --output agent_artifacts/results/neuron_analyze_custom.json \
    --compiler-workdir agent_artifacts/traces/analyze_custom
python scripts/compile_model.py \
    --custom-grid-sample \
    --output agent_artifacts/traces/oneformer_custom_512.pt \
    --compiler-workdir agent_artifacts/traces/oneformer_custom_compile
python scripts/run_validation.py \
    --compiled-model agent_artifacts/traces/oneformer_custom_512.pt \
    --output agent_artifacts/results/custom_validation.json
