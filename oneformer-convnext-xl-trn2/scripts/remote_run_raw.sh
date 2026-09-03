#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source scripts/remote_env.sh

mkdir -p agent_artifacts/results agent_artifacts/traces agent_artifacts/tmp
exec > >(tee agent_artifacts/results/remote_run_raw.log) 2>&1

export TMPDIR="$PROJECT_DIR/agent_artifacts/tmp"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_LOG_LEVEL=INFO

python scripts/prepare_reference.py
python scripts/verify_reference.py
python scripts/test_grid_sample.py --compile --implementation raw
python scripts/analyze_model.py
python scripts/compile_model.py
python scripts/run_validation.py
