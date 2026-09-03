#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source scripts/remote_env.sh

mkdir -p agent_artifacts/results agent_artifacts/traces agent_artifacts/tmp
exec > >(tee agent_artifacts/results/remote_run_components.log) 2>&1

export TMPDIR="$PROJECT_DIR/agent_artifacts/tmp"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_LOG_LEVEL=INFO

python scripts/compile_components.py
python scripts/run_component_validation.py
