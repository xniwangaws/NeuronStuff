#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"
export FLUX2_FP8_MLP=1
export FLUX2_FP8_SCOPE="${FLUX2_FP8_SCOPE:-mlp}"
export FLUX2_FP8_ACTIVATION="${FLUX2_FP8_ACTIVATION:-none}"
export UNSAFE_FP8FNCAST=1
export XLA_HANDLE_SPECIAL_SCALAR=1
export DISABLE_NUMERIC_CC_TOKEN=1

scope="${FLUX2_FP8_SCOPE}"
mode="${FLUX2_FP8_ACTIVATION}"
if [[ "${scope}" == "mlp" ]]; then
  artifact_tag="${mode}"
else
  artifact_tag="${scope}_${mode}"
fi

mkdir -p /mnt/nvme/flux2-klein/logs
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
python /mnt/nvme/flux2-klein/src/bench_klein_1k.py \
  --model /mnt/nvme/flux2-klein/weights \
  --compile-dir "/mnt/nvme/flux2-klein/compiled_fp8_${artifact_tag}" \
  --output-dir "/mnt/nvme/flux2-klein/outputs_fp8_${artifact_tag}" \
  "$@" 2>&1 | tee \
  "/mnt/nvme/flux2-klein/logs/fp8_${artifact_tag}_${run_id}.log"
