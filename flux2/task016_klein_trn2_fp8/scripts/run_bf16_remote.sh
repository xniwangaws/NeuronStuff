#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/remote_env.sh"
unset FLUX2_FP8_MLP
unset FLUX2_FP8_ACTIVATION
unset UNSAFE_FP8FNCAST
unset XLA_HANDLE_SPECIAL_SCALAR
unset DISABLE_NUMERIC_CC_TOKEN

mkdir -p /mnt/nvme/flux2-klein/logs
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
python /mnt/nvme/flux2-klein/src/bench_klein_1k.py \
  --model /mnt/nvme/flux2-klein/weights \
  --compile-dir /mnt/nvme/flux2-klein/compiled_bf16 \
  --output-dir /mnt/nvme/flux2-klein/outputs_bf16 \
  "$@" 2>&1 | tee "/mnt/nvme/flux2-klein/logs/bf16_${run_id}.log"
