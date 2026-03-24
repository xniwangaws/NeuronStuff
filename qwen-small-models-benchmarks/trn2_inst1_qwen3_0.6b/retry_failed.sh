#!/bin/bash
#
# Retry failed tasks that are NOT NKI limits
# Runs sequentially to avoid device contention
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=8080

RETRY_TASKS=(
  "qwen3_0.6b_64k_tp1_lnc1_bs1"
  "qwen3_0.6b_8k_tp2_lnc2_bs16"
  "qwen3_0.6b_8k_tp2_lnc2_bs32"
  "qwen3_0.6b_32k_tp2_lnc2_bs2"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_all() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'neuronx-cc' 2>/dev/null || true
  pkill -9 -f 'walrus_driver' 2>/dev/null || true
  sleep 3
  lsof /dev/neuron* 2>/dev/null | grep -v neuron-mo | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
  sleep 2
}

# Wait for any running waves to finish
log ""
log "========== Retry Failed Tasks =========="
log "  Waiting for running waves to finish..."
while ps aux | grep -E 'wave2_and_64k|wave_tp2' | grep -v grep > /dev/null 2>&1; do
  sleep 30
done
log "  All waves finished. Cleaning up..."

kill_all

# Clear failed neff cache
rm -rf /var/tmp/neuron-compile-cache
log "  Compile cache cleared"

# Find a free device
find_free_device() {
  local used=$(lsof /dev/neuron* 2>/dev/null | grep -v neuron-mo | awk 'NR>1{print $9}' | grep -oP '\d+' | sort -un)
  for d in $(seq 0 15); do
    local in_use=0
    for u in $used; do [[ $d -eq $u ]] && { in_use=1; break; }; done
    if [[ $in_use -eq 0 ]]; then echo $d; return; fi
  done
  echo ""
}

for dir in "${RETRY_TASKS[@]}"; do
  if is_done "$dir"; then
    log "  SKIP (done): $dir"
    continue
  fi

  log ""
  log "  Retrying: $dir"
  task_dir="$BASEDIR/$dir"

  # Determine cores needed
  cores=$(grep 'NEURON_RT_NUM_CORES' "$task_dir/serve.sh" | grep -o '[0-9]*')

  # Find free device
  dev=$(find_free_device)
  if [[ -z "$dev" ]]; then
    log "  ERROR: no free device"
    continue
  fi

  core_start=$((dev * 4))
  if [[ $cores -le 2 ]]; then
    core_end=$((core_start + cores - 1))
  else
    core_end=$((core_start + 3))
  fi
  visible_cores="${core_start}-${core_end}"

  log "  Device $dev, cores=$visible_cores, port=$BASE_PORT"

  # Reset port
  sed -i "s/--port=[0-9]*/--port=${BASE_PORT}/" "$task_dir/serve.sh"
  sed -i "s/^PORT=.*/PORT=${BASE_PORT}/" "$task_dir/bench.sh"

  # Start server
  cd "$task_dir"
  NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
  pid=$!
  cd "$BASEDIR"

  # Wait for ready
  elapsed=0
  ready=0
  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    if ! kill -0 $pid 2>/dev/null; then
      log "  FAILED_SERVE: $dir (process died)"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      break
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${BASE_PORT}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
      ready=1
      log "  Server ready (${elapsed}s)"
      break
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    (( elapsed % 120 == 0 )) && log "  Compiling... ${elapsed}s"
  done

  if [[ $ready -eq 1 ]]; then
    log "  Running bench..."
    cd "$task_dir"; bash bench.sh > bench.log 2>&1; rc=$?; cd "$BASEDIR"
    if [[ $rc -eq 0 ]]; then
      log "  DONE: $dir"
      echo "DONE $dir" >> "$PROGRESS_FILE"
      grep -E 'throughput|TTFT|ITL|TPOT' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
    else
      log "  FAILED_BENCH: $dir"
      echo "FAILED_BENCH $dir" >> "$PROGRESS_FILE"
    fi
  fi

  # Cleanup
  kill -9 $pid 2>/dev/null || true
  kill_all
done

log ""
log "  Retry complete"
