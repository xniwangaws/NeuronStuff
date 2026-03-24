#!/bin/bash
#
# Retry 64K tasks with extended timeout (3600s = 1 hour)
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600  # 1 hour for 64K
BASE_PORT=8080

RETRY_TASKS=(
  "qwen3_0.6b_64k_tp1_lnc1_bs1"
  "qwen3_0.6b_64k_tp1_lnc2_bs1"
  "qwen3_0.6b_64k_tp2_lnc2_bs1"
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

# Wait for retry_failed.sh to finish
log ""
log "========== Retry 64K Tasks (timeout=3600s) =========="
while ps aux | grep 'retry_failed' | grep -v grep > /dev/null 2>&1; do
  sleep 30
done
log "  Previous retry finished. Cleaning up..."
kill_all
rm -rf /var/tmp/neuron-compile-cache
log "  Cache cleared"

for dir in "${RETRY_TASKS[@]}"; do
  if is_done "$dir"; then
    log "  SKIP (done): $dir"
    continue
  fi

  log ""
  log "  Retrying: $dir (timeout=${TIMEOUT_SERVER}s)"
  task_dir="$BASEDIR/$dir"
  cores=$(grep 'NEURON_RT_NUM_CORES' "$task_dir/serve.sh" | grep -o '[0-9]*')

  # Use device 0
  if [[ $cores -le 2 ]]; then
    visible_cores="0-$((cores - 1))"
  else
    visible_cores="0-3"
  fi

  log "  Cores=$visible_cores, port=$BASE_PORT"
  sed -i "s/--port=[0-9]*/--port=${BASE_PORT}/" "$task_dir/serve.sh"
  sed -i "s/^PORT=.*/PORT=${BASE_PORT}/" "$task_dir/bench.sh"

  cd "$task_dir"
  NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
  pid=$!
  cd "$BASEDIR"

  elapsed=0
  ready=0
  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    if ! kill -0 $pid 2>/dev/null; then
      log "  FAILED_SERVE: $dir"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      break
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${BASE_PORT}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
      ready=1; log "  Server ready (${elapsed}s)"; break
    fi
    sleep 10; elapsed=$((elapsed + 10))
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
  elif [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "  TIMEOUT: $dir after ${TIMEOUT_SERVER}s"
    echo "FAILED_TIMEOUT $dir" >> "$PROGRESS_FILE"
  fi

  kill -9 $pid 2>/dev/null || true
  kill_all
done

log ""
log "  Retry 64K complete"
