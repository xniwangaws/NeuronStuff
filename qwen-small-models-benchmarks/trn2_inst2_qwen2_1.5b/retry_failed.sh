#!/bin/bash
#
# Retry failed tasks — 3 small waves to reduce compile contention
# Wave A: 16k_bs1_lnc1 + 32k_bs1_lnc1 (small, fast compile, 2 tasks)
# Wave B: 64k_bs1_lnc2 + 64k_bs2_lnc2 (NEFF cached, fast) + 64k_bs4_lnc2
# Wave C: 64k_bs1_lnc1 + 64k_bs2_lnc1 + 64k_bs4_lnc1
#
set +e

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600  # 1 hour

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

kill_server_tree() {
  local pid=$1
  local port=$2
  pkill -P $pid 2>/dev/null
  kill $pid 2>/dev/null
  sleep 2
  pkill -9 -P $pid 2>/dev/null
  kill -9 $pid 2>/dev/null
  lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 1
}

run_task() {
  local dir=$1
  local task_dir="$BASEDIR/$dir"
  local port
  port=$(grep -- '--port=' "$task_dir/serve.sh" | grep -oP '\d+')

  log "[$dir] Starting serve.sh (port $port)..."
  cd "$task_dir"
  bash serve.sh > serve.log 2>&1 &
  local server_pid=$!

  local elapsed=0
  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null | grep -q "200"; then
      log "[$dir] Server ready after ${elapsed}s"
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if ! kill -0 $server_pid 2>/dev/null; then
      log "[$dir] FAILED: server process died"
      sed -i "/FAILED.*$dir/d" "$PROGRESS_FILE"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: server timeout after ${TIMEOUT_SERVER}s"
    kill_server_tree $server_pid $port
    sed -i "/FAILED.*$dir/d" "$PROGRESS_FILE"
    echo "FAILED_TIMEOUT $dir" >> "$PROGRESS_FILE"
    return 1
  fi

  log "[$dir] Running bench.sh..."
  bash bench.sh > bench.log 2>&1
  local bench_exit=$?

  if [[ $bench_exit -eq 0 ]]; then
    log "[$dir] Benchmark DONE"
    sed -i "/FAILED.*$dir/d" "$PROGRESS_FILE"
    echo "DONE $dir" >> "$PROGRESS_FILE"
  else
    log "[$dir] FAILED: bench exited with $bench_exit"
    sed -i "/FAILED.*$dir/d" "$PROGRESS_FILE"
    echo "FAILED_BENCH $dir" >> "$PROGRESS_FILE"
  fi

  kill_server_tree $server_pid $port
  return $bench_exit
}

run_wave() {
  local wave_name=$1
  shift
  local tasks=("$@")
  local pids=()

  log ""
  log "── $wave_name (${#tasks[@]} tasks parallel) ──"

  for dir in "${tasks[@]}"; do
    if grep -q "^DONE $dir" "$PROGRESS_FILE" 2>/dev/null; then
      log "[$dir] SKIP (already done)"
      continue
    fi
    run_task "$dir" &
    pids+=($!)
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    log "$wave_name: all done, skipping"
    return 0
  fi

  log "$wave_name: waiting for ${#pids[@]} tasks..."
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
  done
  log "$wave_name: done. Success=$((${#pids[@]} - failed)) Failed=$failed"

  # Clean up ports between waves
  for port in $(seq 8080 8097); do
    lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
  done
  # Clean stale compile locks
  find /var/tmp/neuron-compile-cache/ -name "*.lock" -delete 2>/dev/null
  sleep 5
}

# ── Main ──
log ""
log "============================================"
log "RETRY v2: 8 tasks in 3 waves (timeout=${TIMEOUT_SERVER}s)"
log "============================================"

# Wave A: small models, 2 tasks, cores 6 + 9, ports 8086 + 8089
run_wave "Wave A (16k+32k bs1 lnc1)" \
  "qwen2_1.5b_16k_bs1_tp1_lnc1" \
  "qwen2_1.5b_32k_bs1_tp1_lnc1"

# Wave B: 64k lnc2, cores 24-29, ports 8095-8097
# bs1 and bs2 have cached NEFFs, should be fast
run_wave "Wave B (64k lnc2)" \
  "qwen2_1.5b_64k_bs1_tp1_lnc2" \
  "qwen2_1.5b_64k_bs2_tp1_lnc2" \
  "qwen2_1.5b_64k_bs4_tp1_lnc2"

# Wave C: 64k lnc1, cores 12-14, ports 8092-8094
run_wave "Wave C (64k lnc1)" \
  "qwen2_1.5b_64k_bs1_tp1_lnc1" \
  "qwen2_1.5b_64k_bs2_tp1_lnc1" \
  "qwen2_1.5b_64k_bs4_tp1_lnc1"

log ""
log "============================================"
log "RETRY v2 COMPLETE"
log "============================================"
total_done=$(grep -c "^DONE " "$PROGRESS_FILE" 2>/dev/null || echo 0)
total_fail=$(grep -c "^FAILED" "$PROGRESS_FILE" 2>/dev/null || echo 0)
log "  Total DONE: $total_done / 24"
log "  Total FAILED: $total_fail"

# Update results MD
bash "$BASEDIR/update_results_md.sh"
