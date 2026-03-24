#!/bin/bash
# Run TP=2 LNC=2 benchmarks — 14 tasks in parallel
# Each task uses 4 cores (tp=2 * lnc=2), total 56 cores
set +e

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600

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
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: server timeout after ${TIMEOUT_SERVER}s"
    kill_server_tree $server_pid $port
    echo "FAILED_TIMEOUT $dir" >> "$PROGRESS_FILE"
    return 1
  fi

  log "[$dir] Running bench.sh..."
  bash bench.sh > bench.log 2>&1
  local bench_exit=$?

  if [[ $bench_exit -eq 0 ]]; then
    log "[$dir] Benchmark DONE"
    echo "DONE $dir" >> "$PROGRESS_FILE"
  else
    log "[$dir] FAILED: bench exited with $bench_exit"
    echo "FAILED_BENCH $dir" >> "$PROGRESS_FILE"
  fi

  kill_server_tree $server_pid $port
  return $bench_exit
}

TASKS=(
  "qwen2_1.5b_4k_bs8_tp2_lnc2"
  "qwen2_1.5b_4k_bs16_tp2_lnc2"
  "qwen2_1.5b_4k_bs32_tp2_lnc2"
  "qwen2_1.5b_8k_bs8_tp2_lnc2"
  "qwen2_1.5b_8k_bs16_tp2_lnc2"
  "qwen2_1.5b_8k_bs32_tp2_lnc2"
  "qwen2_1.5b_16k_bs1_tp2_lnc2"
  "qwen2_1.5b_16k_bs2_tp2_lnc2"
  "qwen2_1.5b_16k_bs4_tp2_lnc2"
  "qwen2_1.5b_32k_bs1_tp2_lnc2"
  "qwen2_1.5b_32k_bs2_tp2_lnc2"
  "qwen2_1.5b_32k_bs4_tp2_lnc2"
  "qwen2_1.5b_64k_bs1_tp2_lnc2"
  "qwen2_1.5b_64k_bs2_tp2_lnc2"
)

log ""
log "============================================"
log "TP=2 LNC=2: ${#TASKS[@]} tasks parallel (4 cores each, 56 total)"
log "============================================"

pids=()
for dir in "${TASKS[@]}"; do
  if grep -q "^DONE $dir" "$PROGRESS_FILE" 2>/dev/null; then
    log "[$dir] SKIP (already done)"
    continue
  fi
  run_task "$dir" &
  pids+=($!)
done

log "Launched ${#pids[@]} tasks, waiting..."

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=$((failed + 1)); fi
done

log ""
log "============================================"
log "TP=2 LNC=2 COMPLETE: Success=$((${#pids[@]} - failed)) Failed=$failed"
log "============================================"

bash "$BASEDIR/update_results_md.sh"
