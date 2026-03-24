#!/bin/bash
#
# Qwen2-1.5B Benchmark Orchestrator — Parallel Wave Execution
#
# Wave 1: 12x LNC=1 tasks in parallel (cores 0-14, ports 8080-8094)
# Wave 2: 12x LNC=2 tasks in parallel (cores 0-29, ports 8080-8094)
#
set +e  # don't exit on task failure

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
SUMMARY_FILE="$BASEDIR/benchmark_summary.txt"
TIMEOUT_SERVER=1800  # 30min for 32k/64k compile

# ── Wave definitions ──
# Removed: 4k_bs32, 4k_bs64, 8k_bs32 — NeuronX compiler error:
#   "max8 has N elements per partition, must be between 8 and 16384"
WAVE1=(
  "qwen2_1.5b_4k_bs16_tp1_lnc1"
  "qwen2_1.5b_8k_bs8_tp1_lnc1"
  "qwen2_1.5b_8k_bs16_tp1_lnc1"
  "qwen2_1.5b_16k_bs1_tp1_lnc1"
  "qwen2_1.5b_16k_bs2_tp1_lnc1"
  "qwen2_1.5b_16k_bs4_tp1_lnc1"
  "qwen2_1.5b_32k_bs1_tp1_lnc1"
  "qwen2_1.5b_32k_bs2_tp1_lnc1"
  "qwen2_1.5b_32k_bs4_tp1_lnc1"
  "qwen2_1.5b_64k_bs1_tp1_lnc1"
  "qwen2_1.5b_64k_bs2_tp1_lnc1"
  "qwen2_1.5b_64k_bs4_tp1_lnc1"
)

WAVE2=(
  "qwen2_1.5b_4k_bs16_tp1_lnc2"
  "qwen2_1.5b_8k_bs8_tp1_lnc2"
  "qwen2_1.5b_8k_bs16_tp1_lnc2"
  "qwen2_1.5b_16k_bs1_tp1_lnc2"
  "qwen2_1.5b_16k_bs2_tp1_lnc2"
  "qwen2_1.5b_16k_bs4_tp1_lnc2"
  "qwen2_1.5b_32k_bs1_tp1_lnc2"
  "qwen2_1.5b_32k_bs2_tp1_lnc2"
  "qwen2_1.5b_32k_bs4_tp1_lnc2"
  "qwen2_1.5b_64k_bs1_tp1_lnc2"
  "qwen2_1.5b_64k_bs2_tp1_lnc2"
  "qwen2_1.5b_64k_bs4_tp1_lnc2"
)

# ── Helpers ──

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  local dir=$1
  [[ -f "$PROGRESS_FILE" ]] && grep -q "DONE $dir" "$PROGRESS_FILE" 2>/dev/null
}

# Scan all bench.log and write summary
update_summary() {
  {
    echo "# Qwen2-1.5B Benchmark Summary"
    echo "# Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    printf "%-40s %-8s %-10s %-12s %-12s %-10s %-10s %-12s\n" \
      "Task" "Status" "TTFT(ms)" "TPOT(ms)" "ITL(ms)" "Req/s" "Tok/s" "TotalTok/s"
    printf "%-40s %-8s %-10s %-12s %-12s %-10s %-10s %-12s\n" \
      "----------------------------------------" "--------" "----------" "------------" "------------" "----------" "----------" "------------"

    for dir in "$BASEDIR"/qwen2_1.5b_*/; do
      task=$(basename "$dir")
      bench="$dir/bench.log"

      # Check status
      if grep -q "^DONE $task" "$PROGRESS_FILE" 2>/dev/null; then
        status="DONE"
      elif grep -q "^FAILED" "$PROGRESS_FILE" 2>/dev/null && grep -q "$task" "$PROGRESS_FILE" 2>/dev/null; then
        reason=$(grep "$task" "$PROGRESS_FILE" | head -1 | awk '{print $1}')
        status="$reason"
      else
        status="PENDING"
      fi

      if [[ "$status" == "DONE" ]] && [[ -f "$bench" ]]; then
        ttft=$(grep "Mean TTFT" "$bench" | awk '{print $NF}')
        tpot=$(grep "Mean TPOT" "$bench" | awk '{print $NF}')
        itl=$(grep "Mean ITL" "$bench" | awk '{print $NF}')
        reqs=$(grep "Request throughput" "$bench" | awk '{print $NF}')
        toks=$(grep "Output token throughput" "$bench" | head -1 | awk '{print $NF}')
        total_toks=$(grep "Total token throughput" "$bench" | awk '{print $NF}')
        printf "%-40s %-8s %-10s %-12s %-12s %-10s %-10s %-12s\n" \
          "$task" "$status" "$ttft" "$tpot" "$itl" "$reqs" "$toks" "$total_toks"
      else
        printf "%-40s %-8s\n" "$task" "$status"
      fi
    done

    echo ""
    echo "# Done: $(grep -c '^DONE' "$PROGRESS_FILE" 2>/dev/null || echo 0)"
    echo "# Failed: $(grep -c '^FAILED' "$PROGRESS_FILE" 2>/dev/null || echo 0)"
  } > "$SUMMARY_FILE"
}

# Kill server and ALL its child processes (without killing our own process group)
kill_server_tree() {
  local pid=$1
  local port=$2
  # Kill children recursively, then the parent
  pkill -P $pid 2>/dev/null
  kill $pid 2>/dev/null
  sleep 2
  pkill -9 -P $pid 2>/dev/null
  kill -9 $pid 2>/dev/null
  # Also kill anything on this port
  lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 1
}

# Run a single task: start server, wait, bench, kill
run_task() {
  local dir=$1
  local task_dir="$BASEDIR/$dir"
  local port
  port=$(grep -- '--port=' "$task_dir/serve.sh" | grep -oP '\d+')

  log "[$dir] Starting serve.sh (port $port)..."
  cd "$task_dir"
  bash serve.sh > serve.log 2>&1 &
  local server_pid=$!

  # Wait for server ready
  local elapsed=0
  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null | grep -q "200"; then
      log "[$dir] Server ready after ${elapsed}s"
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if ! kill -0 $server_pid 2>/dev/null; then
      log "[$dir] FAILED: server process died. Check $task_dir/serve.log"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      update_summary
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: server timeout after ${TIMEOUT_SERVER}s"
    kill_server_tree $server_pid $port
    echo "FAILED_TIMEOUT $dir" >> "$PROGRESS_FILE"
    update_summary
    return 1
  fi

  # Run benchmark
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

  # Kill server and update summary
  kill_server_tree $server_pid $port
  update_summary

  return $bench_exit
}

# Run a wave: launch all tasks in parallel, wait for all
run_wave() {
  local wave_name=$1
  shift
  local tasks=("$@")
  local pids=()
  local count=${#tasks[@]}
  local skip=0

  log ""
  log "╔══════════════════════════════════════════════════════════╗"
  log "║  $wave_name — $count tasks in parallel"
  log "╚══════════════════════════════════════════════════════════╝"

  for dir in "${tasks[@]}"; do
    if is_done "$dir"; then
      log "[$dir] SKIP (already done)"
      skip=$((skip + 1))
      continue
    fi
    run_task "$dir" &
    pids+=($!)
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    log "$wave_name: all tasks already done, skipping"
    return 0
  fi

  log "$wave_name: launched ${#pids[@]} tasks ($skip skipped), waiting..."

  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed=$((failed + 1))
    fi
  done

  log "$wave_name: finished. Success: $((${#pids[@]} - failed)), Failed: $failed"
  update_summary
  return 0
}

# ── Main ──

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

log "============================================"
log "Qwen2-1.5B Parallel Benchmark Orchestrator"
log "============================================"
log ""
log "Plan:"
log "  Wave 1: 12x LNC=1 — cores 0-14, ports 8080-8094"
log "  Wave 2: 12x LNC=2 — cores 0-29, ports 8080-8094"
log "  TIMEOUT_SERVER=${TIMEOUT_SERVER}s"
log "  Summary: $SUMMARY_FILE"
log ""

# Check neuron cores
used=$(neuron-ls --json-output 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
used=set()
for x in d:
  for p in x.get('neuron_processes',[]):
    used.update(p.get('neuroncore_ids',[]))
print(len(used))
" 2>/dev/null)
if [[ "$used" -gt 0 ]]; then
  log "WARNING: $used neuron core(s) currently in use"
fi

# Wave 1
run_wave "Wave 1 (LNC=1)" "${WAVE1[@]}"

# Cooldown — kill any leftover on all ports
log ""
log "Cooldown between waves (15s)..."
for port in $(seq 8080 8094); do
  lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
done
sleep 15

# Wave 2
run_wave "Wave 2 (LNC=2)" "${WAVE2[@]}"

# ── Final Summary ──
update_summary
log ""
log "============================================"
log "ALL WAVES COMPLETE"
log "============================================"
total_done=$(grep -c "^DONE " "$PROGRESS_FILE" 2>/dev/null || echo 0)
total_fail=$(grep -c "^FAILED" "$PROGRESS_FILE" 2>/dev/null || echo 0)
log "  Completed: $total_done / 24"
log "  Failed:    $total_fail"
if [[ $total_fail -gt 0 ]]; then
  log "  Failed tasks:"
  grep "^FAILED" "$PROGRESS_FILE" | while read -r line; do
    log "    $line"
  done
fi
log ""
log "Summary:  $SUMMARY_FILE"
log "Progress: $PROGRESS_FILE"
