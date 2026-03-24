#!/bin/bash
#
# Qwen3-0.6B Benchmark Orchestrator — Parallel Compilation + Sequential Benchmarking
#
# Strategy:
#   Wave 1: All LNC=1 tasks start in parallel (each on separate device), compile simultaneously
#   Wave 2: All LNC=2 tasks start in parallel (each on separate device), compile simultaneously
#   Within each wave: poll for health, bench each server as it becomes ready, then kill it
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800  # 30 min for compilation
BASE_PORT=8080

# ── Task definitions ──
# Format: "dir_name:lnc:num_cores"

LNC1_TASKS=(
  "qwen3_0.6b_4k_tp1_lnc1_bs16:1:1"
  "qwen3_0.6b_8k_tp1_lnc1_bs8:1:1"
  "qwen3_0.6b_8k_tp1_lnc1_bs16:1:1"
  "qwen3_0.6b_16k_tp1_lnc1_bs1:1:1"
  "qwen3_0.6b_16k_tp1_lnc1_bs2:1:1"
  "qwen3_0.6b_16k_tp1_lnc1_bs4:1:1"
  "qwen3_0.6b_32k_tp1_lnc1_bs1:1:1"
  "qwen3_0.6b_32k_tp1_lnc1_bs2:1:1"
  "qwen3_0.6b_64k_tp1_lnc1_bs1:1:1"
)

LNC2_TASKS=(
  "qwen3_0.6b_4k_tp1_lnc2_bs16:2:2"
  "qwen3_0.6b_4k_tp1_lnc2_bs32:2:2"
  "qwen3_0.6b_4k_tp1_lnc2_bs64:2:2"
  "qwen3_0.6b_8k_tp1_lnc2_bs8:2:2"
  "qwen3_0.6b_8k_tp1_lnc2_bs16:2:2"
  "qwen3_0.6b_8k_tp1_lnc2_bs32:2:2"
  "qwen3_0.6b_16k_tp1_lnc2_bs1:2:2"
  "qwen3_0.6b_16k_tp1_lnc2_bs2:2:2"
  "qwen3_0.6b_16k_tp1_lnc2_bs4:2:2"
  "qwen3_0.6b_32k_tp1_lnc2_bs1:2:2"
  "qwen3_0.6b_32k_tp1_lnc2_bs2:2:2"
  "qwen3_0.6b_64k_tp1_lnc2_bs1:2:2"
)

# ── Helpers ──

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_all_servers() {
  log "  Killing all vllm servers..."
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'neuronx-cc' 2>/dev/null || true
  pkill -9 -f 'walrus_driver' 2>/dev/null || true
  sleep 3
  # Kill any processes still holding neuron devices
  lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
  sleep 2
}

kill_server_on_port() {
  local port=$1
  lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# Run a wave of tasks with parallel compilation
# Args: wave_name, tasks_array_name
run_wave() {
  local wave_name=$1
  shift
  local tasks=("$@")
  local num_tasks=${#tasks[@]}

  # Filter out already-done tasks
  local pending_tasks=()
  local pending_indices=()
  for i in "${!tasks[@]}"; do
    local entry="${tasks[$i]}"
    local dir_name="${entry%%:*}"
    if is_done "$dir_name"; then
      log "  SKIP (done): $dir_name"
    else
      pending_tasks+=("$entry")
      pending_indices+=($i)
    fi
  done

  if [[ ${#pending_tasks[@]} -eq 0 ]]; then
    log "  All tasks in $wave_name already done, skipping wave"
    return
  fi

  log ""
  log "========== $wave_name: ${#pending_tasks[@]} pending tasks =========="

  # Assign each pending task a unique device and port
  local pids=()
  local ports=()
  local dirs=()
  local device_idx=0

  for i in "${!pending_tasks[@]}"; do
    local entry="${pending_tasks[$i]}"
    local dir_name="${entry%%:*}"
    local rest="${entry#*:}"
    local lnc="${rest%%:*}"
    local num_cores="${rest##*:}"

    local port=$((BASE_PORT + i))
    local task_dir="$BASEDIR/$dir_name"

    # Device assignment: each task gets its own device
    # Device N -> physical cores N*4 to N*4+3
    local dev=$device_idx
    device_idx=$((device_idx + 1))

    local core_start=$((dev * 4))
    if [[ $num_cores -eq 1 ]]; then
      local visible_cores="$core_start"
    else
      local core_end=$((core_start + num_cores - 1))
      local visible_cores="${core_start}-${core_end}"
    fi

    log "  Starting $dir_name on device $dev (cores=$visible_cores, port=$port)"

    # Update serve.sh port to unique port for this wave
    sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"

    # Start server
    cd "$task_dir"
    NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
    local pid=$!

    pids+=($pid)
    ports+=($port)
    dirs+=("$dir_name")

    cd "$BASEDIR"
  done

  log "  All ${#pids[@]} servers starting (parallel compilation)..."

  # Poll for health and bench each as it becomes ready
  local done_flags=()
  for i in "${!pids[@]}"; do done_flags+=(0); done

  local elapsed=0
  local all_done=0

  while [[ $elapsed -lt $TIMEOUT_SERVER && $all_done -eq 0 ]]; do
    all_done=1
    for i in "${!pids[@]}"; do
      # Skip already processed
      if [[ ${done_flags[$i]} -ne 0 ]]; then
        continue
      fi
      all_done=0

      local pid=${pids[$i]}
      local port=${ports[$i]}
      local dir_name=${dirs[$i]}
      local task_dir="$BASEDIR/$dir_name"

      # Check if process died
      if ! kill -0 $pid 2>/dev/null; then
        log "  FAILED_SERVE: $dir_name (process died). See $task_dir/serve.log"
        echo "FAILED_SERVE $dir_name" >> "$PROGRESS_FILE"
        done_flags[$i]=1
        continue
      fi

      # Check health
      local http_code
      http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
      if [[ "$http_code" == "200" ]]; then
        log "  Server READY: $dir_name (port=$port, ${elapsed}s)"

        # Update bench.sh port
        sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"

        # Run benchmark
        log "  Running bench for $dir_name..."
        cd "$task_dir"
        bash bench.sh > bench.log 2>&1
        local bench_exit=$?
        cd "$BASEDIR"

        if [[ $bench_exit -eq 0 ]]; then
          log "  DONE: $dir_name"
          echo "DONE $dir_name" >> "$PROGRESS_FILE"
          grep -E 'throughput|TTFT|ITL|TPOT' "$task_dir/bench.log" | head -10 | while read -r line; do
            log "    $line"
          done
        else
          log "  FAILED_BENCH: $dir_name (exit=$bench_exit)"
          echo "FAILED_BENCH $dir_name" >> "$PROGRESS_FILE"
        fi

        # Kill this server to free the device
        kill -9 $pid 2>/dev/null || true
        kill_server_on_port $port
        done_flags[$i]=1
      fi
    done

    if [[ $all_done -eq 0 ]]; then
      sleep 10
      elapsed=$((elapsed + 10))
      if (( elapsed % 120 == 0 )); then
        local still_compiling=0
        for i in "${!done_flags[@]}"; do
          if [[ ${done_flags[$i]} -eq 0 ]]; then still_compiling=$((still_compiling + 1)); fi
        done
        log "  Compiling... ${elapsed}s elapsed, $still_compiling tasks still compiling"
      fi
    fi
  done

  # Handle timeouts
  for i in "${!done_flags[@]}"; do
    if [[ ${done_flags[$i]} -eq 0 ]]; then
      local dir_name=${dirs[$i]}
      local pid=${pids[$i]}
      log "  TIMEOUT: $dir_name after ${TIMEOUT_SERVER}s"
      echo "FAILED_TIMEOUT $dir_name" >> "$PROGRESS_FILE"
      kill -9 $pid 2>/dev/null || true
    fi
  done

  # Clean up any remaining servers from this wave
  kill_all_servers

  # Reset ports back to 8080
  for i in "${!pending_tasks[@]}"; do
    local entry="${pending_tasks[$i]}"
    local dir_name="${entry%%:*}"
    local task_dir="$BASEDIR/$dir_name"
    sed -i "s/--port=[0-9]*/--port=8080/" "$task_dir/serve.sh"
    sed -i "s/^PORT=.*/PORT=8080/" "$task_dir/bench.sh"
  done
}

# ── Main ──

log "============================================"
log "Qwen3-0.6B PARALLEL Compilation Benchmark"
log "LNC=1 tasks: ${#LNC1_TASKS[@]} | LNC=2 tasks: ${#LNC2_TASKS[@]}"
log "============================================"

# Wave 1: LNC=1 (9 tasks, each on separate device)
run_wave "Wave 1 (LNC=1)" "${LNC1_TASKS[@]}"

# Wave 2: LNC=2 (12 tasks, each on separate device)
run_wave "Wave 2 (LNC=2)" "${LNC2_TASKS[@]}"

# ── Summary ──
log ""
log "============================================"
log "ALL WAVES COMPLETE"
log "============================================"
if [[ -f "$PROGRESS_FILE" ]]; then
  total_done=$(grep -c "^DONE " "$PROGRESS_FILE" 2>/dev/null || echo 0)
  total_fail=$(grep -c "^FAILED" "$PROGRESS_FILE" 2>/dev/null || echo 0)
  log "  Completed: $total_done"
  log "  Failed:    $total_fail"
  if [[ $total_fail -gt 0 ]]; then
    log "  Failed tasks:"
    grep "^FAILED" "$PROGRESS_FILE" | while read -r line; do log "    $line"; done
  fi
fi
