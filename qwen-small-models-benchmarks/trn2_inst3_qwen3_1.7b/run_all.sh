#!/bin/bash
#
# Parallel Compilation + Sequential Benchmarking Orchestrator
# Qwen3-1.7B on trn2.48xlarge (16 devices, LNC=2)
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=8080
TOTAL_DEVICES=16

# Format: "dir_name:cores_per_device:devices_per_task"

# Wave 1: TP=1, LNC=1 — 1 core/dev, 1 dev/task, max 16 parallel
LNC1_TASKS=(
  "qwen3_4k_tp1_lnc1_bs1:1:1"
  "qwen3_4k_tp1_lnc1_bs2:1:1"
  "qwen3_4k_tp1_lnc1_bs16:1:1"
  "qwen3_4k_tp1_lnc1_bs64:1:1"
  "qwen3_8k_tp1_lnc1_bs1:1:1"
  "qwen3_8k_tp1_lnc1_bs2:1:1"
  "qwen3_8k_tp1_lnc1_bs16:1:1"
  "qwen3_8k_tp1_lnc1_bs64:1:1"
  "qwen3_16k_tp1_lnc1_bs1:1:1"
  "qwen3_16k_tp1_lnc1_bs2:1:1"
  "qwen3_16k_tp1_lnc1_bs16:1:1"
  "qwen3_16k_tp1_lnc1_bs64:1:1"
  "qwen3_32k_tp1_lnc1_bs1:1:1"
  "qwen3_32k_tp1_lnc1_bs2:1:1"
  "qwen3_64k_tp1_lnc1_bs1:1:1"
  "qwen3_64k_tp1_lnc1_bs2:1:1"
)

# Wave 2a: TP=1, LNC=2 — 2 cores/dev, 1 dev/task, max 16 parallel
WAVE2A_TASKS=(
  "qwen3_4k_tp1_lnc2_bs1:2:1"
  "qwen3_4k_tp1_lnc2_bs2:2:1"
  "qwen3_4k_tp1_lnc2_bs16:2:1"
  "qwen3_4k_tp1_lnc2_bs64:2:1"
  "qwen3_8k_tp1_lnc2_bs1:2:1"
  "qwen3_8k_tp1_lnc2_bs2:2:1"
  "qwen3_8k_tp1_lnc2_bs16:2:1"
  "qwen3_8k_tp1_lnc2_bs64:2:1"
  "qwen3_16k_tp1_lnc2_bs1:2:1"
  "qwen3_16k_tp1_lnc2_bs2:2:1"
  "qwen3_16k_tp1_lnc2_bs16:2:1"
  "qwen3_16k_tp1_lnc2_bs64:2:1"
  "qwen3_32k_tp1_lnc2_bs1:2:1"
  "qwen3_32k_tp1_lnc2_bs2:2:1"
  "qwen3_64k_tp1_lnc2_bs1:2:1"
  "qwen3_64k_tp1_lnc2_bs2:2:1"
)

# Wave 2b: TP=2, LNC=2 — 4 logical cores/task (1 full device), max 16 parallel
# TP=2 × NEURON_RT_NUM_CORES=2 = 4 logical cores = 1 device
WAVE2B_TASKS=(
  "qwen3_4k_tp2_lnc2_bs1:4:1"
  "qwen3_4k_tp2_lnc2_bs2:4:1"
  "qwen3_4k_tp2_lnc2_bs16:4:1"
  "qwen3_4k_tp2_lnc2_bs64:4:1"
  "qwen3_8k_tp2_lnc2_bs1:4:1"
  "qwen3_8k_tp2_lnc2_bs2:4:1"
  "qwen3_8k_tp2_lnc2_bs16:4:1"
  "qwen3_8k_tp2_lnc2_bs64:4:1"
  "qwen3_16k_tp2_lnc2_bs1:4:1"
  "qwen3_16k_tp2_lnc2_bs2:4:1"
  "qwen3_16k_tp2_lnc2_bs16:4:1"
  "qwen3_16k_tp2_lnc2_bs64:4:1"
  "qwen3_32k_tp2_lnc2_bs1:4:1"
  "qwen3_32k_tp2_lnc2_bs2:4:1"
  "qwen3_64k_tp2_lnc2_bs1:4:1"
  "qwen3_64k_tp2_lnc2_bs2:4:1"
)

# Wave 2c: TP=4, LNC=2 — 8 logical cores/task (2 full devices), max 8 parallel
# TP=4 × NEURON_RT_NUM_CORES=2 = 8 logical cores = 2 devices
WAVE2C_TASKS=(
  "qwen3_4k_tp4_lnc2_bs1:8:2"
  "qwen3_4k_tp4_lnc2_bs2:8:2"
  "qwen3_4k_tp4_lnc2_bs16:8:2"
  "qwen3_4k_tp4_lnc2_bs64:8:2"
  "qwen3_8k_tp4_lnc2_bs1:8:2"
  "qwen3_8k_tp4_lnc2_bs2:8:2"
  "qwen3_8k_tp4_lnc2_bs16:8:2"
  "qwen3_8k_tp4_lnc2_bs64:8:2"
  "qwen3_16k_tp4_lnc2_bs1:8:2"
  "qwen3_16k_tp4_lnc2_bs2:8:2"
  "qwen3_16k_tp4_lnc2_bs16:8:2"
  "qwen3_16k_tp4_lnc2_bs64:8:2"
  "qwen3_32k_tp4_lnc2_bs1:8:2"
  "qwen3_32k_tp4_lnc2_bs2:8:2"
  "qwen3_64k_tp4_lnc2_bs1:8:2"
  "qwen3_64k_tp4_lnc2_bs2:8:2"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_all_servers() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'neuronx-cc' 2>/dev/null || true
  pkill -9 -f 'walrus_driver' 2>/dev/null || true
  sleep 3
  lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
  sleep 2
}

kill_server_on_port() {
  lsof -ti :${1} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# Build NEURON_RT_VISIBLE_CORES string
# Args: base_device, total_logical_cores
# LNC=2: each device has 4 logical cores (8 physical / 2)
# Core IDs are contiguous: device 0 = 0-3, device 1 = 4-7, etc.
build_visible_cores() {
  local base_dev=$1 total_cores=$2
  local core_start=$((base_dev * 4))
  if [[ $total_cores -eq 1 ]]; then
    echo "$core_start"
  else
    echo "${core_start}-$((core_start + total_cores - 1))"
  fi
}

# Run a wave: starts tasks in batches fitting within available devices
# Args: wave_name, tasks...
run_wave() {
  local wave_name=$1; shift
  local tasks=("$@")

  # Filter out already-done tasks
  local pending_tasks=()
  for entry in "${tasks[@]}"; do
    local dir_name="${entry%%:*}"
    if is_done "$dir_name"; then
      log "  SKIP (done): $dir_name"
    else
      pending_tasks+=("$entry")
    fi
  done

  [[ ${#pending_tasks[@]} -eq 0 ]] && { log "  $wave_name: all done"; return; }

  # Parse devices_per_task from first entry to determine batch size
  local first="${pending_tasks[0]}"
  local devs_per_task="${first##*:}"
  local max_parallel=$((TOTAL_DEVICES / devs_per_task))

  log "========== $wave_name: ${#pending_tasks[@]} pending (max $max_parallel parallel, $devs_per_task dev/task) =========="

  # Process in batches
  local batch_start=0
  while [[ $batch_start -lt ${#pending_tasks[@]} ]]; do
    local batch_end=$((batch_start + max_parallel))
    [[ $batch_end -gt ${#pending_tasks[@]} ]] && batch_end=${#pending_tasks[@]}
    local batch_size=$((batch_end - batch_start))

    log "  --- Batch: tasks $((batch_start+1))-$batch_end of ${#pending_tasks[@]} ---"

    local pids=() ports=() dirs=()

    for ((i=batch_start; i<batch_end; i++)); do
      local entry="${pending_tasks[$i]}"
      local dir_name="${entry%%:*}"
      local rest="${entry#*:}"
      local total_cores="${rest%%:*}"
      local slot=$((i - batch_start))
      local base_dev=$((slot * devs_per_task))
      local port=$((BASE_PORT + slot))
      local task_dir="$BASEDIR/$dir_name"

      local visible_cores
      visible_cores=$(build_visible_cores "$base_dev" "$total_cores")

      log "  Starting $dir_name devices=$base_dev-$((base_dev+devs_per_task-1)) cores=$visible_cores port=$port"
      sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"
      cd "$task_dir"
      NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
      pids+=($!); ports+=($port); dirs+=("$dir_name")
      cd "$BASEDIR"
    done

    log "  ${#pids[@]} servers starting (parallel compilation)..."

    local done_flags=()
    for ((i=0; i<${#pids[@]}; i++)); do done_flags+=(0); done
    local elapsed=0

    while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
      local all_done=1
      for i in "${!pids[@]}"; do
        [[ ${done_flags[$i]} -ne 0 ]] && continue
        all_done=0
        local pid=${pids[$i]} port=${ports[$i]} dir_name=${dirs[$i]}
        local task_dir="$BASEDIR/$dir_name"

        if ! kill -0 "$pid" 2>/dev/null; then
          log "  FAILED_SERVE: $dir_name (process died)"
          echo "FAILED_SERVE $dir_name" >> "$PROGRESS_FILE"
          done_flags[$i]=1
          continue
        fi

        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
        if [[ "$code" == "200" ]]; then
          log "  READY: $dir_name (${elapsed}s)"
          sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"
          cd "$task_dir"
          bash bench.sh > bench.log 2>&1
          local rc=$?
          cd "$BASEDIR"
          if [[ $rc -eq 0 ]]; then
            log "  DONE: $dir_name"
            echo "DONE $dir_name" >> "$PROGRESS_FILE"
            grep -E 'throughput|TTFT|ITL' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
          else
            log "  FAILED_BENCH: $dir_name (rc=$rc)"
            echo "FAILED_BENCH $dir_name" >> "$PROGRESS_FILE"
          fi
          kill -9 "$pid" 2>/dev/null || true
          kill_server_on_port "$port"
          done_flags[$i]=1
        fi
      done
      [[ $all_done -eq 1 ]] && break
      sleep 10; elapsed=$((elapsed + 10))
      (( elapsed % 120 == 0 )) && {
        local n=0; for f in "${done_flags[@]}"; do [[ $f -eq 0 ]] && n=$((n+1)); done
        log "  Compiling... ${elapsed}s, $n remaining"
      }
    done

    # Handle timeouts
    for i in "${!done_flags[@]}"; do
      [[ ${done_flags[$i]} -eq 0 ]] && {
        log "  TIMEOUT: ${dirs[$i]}"
        echo "FAILED_TIMEOUT ${dirs[$i]}" >> "$PROGRESS_FILE"
        kill -9 "${pids[$i]}" 2>/dev/null || true
      }
    done

    kill_all_servers

    # Reset ports back to 8080
    for ((i=batch_start; i<batch_end; i++)); do
      local dir_name="${pending_tasks[$i]%%:*}"
      local td="$BASEDIR/$dir_name"
      sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
      sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
    done

    batch_start=$batch_end
  done
}

# ── Main ──
log "=== Qwen3-1.7B Parallel Compilation Benchmark ==="
log "=== $(date) ==="

run_wave "Wave 1 (TP=1, LNC=1)" "${LNC1_TASKS[@]}"
run_wave "Wave 2a (TP=1, LNC=2)" "${WAVE2A_TASKS[@]}"
run_wave "Wave 2b (TP=2, LNC=2)" "${WAVE2B_TASKS[@]}"
run_wave "Wave 2c (TP=4, LNC=2)" "${WAVE2C_TASKS[@]}"

log "=== ALL WAVES COMPLETE ==="
[[ -f "$PROGRESS_FILE" ]] && {
  log "  Done: $(grep -c '^DONE ' "$PROGRESS_FILE" || echo 0)"
  log "  Failed: $(grep -c '^FAILED' "$PROGRESS_FILE" || echo 0)"
  grep "^FAILED" "$PROGRESS_FILE" | while read -r l; do log "    $l"; done
}
