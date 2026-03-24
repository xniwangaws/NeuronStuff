#!/bin/bash
#
# Supplementary benchmarks: TP=2 BS=16@16K + TP=4 full sweep
# Wave 1: 8 TP=4 tasks (devices 0-15, 2 devices each)
# Wave 2: 6 TP=4 tasks + 1 TP=2 task
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600  # 1 hour for 64K

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_server_on_port() {
  lsof -ti :${1} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

kill_all_servers() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'neuronx-cc' 2>/dev/null || true
  pkill -9 -f 'walrus_driver' 2>/dev/null || true
  sleep 3
  lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
  sleep 2
}

run_wave() {
  local wave_name=$1; shift
  local tasks=("$@")
  # Each entry: "dir_name:cores:timeout"

  local pending=()
  for entry in "${tasks[@]}"; do
    local dir="${entry%%:*}"
    if is_done "$dir"; then
      log "  SKIP (done): $dir"
    else
      pending+=("$entry")
    fi
  done

  [[ ${#pending[@]} -eq 0 ]] && { log "  $wave_name: all done"; return; }

  log ""
  log "========== $wave_name: ${#pending[@]} tasks =========="

  local pids=() ports=() dirs=() device_idx=0
  local base_port=9300

  for i in "${!pending[@]}"; do
    local entry="${pending[$i]}"
    IFS=':' read -r dir cores timeout <<< "$entry"
    local port=$((base_port + i))
    local task_dir="$BASEDIR/$dir"

    # Calculate visible cores
    local core_start=$((device_idx * 4))
    local core_end=$((core_start + cores - 1))
    local vcores="${core_start}-${core_end}"
    # Advance device_idx by number of devices used
    local num_devices=$(( (cores + 3) / 4 ))
    device_idx=$((device_idx + num_devices))

    log "  Starting $dir (cores=$vcores, port=$port)"
    sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"

    cd "$task_dir"
    NEURON_RT_VISIBLE_CORES="$vcores" bash serve.sh > serve.log 2>&1 &
    pids+=($!)
    ports+=($port)
    dirs+=("$dir")
    cd "$BASEDIR"
  done

  log "  ${#pids[@]} servers starting..."

  local done_flags=()
  for i in "${!pids[@]}"; do done_flags+=(0); done
  local elapsed=0

  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    local all_done=1
    for i in "${!pids[@]}"; do
      [[ ${done_flags[$i]} -ne 0 ]] && continue
      all_done=0
      local pid=${pids[$i]} port=${ports[$i]} dir=${dirs[$i]}
      local task_dir="$BASEDIR/$dir"

      if ! kill -0 $pid 2>/dev/null; then
        log "  FAILED_SERVE: $dir. Check $task_dir/serve.log"
        echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
        done_flags[$i]=1
        continue
      fi

      local code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
      if [[ "$code" == "200" ]]; then
        log "  READY: $dir (${elapsed}s)"
        sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"
        cd "$task_dir"; bash bench.sh > bench.log 2>&1; local rc=$?; cd "$BASEDIR"
        if [[ $rc -eq 0 ]]; then
          log "  DONE: $dir"
          echo "DONE $dir" >> "$PROGRESS_FILE"
          grep -E 'throughput|TTFT|ITL|TPOT' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
        else
          log "  FAILED_BENCH: $dir (exit=$rc)"
          echo "FAILED_BENCH $dir" >> "$PROGRESS_FILE"
        fi
        kill -9 $pid 2>/dev/null || true
        kill_server_on_port $port
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

  for i in "${!done_flags[@]}"; do
    [[ ${done_flags[$i]} -eq 0 ]] && {
      log "  TIMEOUT: ${dirs[$i]}"
      echo "FAILED_TIMEOUT ${dirs[$i]}" >> "$PROGRESS_FILE"
      kill -9 ${pids[$i]} 2>/dev/null || true
    }
  done

  kill_all_servers

  # Reset ports
  for entry in "${pending[@]}"; do
    local dir="${entry%%:*}"
    local td="$BASEDIR/$dir"
    [[ -f "$td/serve.sh" ]] && sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
    [[ -f "$td/bench.sh" ]] && sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
  done
}

# ── Main ──
log ""
log "=== Supplementary Benchmarks (TP=2 BS=16@16K + TP=4 sweep) ==="

# Wave 1: 8 TP=4 tasks (small/medium: 4K and 8K)
# Each TP=4 = 8 cores = 2 devices. 16 devices → 8 tasks max
WAVE1=(
  "qwen3_0.6b_4k_tp4_lnc2_bs1:8:1800"
  "qwen3_0.6b_4k_tp4_lnc2_bs2:8:1800"
  "qwen3_0.6b_4k_tp4_lnc2_bs16:8:1800"
  "qwen3_0.6b_4k_tp4_lnc2_bs64:8:1800"
  "qwen3_0.6b_8k_tp4_lnc2_bs1:8:1800"
  "qwen3_0.6b_8k_tp4_lnc2_bs2:8:1800"
  "qwen3_0.6b_8k_tp4_lnc2_bs16:8:1800"
  "qwen3_0.6b_8k_tp4_lnc2_bs64:8:1800"
)

# Wave 2: 16K + 32K + TP=2 supplement
# 4 TP=4 @16K (8 cores each) + 2 TP=4 @32K + 1 TP=2 @16K (4 cores) = 7 tasks
WAVE2=(
  "qwen3_0.6b_16k_tp4_lnc2_bs1:8:1800"
  "qwen3_0.6b_16k_tp4_lnc2_bs2:8:1800"
  "qwen3_0.6b_16k_tp4_lnc2_bs4:8:1800"
  "qwen3_0.6b_16k_tp4_lnc2_bs16:8:1800"
  "qwen3_0.6b_32k_tp4_lnc2_bs1:8:1800"
  "qwen3_0.6b_32k_tp4_lnc2_bs2:8:1800"
  "qwen3_0.6b_16k_tp2_lnc2_bs16:4:1800"
)

# Wave 3: 64K (slow compilation)
WAVE3=(
  "qwen3_0.6b_64k_tp4_lnc2_bs1:8:3600"
)

run_wave "Wave 1 (4K+8K TP=4)" "${WAVE1[@]}"
run_wave "Wave 2 (16K+32K TP=4 + 16K TP=2)" "${WAVE2[@]}"
run_wave "Wave 3 (64K TP=4)" "${WAVE3[@]}"

log ""
log "=== ALL SUPPLEMENTARY BENCHMARKS COMPLETE ==="
[[ -f "$PROGRESS_FILE" ]] && {
  log "  Total Done: $(grep -c '^DONE ' "$PROGRESS_FILE" || echo 0)"
  log "  Total Failed: $(grep -c '^FAILED' "$PROGRESS_FILE" || echo 0)"
}
