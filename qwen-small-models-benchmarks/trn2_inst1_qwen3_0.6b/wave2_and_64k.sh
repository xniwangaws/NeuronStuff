#!/bin/bash
#
# Start Wave 2 (LNC=2) on devices 0-3,5-15 while 64k_lnc1 compiles on device 4
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=8090  # Use 8090+ for wave2 to avoid conflict with 64k on 8084

# 64k task info (already running)
LNC1_64K_DIR="qwen3_0.6b_64k_tp1_lnc1_bs1"
LNC1_64K_PORT=8084
LNC1_64K_PID=195400

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

# Available devices: 0-3, 5-15 (skip 4, used by 64k_lnc1)
AVAILABLE_DEVICES=(0 1 2 3 5 6 7 8 9 10 11 12 13 14 15)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_server_on_port() {
  lsof -ti :${1} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# ── Start all LNC=2 tasks ──

log ""
log "========== Wave 2 (LNC=2) + 64k_lnc1 monitor =========="

pids=()
ports=()
dirs=()
dev_idx=0

for i in "${!LNC2_TASKS[@]}"; do
  entry="${LNC2_TASKS[$i]}"
  dir_name="${entry%%:*}"
  rest="${entry#*:}"
  num_cores="${rest##*:}"

  if is_done "$dir_name"; then
    log "  SKIP (done): $dir_name"
    pids+=("")
    ports+=(0)
    dirs+=("$dir_name")
    continue
  fi

  port=$((BASE_PORT + i))
  task_dir="$BASEDIR/$dir_name"
  dev=${AVAILABLE_DEVICES[$dev_idx]}
  dev_idx=$((dev_idx + 1))

  core_start=$((dev * 4))
  core_end=$((core_start + num_cores - 1))
  visible_cores="${core_start}-${core_end}"

  log "  Starting $dir_name on device $dev (cores=$visible_cores, port=$port)"

  sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"

  cd "$task_dir"
  NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
  pid=$!

  pids+=($pid)
  ports+=($port)
  dirs+=("$dir_name")
  cd "$BASEDIR"
done

log "  Wave 2: ${#LNC2_TASKS[@]} tasks launched. Also monitoring 64k_lnc1 on port $LNC1_64K_PORT"

# ── Poll and bench ──

done_flags=()
for i in "${!LNC2_TASKS[@]}"; do
  if is_done "${dirs[$i]}"; then done_flags+=(1); else done_flags+=(0); fi
done

# Also track 64k_lnc1
lnc1_64k_done=0
if is_done "$LNC1_64K_DIR"; then lnc1_64k_done=1; fi

elapsed=0
while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
  all_done=1

  # Check 64k_lnc1
  if [[ $lnc1_64k_done -eq 0 ]]; then
    if ! kill -0 $LNC1_64K_PID 2>/dev/null; then
      log "  FAILED_SERVE: $LNC1_64K_DIR (process died)"
      echo "FAILED_SERVE $LNC1_64K_DIR" >> "$PROGRESS_FILE"
      lnc1_64k_done=1
    else
      code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${LNC1_64K_PORT}/health" 2>/dev/null || echo "000")
      if [[ "$code" == "200" ]]; then
        log "  Server READY: $LNC1_64K_DIR (port=$LNC1_64K_PORT, ${elapsed}s)"
        task_dir="$BASEDIR/$LNC1_64K_DIR"
        sed -i "s/^PORT=.*/PORT=${LNC1_64K_PORT}/" "$task_dir/bench.sh"
        log "  Running bench for $LNC1_64K_DIR..."
        cd "$task_dir"; bash bench.sh > bench.log 2>&1; rc=$?; cd "$BASEDIR"
        if [[ $rc -eq 0 ]]; then
          log "  DONE: $LNC1_64K_DIR"
          echo "DONE $LNC1_64K_DIR" >> "$PROGRESS_FILE"
          grep -E 'throughput|TTFT|ITL|TPOT' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
        else
          log "  FAILED_BENCH: $LNC1_64K_DIR"
          echo "FAILED_BENCH $LNC1_64K_DIR" >> "$PROGRESS_FILE"
        fi
        kill -9 $LNC1_64K_PID 2>/dev/null || true
        kill_server_on_port $LNC1_64K_PORT
        lnc1_64k_done=1
      else
        all_done=0
      fi
    fi
  fi

  # Check LNC=2 tasks
  for i in "${!LNC2_TASKS[@]}"; do
    [[ ${done_flags[$i]} -ne 0 ]] && continue
    all_done=0

    pid=${pids[$i]}
    port=${ports[$i]}
    dir_name=${dirs[$i]}
    task_dir="$BASEDIR/$dir_name"

    if ! kill -0 $pid 2>/dev/null; then
      log "  FAILED_SERVE: $dir_name (process died). See $task_dir/serve.log"
      echo "FAILED_SERVE $dir_name" >> "$PROGRESS_FILE"
      done_flags[$i]=1
      continue
    fi

    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
      log "  Server READY: $dir_name (port=$port, ${elapsed}s)"
      sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"
      log "  Running bench for $dir_name..."
      cd "$task_dir"; bash bench.sh > bench.log 2>&1; rc=$?; cd "$BASEDIR"
      if [[ $rc -eq 0 ]]; then
        log "  DONE: $dir_name"
        echo "DONE $dir_name" >> "$PROGRESS_FILE"
        grep -E 'throughput|TTFT|ITL|TPOT' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
      else
        log "  FAILED_BENCH: $dir_name (exit=$rc)"
        echo "FAILED_BENCH $dir_name" >> "$PROGRESS_FILE"
      fi
      kill -9 $pid 2>/dev/null || true
      kill_server_on_port $port
      done_flags[$i]=1
    fi
  done

  [[ $all_done -eq 1 ]] && break
  sleep 10
  elapsed=$((elapsed + 10))
  if (( elapsed % 120 == 0 )); then
    n=0
    for f in "${done_flags[@]}"; do [[ $f -eq 0 ]] && n=$((n+1)); done
    [[ $lnc1_64k_done -eq 0 ]] && n=$((n+1))
    log "  Compiling... ${elapsed}s, $n tasks remaining"
  fi
done

# Handle timeouts
for i in "${!done_flags[@]}"; do
  if [[ ${done_flags[$i]} -eq 0 ]]; then
    log "  TIMEOUT: ${dirs[$i]}"
    echo "FAILED_TIMEOUT ${dirs[$i]}" >> "$PROGRESS_FILE"
    kill -9 ${pids[$i]} 2>/dev/null || true
  fi
done
if [[ $lnc1_64k_done -eq 0 ]]; then
  log "  TIMEOUT: $LNC1_64K_DIR"
  echo "FAILED_TIMEOUT $LNC1_64K_DIR" >> "$PROGRESS_FILE"
  kill -9 $LNC1_64K_PID 2>/dev/null || true
fi

# Cleanup
log "  Cleaning up..."
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'neuronx-cc' 2>/dev/null || true
pkill -9 -f 'walrus_driver' 2>/dev/null || true
sleep 3
lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9 2>/dev/null || true

# Reset ports
for i in "${!LNC2_TASKS[@]}"; do
  dir_name="${LNC2_TASKS[$i]%%:*}"
  td="$BASEDIR/$dir_name"
  sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
  sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
done
sed -i "s/^PORT=.*/PORT=8080/" "$BASEDIR/$LNC1_64K_DIR/bench.sh"

# Summary
log ""
log "=== ALL COMPLETE ==="
total_done=$(grep -c "^DONE " "$PROGRESS_FILE" 2>/dev/null || echo 0)
total_fail=$(grep -c "^FAILED" "$PROGRESS_FILE" 2>/dev/null || echo 0)
log "  Done: $total_done | Failed: $total_fail"
grep "^FAILED" "$PROGRESS_FILE" 2>/dev/null | while read -r l; do log "    $l"; done
