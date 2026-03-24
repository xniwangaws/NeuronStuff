#!/bin/bash
#
# Run 64K CP (Context Parallel) benchmarks in parallel
# Task 1: TP=2 CP=2 → 4 cores (device 0)
# Task 2: TP=4 CP=2 → 8 cores (devices 1-2)
# Task 3: TP=4 CP=4 → 8 cores (devices 3-4)
# Total: 20 cores across 5 devices
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600  # 1 hour for 64K

# task_name:cores:port:visible_cores
CP_TASKS=(
  "qwen3_0.6b_64k_tp2_cp2_lnc2_bs1:4:9200:0-3"
  "qwen3_0.6b_64k_tp4_cp2_lnc2_bs1:8:9201:4-11"
  "qwen3_0.6b_64k_tp4_cp4_lnc2_bs1:8:9202:12-19"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

is_done() {
  [[ -f "$PROGRESS_FILE" ]] && grep -q "^DONE $1$" "$PROGRESS_FILE" 2>/dev/null
}

kill_server_on_port() {
  lsof -ti :${1} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

log ""
log "========== 64K Context Parallel Benchmarks =========="

pids=()
ports=()
dirs=()

for entry in "${CP_TASKS[@]}"; do
  IFS=':' read -r dir cores port vcores <<< "$entry"

  if is_done "$dir"; then
    log "  SKIP (done): $dir"
    pids+=("")
    ports+=($port)
    dirs+=("$dir")
    continue
  fi

  task_dir="$BASEDIR/$dir"
  log "  Starting $dir (cores=$vcores, port=$port)"
  sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"

  cd "$task_dir"
  NEURON_RT_VISIBLE_CORES="$vcores" bash serve.sh > serve.log 2>&1 &
  pids+=($!)
  ports+=($port)
  dirs+=("$dir")
  cd "$BASEDIR"
done

log "  ${#CP_TASKS[@]} tasks launched"

# Poll and bench
done_flags=()
for i in "${!CP_TASKS[@]}"; do
  dir="${CP_TASKS[$i]%%:*}"
  if is_done "$dir"; then done_flags+=(1); else done_flags+=(0); fi
done

elapsed=0
while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
  all_done=1
  for i in "${!CP_TASKS[@]}"; do
    [[ ${done_flags[$i]} -ne 0 ]] && continue
    all_done=0

    pid=${pids[$i]}
    port=${ports[$i]}
    dir=${dirs[$i]}
    task_dir="$BASEDIR/$dir"

    if [[ -z "$pid" ]]; then
      done_flags[$i]=1
      continue
    fi

    if ! kill -0 $pid 2>/dev/null; then
      log "  FAILED_SERVE: $dir (process died). Check $task_dir/serve.log"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      done_flags[$i]=1
      continue
    fi

    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then
      log "  Server READY: $dir (port=$port, ${elapsed}s)"
      sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"
      log "  Running bench for $dir..."
      cd "$task_dir"; bash bench.sh > bench.log 2>&1; rc=$?; cd "$BASEDIR"
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
  sleep 10
  elapsed=$((elapsed + 10))
  if (( elapsed % 120 == 0 )); then
    n=0; for f in "${done_flags[@]}"; do [[ $f -eq 0 ]] && n=$((n+1)); done
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

# Reset ports
for entry in "${CP_TASKS[@]}"; do
  dir="${entry%%:*}"
  td="$BASEDIR/$dir"
  [[ -f "$td/serve.sh" ]] && sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
  [[ -f "$td/bench.sh" ]] && sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
done

log ""
log "  CP benchmarks complete"
