#!/bin/bash
#
# TP=2 LNC=2 tasks — parallel compilation on free devices
# Each task needs NEURON_RT_NUM_CORES=4 = 1 full device
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=9000  # Avoid conflict with wave2 (8090-8101)

# All TP=2 tasks
TP2_TASKS=(
  "qwen3_0.6b_4k_tp2_lnc2_bs16"
  "qwen3_0.6b_4k_tp2_lnc2_bs32"
  "qwen3_0.6b_4k_tp2_lnc2_bs64"
  "qwen3_0.6b_8k_tp2_lnc2_bs8"
  "qwen3_0.6b_8k_tp2_lnc2_bs16"
  "qwen3_0.6b_8k_tp2_lnc2_bs32"
  "qwen3_0.6b_16k_tp2_lnc2_bs1"
  "qwen3_0.6b_16k_tp2_lnc2_bs2"
  "qwen3_0.6b_16k_tp2_lnc2_bs4"
  "qwen3_0.6b_32k_tp2_lnc2_bs1"
  "qwen3_0.6b_32k_tp2_lnc2_bs2"
  "qwen3_0.6b_64k_tp2_lnc2_bs1"
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

# Find free devices (not held by any non-neuron-monitor process)
find_free_devices() {
  local all_devs=($(seq 0 15))
  local used_devs=$(lsof /dev/neuron* 2>/dev/null | grep -v neuron-mo | awk 'NR>1{print $9}' | grep -oP '\d+' | sort -un)
  local free=()
  for d in "${all_devs[@]}"; do
    local in_use=0
    for u in $used_devs; do
      [[ $d -eq $u ]] && { in_use=1; break; }
    done
    [[ $in_use -eq 0 ]] && free+=($d)
  done
  echo "${free[@]}"
}

log ""
log "========== Wave TP=2 (LNC=2, CORES=4) =========="

# Filter pending tasks
pending=()
for t in "${TP2_TASKS[@]}"; do
  if is_done "$t"; then
    log "  SKIP (done): $t"
  else
    pending+=("$t")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  log "  All TP=2 tasks done"
  exit 0
fi

log "  ${#pending[@]} pending tasks"

# Get free devices
read -ra free_devs <<< "$(find_free_devices)"
log "  Free devices: ${free_devs[*]}"

# Start as many as we have free devices
batch_size=${#free_devs[@]}
if [[ $batch_size -gt ${#pending[@]} ]]; then
  batch_size=${#pending[@]}
fi

pids=()
ports=()
dirs=()

for i in $(seq 0 $((batch_size - 1))); do
  dir="${pending[$i]}"
  dev=${free_devs[$i]}
  port=$((BASE_PORT + i))
  task_dir="$BASEDIR/$dir"

  # TP=2 needs all 4 cores on the device
  core_start=$((dev * 4))
  core_end=$((core_start + 3))
  visible_cores="${core_start}-${core_end}"

  log "  Starting $dir on device $dev (cores=$visible_cores, port=$port)"
  sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"

  cd "$task_dir"
  NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
  pids+=($!)
  ports+=($port)
  dirs+=("$dir")
  cd "$BASEDIR"
done

log "  ${#pids[@]} servers starting (parallel compilation)..."

# Remaining tasks to run after first batch frees devices
remaining_start=$batch_size

# Poll and bench
done_flags=()
for i in "${!pids[@]}"; do done_flags+=(0); done

elapsed=0
while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
  all_done=1
  for i in "${!pids[@]}"; do
    [[ ${done_flags[$i]} -ne 0 ]] && continue
    all_done=0

    pid=${pids[$i]}
    port=${ports[$i]}
    dir=${dirs[$i]}
    task_dir="$BASEDIR/$dir"

    if ! kill -0 $pid 2>/dev/null; then
      log "  FAILED_SERVE: $dir (process died). See $task_dir/serve.log"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      done_flags[$i]=1

      # Try to start a remaining task on the freed device
      if [[ $remaining_start -lt ${#pending[@]} ]]; then
        next_dir="${pending[$remaining_start]}"
        if ! is_done "$next_dir"; then
          read -ra cur_free <<< "$(find_free_devices)"
          if [[ ${#cur_free[@]} -gt 0 ]]; then
            next_dev=${cur_free[0]}
            next_port=$((BASE_PORT + ${#pids[@]}))
            next_cs=$((next_dev * 4))
            next_vc="${next_cs}-$((next_cs + 3))"
            next_td="$BASEDIR/$next_dir"
            log "  Backfill: starting $next_dir on device $next_dev (cores=$next_vc, port=$next_port)"
            sed -i "s/--port=[0-9]*/--port=${next_port}/" "$next_td/serve.sh"
            cd "$next_td"
            NEURON_RT_VISIBLE_CORES="$next_vc" bash serve.sh > serve.log 2>&1 &
            pids+=($!)
            ports+=($next_port)
            dirs+=("$next_dir")
            done_flags+=(0)
            cd "$BASEDIR"
          fi
        fi
        remaining_start=$((remaining_start + 1))
      fi
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
      # Kill orphan EngineCores on this device's neuron handle
      sleep 1
      done_flags[$i]=1

      # Backfill remaining tasks
      if [[ $remaining_start -lt ${#pending[@]} ]]; then
        next_dir="${pending[$remaining_start]}"
        if ! is_done "$next_dir"; then
          sleep 2
          read -ra cur_free <<< "$(find_free_devices)"
          if [[ ${#cur_free[@]} -gt 0 ]]; then
            next_dev=${cur_free[0]}
            next_port=$((BASE_PORT + ${#pids[@]}))
            next_cs=$((next_dev * 4))
            next_vc="${next_cs}-$((next_cs + 3))"
            next_td="$BASEDIR/$next_dir"
            log "  Backfill: starting $next_dir on device $next_dev (cores=$next_vc, port=$next_port)"
            sed -i "s/--port=[0-9]*/--port=${next_port}/" "$next_td/serve.sh"
            cd "$next_td"
            NEURON_RT_VISIBLE_CORES="$next_vc" bash serve.sh > serve.log 2>&1 &
            pids+=($!)
            ports+=($next_port)
            dirs+=("$next_dir")
            done_flags+=(0)
            cd "$BASEDIR"
          fi
        fi
        remaining_start=$((remaining_start + 1))
      fi
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

# Timeouts
for i in "${!done_flags[@]}"; do
  [[ ${done_flags[$i]} -eq 0 ]] && {
    log "  TIMEOUT: ${dirs[$i]}"
    echo "FAILED_TIMEOUT ${dirs[$i]}" >> "$PROGRESS_FILE"
    kill -9 ${pids[$i]} 2>/dev/null || true
  }
done

# Reset ports
for t in "${TP2_TASKS[@]}"; do
  td="$BASEDIR/$t"
  [[ -f "$td/serve.sh" ]] && sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
  [[ -f "$td/bench.sh" ]] && sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
done

log "  Wave TP=2 complete"
