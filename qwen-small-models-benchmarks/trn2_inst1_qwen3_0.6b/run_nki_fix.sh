#!/bin/bash
#
# Run the 7 previously-NKI_LIMIT tasks after topk.py fix (NIO-2589)
# These can now compile because we disabled forced NKI topk kernel.
# Runs all 7 in parallel on free devices (avoids device 0 used by retry_64k).
#
set -uo pipefail

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=9100

# task_name:cores_needed
NKI_TASKS=(
  "qwen3_0.6b_4k_tp1_lnc1_bs32:1"
  "qwen3_0.6b_4k_tp1_lnc1_bs64:1"
  "qwen3_0.6b_8k_tp1_lnc1_bs32:1"
  "qwen3_0.6b_4k_tp1_lnc2_bs32:2"
  "qwen3_0.6b_4k_tp1_lnc2_bs64:2"
  "qwen3_0.6b_8k_tp1_lnc2_bs32:2"
  "qwen3_0.6b_4k_tp2_lnc2_bs64:4"
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
log "========== NKI Fix Tasks (NIO-2589 topk.py patched) =========="

# Find free devices (skip device 0 which retry_64k may be using)
find_free_devices() {
  local used_devs=$(lsof /dev/neuron* 2>/dev/null | grep -v neuron-mo | awk 'NR>1{print $9}' | grep -oP '\d+' | sort -un)
  local free=()
  for d in $(seq 1 15); do
    local in_use=0
    for u in $used_devs; do [[ $d -eq $u ]] && { in_use=1; break; }; done
    [[ $in_use -eq 0 ]] && free+=($d)
  done
  echo "${free[@]}"
}

read -ra free_devs <<< "$(find_free_devices)"
log "  Free devices: ${free_devs[*]}"

# Assign devices to tasks
pids=()
ports=()
dirs=()
dev_idx=0

for entry in "${NKI_TASKS[@]}"; do
  dir="${entry%%:*}"
  cores="${entry##*:}"

  if is_done "$dir"; then
    log "  SKIP (done): $dir"
    pids+=("")
    ports+=(0)
    dirs+=("$dir")
    continue
  fi

  port=$((BASE_PORT + ${#pids[@]}))
  task_dir="$BASEDIR/$dir"
  dev=${free_devs[$dev_idx]}
  dev_idx=$((dev_idx + 1))

  core_start=$((dev * 4))
  core_end=$((core_start + cores - 1))
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

log "  ${#NKI_TASKS[@]} tasks launched"

# Poll and bench
done_flags=()
for i in "${!NKI_TASKS[@]}"; do
  dir="${NKI_TASKS[$i]%%:*}"
  if is_done "$dir"; then done_flags+=(1); else done_flags+=(0); fi
done

elapsed=0
while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
  all_done=1
  for i in "${!NKI_TASKS[@]}"; do
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
for entry in "${NKI_TASKS[@]}"; do
  dir="${entry%%:*}"
  td="$BASEDIR/$dir"
  [[ -f "$td/serve.sh" ]] && sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
  [[ -f "$td/bench.sh" ]] && sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
done

log ""
log "  NKI fix tasks complete"
