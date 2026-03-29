# Neuron Benchmark Agent Guide

## Goal
Systematically benchmark LLM models on Trn2, covering combinations of input lengths, TP, LNC, and batch sizes. Each combination gets its own folder with `serve.sh` and `bench.sh`. All compilations run in parallel; benchmarks run sequentially as servers become ready.

## Environment
- **Instance**: trn2.48xlarge (16 Neuron devices, 64 physical cores)
- **venv**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`
- **vLLM**: v0.13.0 with Neuron plugin (`vllm_neuron`), framework: `neuronx-distributed-inference`

## Critical Rules & Principles

### 1. logical-neuroncore-config (System-Level LNC)
Check via `neuron-ls`. trn2.48xlarge defaults to LNC=2.
- LNC=1 tasks: `NEURON_RT_NUM_CORES=1`, model config `logical_nc_config: 1`
- LNC=2 tasks: `NEURON_RT_NUM_CORES=2`, model config `logical_nc_config: 2`

### 2. NKI 16384 Elements-Per-Partition Limit
The NKI kernel `max8` op limits each partition to 16384 elements max. LNC=1 with large batch sizes can exceed this. The error looks like:
```
SyntaxError: max8 has 18992 elements per partition, but must be between 8 and 16384
```
**Rule**: If a config fails with this error, it's a hard limit — skip it and mark as NKI_LIMIT.

### 3. Isolated BASE_COMPILE_WORK_DIR Per Task
NxD inference defaults compile workdir to `/tmp/nxd_model/`. Multiple vllm servers starting in parallel **write to the same directory and corrupt each other**.
- **Fix**: `export BASE_COMPILE_WORK_DIR="/tmp/compile_<unique_task_name>/"` per task
- Source: `neuronx_distributed_inference/models/model_wrapper.py` uses `os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")`

### 4. Parallel Compilation is Wave-Based
Multiple servers can compile in parallel if each gets its own Neuron device and unique port:
- **Wave 1 (LNC=1)**: Each task on a separate device (1 core per device), up to 16 parallel
- **Wave 2 (LNC=2)**: Each task on a separate device (2 cores per device), up to 16 parallel
- **Cannot mix LNC=1 and LNC=2 in the same wave** — they need different core counts on the same device
- Poll health endpoints; benchmark each server as it becomes ready; then kill it to free the device

### 5. Device-to-Core Mapping
trn2.48xlarge: 16 devices (0-15), each with 4 physical cores.
- Device N -> cores [N*4, N*4+3]
- LNC=1: assign 1 core per device (e.g., core 0 on device 0)
- LNC=2: assign 2 consecutive cores on same device (e.g., cores 0-1 on device 0)
- Use `NEURON_RT_VISIBLE_CORES` to pin each server

### 6. No Parallel Model Loading on Same Device
Multiple servers loading models onto the **same** Neuron device simultaneously will fail:
```
nrt_allocate_neuron_cores: Logical Neuron Core(s) not available
```
**Fix**: Each task must use a **separate** Neuron device.

### 7. Compilation Cache Caches Failures
- Cache location: `/var/tmp/neuron-compile-cache/`
- Failed compilations get cached, causing instant `cached failed neff` errors on retry
- **Fix**: `rm -rf /var/tmp/neuron-compile-cache` before retrying failed tasks

### 8. Compilation Times
Neuron compilation (neuronx-cc) is very slow:
- Small configs (4K): ~3-5 min
- Medium configs (8K-16K): ~5-15 min
- Large configs (32K-64K): ~15-30 min

Server startup timeout should be **1800s (30 minutes)**.

### 9. Process Cleanup Must Be Thorough
vllm forks many child processes (EngineCore, neuronx-cc, walrus_driver). Killing the main process is NOT enough:
```bash
pkill -9 -f 'vllm.entrypoints'
pkill -9 -f 'neuronx-cc'
pkill -9 -f 'walrus_driver'
sleep 3
lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9
```

### 10. sequence_parallel_enabled
All serve.sh configs must include `"sequence_parallel_enabled": true` in `override_neuron_config`.

### 11. Bucket Configuration Rules
- `context_encoding_buckets`: Single bucket matching the input length (e.g., [4096] for 4K)
- `token_generation_buckets`: context_bucket + output_len (e.g., [4096 + 300] = [4396])
- `seq_len` / `max_model_len`: Same as token_generation_bucket value
- `bench INPUT_LEN`: Must satisfy `INPUT_LEN * 1.1 <= context_bucket` (due to `--random-range-ratio 0.1`)

### 12. Benchmark Parameters
- `max-concurrency` = batch_size
- `num-prompts`: Scale with batch size (e.g., BS=1 -> 8-16, BS=16 -> 64, BS=64 -> 256)
- `--request-rate inf` for max throughput testing
- `--random-range-ratio 0.1` to add slight variance to input lengths

### 13. Results Extraction from bench.log
- **ITL**: `Mean ITL (ms)` — inter-token latency
- **TTFT**: `Mean TTFT (ms)` / 1000 — time to first token in seconds
- **E2E**: `Benchmark duration (s)` — end to end
- **Req/min**: `Request throughput (req/s)` × 60
- **Tok/sec**: `Output token throughput (tok/s)`

## Folder Naming Convention
```
{model_short}_{input_len}_tp{tp}_lnc{lnc}_bs{bs}/
```

## Execution Strategy

### Wave-Based Parallel Compilation + Sequential Benchmarking

```
┌──────────────────────────────────────────┐
│  Wave N (same LNC):                      │
│                                          │
│  Start ALL servers in parallel            │
│    task0 -> device0 : port 8080          │
│    task1 -> device1 : port 8081          │
│    task2 -> device2 : port 8082          │
│    ...                                   │
│                                          │
│  Poll health endpoints every 10s         │
│    server ready?                         │
│    → Run bench.sh                        │
│    → Kill server, free device            │
│    → Continue polling others             │
│                                          │
│  After all done or timeout (1800s):      │
│    kill_all_servers()                    │
└──────────────────────────────────────────┘
```

### Resume Support
Track progress in `benchmark_progress.txt`:
```
DONE {task_name}
FAILED_SERVE {task_name}
FAILED_BENCH {task_name}
FAILED_TIMEOUT {task_name}
```
On restart, skip lines starting with `DONE`.

### Before Retrying Failed Tasks
1. Clear failed cache: `rm -rf /var/tmp/neuron-compile-cache`
2. Remove `FAILED_*` lines from `benchmark_progress.txt`
3. Kill all residual processes (see Rule 9)
4. Restart `run_all.sh`

## Quick Start on New Instance

```bash
# 1. Verify environment
neuron-ls                    # Check LNC and device count
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
python3 -c "import vllm; print(vllm.__version__)"

# 2. Clear old caches
rm -rf /var/tmp/neuron-compile-cache /tmp/compile_* /tmp/nxd_model

# 3. Download model
# huggingface-cli download <model> --local-dir <path>

# 4. Generate task folders (use generate_tasks.sh or let agent create them)
# 5. Run orchestrator
nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```

---

## Appendix A: serve.sh Template

```bash
#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES={CORES}
export BASE_COMPILE_WORK_DIR="/tmp/compile_{DIR_NAME}/"

python3 -m vllm.entrypoints.openai.api_server \
  --model="{MODEL_PATH}" \
  --tensor-parallel-size={TP} \
  --max-num-seqs={BS} \
  --max-model-len={SEQ_LEN} \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": {BS},
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": {LNC},
      "seq_len": {SEQ_LEN},
      "torch_dtype": "bfloat16",
      "tp_degree": {TP},
      "context_encoding_buckets": [{CTX_BUCKET}],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [{TKN_BUCKET}]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8080
```

## Appendix B: bench.sh Template

```bash
#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

MODEL="{MODEL_PATH}"
HOST="0.0.0.0"
PORT=8080
OUTPUT_LEN=300
INPUT_LEN={INPUT_LEN}
NUM_PROMPTS={NUM_PROMPTS}
SEED=42

echo "=========================================="
echo "Testing input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"
echo "=========================================="

vllm bench serve \
  --host ${HOST} \
  --port ${PORT} \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name random \
  --model ${MODEL} \
  --random-input-len ${INPUT_LEN} \
  --random-output-len ${OUTPUT_LEN} \
  --random-range-ratio 0.1 \
  --max-concurrency {MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --seed ${SEED}
```

## Appendix C: run_all.sh Orchestrator Template

```bash
#!/bin/bash
#
# Parallel Compilation + Sequential Benchmarking Orchestrator
#
set -uo pipefail

BASEDIR="/path/to/workdir"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=1800
BASE_PORT=8080

# Format: "dir_name:lnc:num_cores"
LNC1_TASKS=(
  # "task_dir_name:1:1"
)
LNC2_TASKS=(
  # "task_dir_name:2:2"
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

run_wave() {
  local wave_name=$1; shift
  local tasks=("$@")

  local pending_tasks=()
  for i in "${!tasks[@]}"; do
    local entry="${tasks[$i]}"
    local dir_name="${entry%%:*}"
    if is_done "$dir_name"; then
      log "  SKIP (done): $dir_name"
    else
      pending_tasks+=("$entry")
    fi
  done

  [[ ${#pending_tasks[@]} -eq 0 ]] && { log "  $wave_name: all done"; return; }

  log "========== $wave_name: ${#pending_tasks[@]} pending =========="

  local pids=() ports=() dirs=() device_idx=0

  for i in "${!pending_tasks[@]}"; do
    local entry="${pending_tasks[$i]}"
    local dir_name="${entry%%:*}"
    local rest="${entry#*:}"; local num_cores="${rest##*:}"
    local port=$((BASE_PORT + i))
    local task_dir="$BASEDIR/$dir_name"
    local dev=$device_idx; device_idx=$((device_idx + 1))
    local core_start=$((dev * 4))
    local visible_cores="$core_start"
    [[ $num_cores -gt 1 ]] && visible_cores="${core_start}-$((core_start + num_cores - 1))"

    log "  Starting $dir_name device=$dev cores=$visible_cores port=$port"
    sed -i "s/--port=[0-9]*/--port=${port}/" "$task_dir/serve.sh"
    cd "$task_dir"
    NEURON_RT_VISIBLE_CORES="$visible_cores" bash serve.sh > serve.log 2>&1 &
    pids+=($!); ports+=($port); dirs+=("$dir_name")
    cd "$BASEDIR"
  done

  log "  ${#pids[@]} servers starting (parallel compilation)..."

  local done_flags=(); for i in "${!pids[@]}"; do done_flags+=(0); done
  local elapsed=0

  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    local all_done=1
    for i in "${!pids[@]}"; do
      [[ ${done_flags[$i]} -ne 0 ]] && continue
      all_done=0
      local pid=${pids[$i]} port=${ports[$i]} dir_name=${dirs[$i]}
      local task_dir="$BASEDIR/$dir_name"

      if ! kill -0 $pid 2>/dev/null; then
        log "  FAILED_SERVE: $dir_name"; echo "FAILED_SERVE $dir_name" >> "$PROGRESS_FILE"
        done_flags[$i]=1; continue
      fi

      local code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
      if [[ "$code" == "200" ]]; then
        log "  READY: $dir_name (${elapsed}s)"
        sed -i "s/^PORT=.*/PORT=${port}/" "$task_dir/bench.sh"
        cd "$task_dir"; bash bench.sh > bench.log 2>&1; local rc=$?; cd "$BASEDIR"
        if [[ $rc -eq 0 ]]; then
          log "  DONE: $dir_name"; echo "DONE $dir_name" >> "$PROGRESS_FILE"
          grep -E 'throughput|TTFT|ITL' "$task_dir/bench.log" | head -8 | while read -r l; do log "    $l"; done
        else
          log "  FAILED_BENCH: $dir_name"; echo "FAILED_BENCH $dir_name" >> "$PROGRESS_FILE"
        fi
        kill -9 $pid 2>/dev/null || true; kill_server_on_port $port
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
      log "  TIMEOUT: ${dirs[$i]}"; echo "FAILED_TIMEOUT ${dirs[$i]}" >> "$PROGRESS_FILE"
      kill -9 ${pids[$i]} 2>/dev/null || true
    }
  done

  kill_all_servers

  for i in "${!pending_tasks[@]}"; do
    local dir_name="${pending_tasks[$i]%%:*}"; local td="$BASEDIR/$dir_name"
    sed -i "s/--port=[0-9]*/--port=8080/" "$td/serve.sh"
    sed -i "s/^PORT=.*/PORT=8080/" "$td/bench.sh"
  done
}

# ── Main ──
log "=== Parallel Compilation Benchmark ==="
run_wave "Wave 1 (LNC=1)" "${LNC1_TASKS[@]}"
run_wave "Wave 2 (LNC=2)" "${LNC2_TASKS[@]}"

log "=== ALL WAVES COMPLETE ==="
[[ -f "$PROGRESS_FILE" ]] && {
  log "  Done: $(grep -c '^DONE ' "$PROGRESS_FILE" || echo 0)"
  log "  Failed: $(grep -c '^FAILED' "$PROGRESS_FILE" || echo 0)"
  grep "^FAILED" "$PROGRESS_FILE" | while read -r l; do log "    $l"; done
}
```
