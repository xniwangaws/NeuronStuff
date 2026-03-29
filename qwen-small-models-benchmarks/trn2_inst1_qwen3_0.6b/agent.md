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

### 2. NKI TopK 16384 Elements-Per-Partition Limit (NIO-2589)

**Problem**: `neuronx_distributed` forces NKI topk kernel on TRN2/TRN3. This kernel uses `nki.isa.max8` ISA instruction with a **hardware limit of 16384 elements per partition**. When `batch_size` is large (e.g., 4K@bs32+, 8K@bs32+), the token generation tensor exceeds this limit and compilation fails:
```
SyntaxError: max8 has 18992 elements per partition, but must be between 8 and 16384
```

**Root Cause File**: `<venv>/lib/python3.12/site-packages/neuronx_distributed/operators/topk.py`

**Root Cause Code** (in `get_topk_implementation()` function):
```python
# This code forces NKI topk on TRN2, triggering the 16384 limit:
can_use_nki_topk = dim in (-1, len(tensor.shape) - 1) and _is_nki_topk_available()
if can_use_nki_topk:
    stages = 1
    topk_implementation = nki_topk    # NKI version → max8 16384 limit
else:
    topk_implementation = TopK.apply   # XLA version, no limit
```
`_is_nki_topk_available()` returns True for TRN2/TRN3 hardware, so it always takes the NKI path.

**Fix** (NIO-2589 workaround): Replace the conditional block with:
```python
# NIO-2589: Disable forced NKI topk to avoid max8 16384 limit
topk_implementation = TopK.apply
```

**Key Points**:
- **No config option exists** to disable this. NeuronConfig has kernel flags for attention, QKV, MLP, RMSnorm — but NOT topk. Must patch source code.
- After patching, some configs may expose **HBM OOM** errors that were previously masked by the NKI error (NKI fails first during tracing, before HBM check).
- Performance impact: `TopK.apply` (XLA) may be slightly slower than NKI topk for small tensors, but it works for all sizes.
- Official fix planned in Neuron 2.28+ release.
- Tested on `neuronx-distributed` version `0.17.26814+4b18de63`.

### 2b. HBM Memory Limit (NCC_EVRF009)

After applying the NKI fix, large batch configs may hit HBM OOM:
```
[NCC_EVRF009] Size of total input and output tensors exceeds HBM limit of Trainium2.
Needed 33,771,172,352 bytes (31 GB) vs. available 25,769,803,776 bytes (24 GB).
```
- Each TRN2 logical NeuronCore has **24 GB HBM** (with LNC=2).
- KV cache size = `batch_size × num_kv_heads × seq_len × head_dim × 2 (K+V) × 2 bytes (bf16)`.
- **Fix**: Use TP=2 to shard KV cache across 2 cores, halving per-core memory.
- If TP=2 still OOMs, reduce batch_size or use higher TP degree.
- Mark these as `HBM_OOM` in results — this is a hard hardware limit for that TP/BS combination.

### 2c. MATCH_REPLACE8 Minimum Partition Size (TP=4 + small BS)

When using high TP degrees (e.g., TP=4) with small batch sizes (e.g., BS=2), the compiler may fail:
```
Assertion failure: (numInPerPartition >= 8)
MATCH_REPLACE8 Instruction expects at least 8 input elements per partition
```
- High TP shards tensors across many ranks. Combined with small BS, some intermediate tensor partitions drop below 8 elements.
- **Systematically affects BS=2 with TP=4** across all input lengths (4K/8K/16K/32K tested).
- BS=1, BS=4, BS=16, BS=64 all work fine with TP=4 — only BS=2 hits this.
- This is a neuronx-cc compiler hardware instruction constraint, not a software bug.
- **Rule**: Skip BS=2 when using TP=4. Mark as `COMPILER_LIMIT`.

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
- Small configs (4K): ~3-7 min
- Medium configs (8K-16K): ~5-15 min
- Large configs (32K): ~15-30 min
- **64K: >1 hour** — may exceed even 3600s timeout. Use separate retry script with extended timeout.

Server startup timeout should be **1800s (30 minutes)** for most configs, **3600s (1 hour)** for 64K.

### 9. Process Cleanup Must Be Thorough
vllm forks many child processes (EngineCore, neuronx-cc, walrus_driver). Killing the main process is NOT enough:
```bash
pkill -9 -f 'vllm.entrypoints'
pkill -9 -f 'neuronx-cc'
pkill -9 -f 'walrus_driver'
sleep 3
lsof /dev/neuron* 2>/dev/null | awk 'NR>1 && $1!="neuron-mo"{print $2}' | sort -u | xargs kill -9
```

### 10. TP=2 and TP=4 Configuration
When TP=1 hits HBM OOM or for better latency/throughput, increase TP degree:

**TP=2 with LNC=2:**
- `NEURON_RT_NUM_CORES=4` (2 logical cores × 2 physical/logical = 4 physical cores = 1 device)
- `tp_degree: 2`, `logical_nc_config: 2`
- Each task occupies 1 full Neuron device (4 cores)
- `NEURON_RT_VISIBLE_CORES`: e.g., `0-3` for device 0

**TP=4 with LNC=2:**
- `NEURON_RT_NUM_CORES=8` (4 logical cores × 2 physical/logical = 8 physical cores = 2 devices)
- `tp_degree: 4`, `logical_nc_config: 2`
- Each task occupies 2 full Neuron devices (8 cores)
- `NEURON_RT_VISIBLE_CORES`: must be 8 consecutive cores (e.g., `0-7` for devices 0-1)
- 16 devices → max 8 TP=4 tasks in parallel
- **Avoid BS=2 with TP=4** (MATCH_REPLACE8 partition limit, see Rule 2c)
- TP=4 dramatically helps 64K: compilation 14min vs >1hr, TTFT halved, throughput doubled vs TP=2

### 10b. Context Parallelism (CP) Constraints

NxD Inference supports `cp_degree` in `override_neuron_config`, but it has a hard constraint:
```
AssertionError: CP is with full TP decode is currently not supported with num_kv_heads > world_size
```
- `world_size = tp_degree` (for single-node inference)
- Requirement: **`num_key_value_heads <= tp_degree`**
- Example: Qwen3-0.6B has `num_key_value_heads=8`, so needs TP≥8 for CP. On a 0.6B model, TP=8 is wasteful.
- Example: Qwen3-235B-A22B uses TP=64, CP=16 (num_kv_heads=4 ≤ 64). Large models benefit more.
- Related configs: `cp_degree`, `strided_context_parallel_kernel_enabled: true`, `sequence_parallel_enabled: true`
- Config validation in `neuronx_distributed_inference/models/config.py` line 370: `tp_degree % cp_degree == 0`

**Rule**: Before attempting CP, check model's `num_key_value_heads` in `config.json`. If `num_kv_heads > tp_degree`, CP will fail immediately. For small models with many KV heads relative to their size, CP is not viable.

### 11. sequence_parallel_enabled
All serve.sh configs must include `"sequence_parallel_enabled": true` in `override_neuron_config`.

### 12. Bucket Configuration Rules
- `context_encoding_buckets`: Single bucket matching the input length (e.g., [4096] for 4K)
- `token_generation_buckets`: context_bucket + output_len (e.g., [4096 + 300] = [4396])
- `seq_len` / `max_model_len`: Same as token_generation_bucket value
- `bench INPUT_LEN`: Must satisfy `INPUT_LEN * 1.1 <= context_bucket` (due to `--random-range-ratio 0.1`)

### 13. Benchmark Parameters
- `max-concurrency` = batch_size
- `num-prompts`: Scale with batch size (e.g., BS=1 -> 8-16, BS=16 -> 64, BS=64 -> 256)
- `--request-rate inf` for max throughput testing
- `--random-range-ratio 0.1` to add slight variance to input lengths

### 14. Results Extraction from bench.log
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
