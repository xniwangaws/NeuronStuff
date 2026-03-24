#!/bin/bash
# Test ctx_batch_size > 1 on best-performing TP=2 LNC=2 configs
# 8 tasks × 4 cores = 32 cores total
set +e

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
PROGRESS_FILE="$BASEDIR/benchmark_progress.txt"
TIMEOUT_SERVER=3600

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_LOG"
}

kill_server_tree() {
  local pid=$1
  local port=$2
  pkill -P $pid 2>/dev/null
  kill $pid 2>/dev/null
  sleep 2
  pkill -9 -P $pid 2>/dev/null
  kill -9 $pid 2>/dev/null
  lsof -ti :${port} 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 1
}

run_task() {
  local dir=$1
  local task_dir="$BASEDIR/$dir"
  local port
  port=$(grep -- '--port=' "$task_dir/serve.sh" | grep -oP '\d+')

  log "[$dir] Starting serve.sh (port $port)..."
  cd "$task_dir"
  bash serve.sh > serve.log 2>&1 &
  local server_pid=$!

  local elapsed=0
  while [[ $elapsed -lt $TIMEOUT_SERVER ]]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null | grep -q "200"; then
      log "[$dir] Server ready after ${elapsed}s"
      break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if ! kill -0 $server_pid 2>/dev/null; then
      log "[$dir] FAILED: server process died"
      echo "FAILED_SERVE $dir" >> "$PROGRESS_FILE"
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: server timeout after ${TIMEOUT_SERVER}s"
    kill_server_tree $server_pid $port
    echo "FAILED_TIMEOUT $dir" >> "$PROGRESS_FILE"
    return 1
  fi

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

  kill_server_tree $server_pid $port
  return $bench_exit
}

# Generate task folders
python3 << 'PYEOF'
import os, json

BASEDIR = "/home/ubuntu/test-bytedance"
MODEL = "/home/ubuntu/test-bytedance/Qwen2-1.5B/"

# (label, input_len, ctx_bucket, seq_len, bs, ctx_bs, cores, port)
tasks = [
    # 4K configs - baseline bs16=688, bs32=737 tok/s
    ("4k", 3700, 4096, 4396, 16, 2, "0-3", 8130),
    ("4k", 3700, 4096, 4396, 16, 4, "4-7", 8131),
    ("4k", 3700, 4096, 4396, 32, 2, "8-11", 8132),
    ("4k", 3700, 4096, 4396, 32, 4, "12-15", 8133),
    # 8K configs - baseline bs16=507, bs32=438 tok/s
    ("8k", 7400, 8192, 8492, 16, 2, "16-19", 8134),
    ("8k", 7400, 8192, 8492, 16, 4, "20-23", 8135),
    ("8k", 7400, 8192, 8492, 32, 2, "24-27", 8136),
    # 16K config - baseline bs4=161 tok/s
    ("16k", 14800, 16384, 16684, 4, 2, "28-31", 8137),
]

for label, input_len, ctx_bucket, seq_len, bs, ctx_bs, cores, port in tasks:
    folder = f"qwen2_1.5b_{label}_bs{bs}_tp2_lnc2_ctxbs{ctx_bs}"
    task_dir = os.path.join(BASEDIR, folder)
    os.makedirs(task_dir, exist_ok=True)

    serve_sh = f"""#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="{cores}"
export NEURON_RT_NUM_CORES=4
export BASE_COMPILE_WORK_DIR="{task_dir}/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \\
  --model="{MODEL}" \\
  --tensor-parallel-size=2 \\
  --max-num-seqs={bs} \\
  --max-model-len={seq_len} \\
  --additional-config='{{
    "override_neuron_config": {{
      "async_mode": true,
      "batch_size": {bs},
      "ctx_batch_size": {ctx_bs},
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "sequence_parallel_enabled": true,
      "seq_len": {seq_len},
      "torch_dtype": "bfloat16",
      "tp_degree": 2,
      "context_encoding_buckets": [{ctx_bucket}],
      "token_generation_buckets": [{seq_len}]
    }}
  }}' \\
  --no-enable-prefix-caching \\
  --port={port}
"""

    bench_sh = f"""#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
MODEL="{MODEL}"
vllm bench serve \\
  --host 0.0.0.0 --port {port} --backend vllm \\
  --endpoint /v1/completions --dataset-name random \\
  --model ${{MODEL}} --random-input-len {input_len} \\
  --random-output-len 300 --random-range-ratio 0.1 \\
  --max-concurrency {bs} --num-prompts 64 \\
  --request-rate inf --seed 42
"""

    with open(os.path.join(task_dir, "serve.sh"), "w") as f:
        f.write(serve_sh)
    with open(os.path.join(task_dir, "bench.sh"), "w") as f:
        f.write(bench_sh)
    os.chmod(os.path.join(task_dir, "serve.sh"), 0o755)
    os.chmod(os.path.join(task_dir, "bench.sh"), 0o755)
    print(f"Created {folder} (cores={cores}, port={port}, ctx_bs={ctx_bs})")

PYEOF

TASKS=(
  "qwen2_1.5b_4k_bs16_tp2_lnc2_ctxbs2"
  "qwen2_1.5b_4k_bs16_tp2_lnc2_ctxbs4"
  "qwen2_1.5b_4k_bs32_tp2_lnc2_ctxbs2"
  "qwen2_1.5b_4k_bs32_tp2_lnc2_ctxbs4"
  "qwen2_1.5b_8k_bs16_tp2_lnc2_ctxbs2"
  "qwen2_1.5b_8k_bs16_tp2_lnc2_ctxbs4"
  "qwen2_1.5b_8k_bs32_tp2_lnc2_ctxbs2"
  "qwen2_1.5b_16k_bs4_tp2_lnc2_ctxbs2"
)

log ""
log "============================================"
log "ctx_batch_size experiment: ${#TASKS[@]} tasks (4 cores each, 32 total)"
log "============================================"

pids=()
for dir in "${TASKS[@]}"; do
  if grep -q "^DONE $dir" "$PROGRESS_FILE" 2>/dev/null; then
    log "[$dir] SKIP (already done)"
    continue
  fi
  run_task "$dir" &
  pids+=($!)
done

log "Launched ${#pids[@]} tasks, waiting..."

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=$((failed + 1)); fi
done

log ""
log "============================================"
log "ctx_batch_size COMPLETE: Success=$((${#pids[@]} - failed)) Failed=$failed"
log "============================================"

# Print results summary
echo ""
echo "=== ctx_batch_size Results ==="
for dir in "${TASKS[@]}"; do
  bench="$BASEDIR/$dir/bench.log"
  if [[ -f "$bench" ]]; then
    toks=$(grep "Output token throughput" "$bench" | grep -oP '[\d.]+')
    itl=$(grep "Mean ITL" "$bench" | grep -oP '[\d.]+')
    ttft=$(grep "Mean TTFT" "$bench" | grep -oP '[\d.]+')
    if [[ -n "$toks" ]]; then
      echo "$dir: ITL=${itl}ms TTFT=${ttft}ms Tok/s=${toks}"
    else
      echo "$dir: FAILED (check serve.log)"
    fi
  else
    echo "$dir: NO bench.log"
  fi
done
