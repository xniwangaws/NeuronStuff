#!/bin/bash
# Test NKI kernels on 4K - v3: try individual kernel groups to find which work
set +e

BASEDIR="/home/ubuntu/test-bytedance"
RESULTS_LOG="$BASEDIR/benchmark_results.log"
TIMEOUT_SERVER=3600
MODEL="/home/ubuntu/test-bytedance/Qwen2-1.5B/"

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
      log "[$dir] FAILED: server died after ${elapsed}s"
      grep -E "Error|Exception|assert|Syntax" "$task_dir/serve.log" 2>/dev/null | tail -3
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: timeout"
    kill_server_tree $server_pid $port
    return 1
  fi

  log "[$dir] Running bench.sh..."
  bash bench.sh > bench.log 2>&1
  local bench_exit=$?

  if [[ $bench_exit -eq 0 ]]; then
    log "[$dir] Benchmark DONE"
  else
    log "[$dir] FAILED: bench exited $bench_exit"
  fi

  kill_server_tree $server_pid $port
  return $bench_exit
}

# Create test configs - each tests a different kernel combo
# All based on TP=1 LNC=1 BS=16 (baseline: 425 tok/s)
python3 << 'PYEOF'
import os, json

BASEDIR = "/home/ubuntu/test-bytedance"
MODEL = "/home/ubuntu/test-bytedance/Qwen2-1.5B/"

# Test individual kernel groups to find what works on Qwen2-1.5B
configs = [
    # A: Only attention kernels
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_only",
        "cores": "0", "num_cores": 1, "port": 8130,
        "extra": {
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": True,
        },
    },
    # B: Only MLP kernels
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_mlp_only",
        "cores": "1", "num_cores": 1, "port": 8131,
        "extra": {
            "mlp_kernel_enabled": True,
            "mlp_tkg_nki_kernel_enabled": True,
        },
    },
    # C: Only k_cache_transposed
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_kcache",
        "cores": "2", "num_cores": 1, "port": 8132,
        "extra": {
            "k_cache_transposed": True,
        },
    },
    # D: attn + mlp (no qkv, no fused_qkv)
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_mlp",
        "cores": "3", "num_cores": 1, "port": 8133,
        "extra": {
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "mlp_tkg_nki_kernel_enabled": True,
        },
    },
    # E: attn + mlp + kcache
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_mlp_kcache",
        "cores": "4", "num_cores": 1, "port": 8134,
        "extra": {
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "mlp_tkg_nki_kernel_enabled": True,
            "k_cache_transposed": True,
        },
    },
    # F: qkv + fused_qkv only (isolate the eps error)
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_qkv_only",
        "cores": "5", "num_cores": 1, "port": 8135,
        "extra": {
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
        },
    },
]

for c in configs:
    task_dir = os.path.join(BASEDIR, c["name"])
    os.makedirs(task_dir, exist_ok=True)

    nc = {
        "async_mode": True,
        "batch_size": 16,
        "ctx_batch_size": 1,
        "enable_bucketing": True,
        "is_continuous_batching": True,
        "logical_nc_config": 1,
        "sequence_parallel_enabled": True,
        "seq_len": 4396,
        "torch_dtype": "bfloat16",
        "tp_degree": 1,
        "context_encoding_buckets": [4096],
        "token_generation_buckets": [4396],
    }
    nc.update(c["extra"])

    nc_json = json.dumps(nc, indent=6)

    serve_sh = f"""#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="{c['cores']}"
export NEURON_RT_NUM_CORES={c['num_cores']}
export BASE_COMPILE_WORK_DIR="{task_dir}/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \\
  --model="{MODEL}" \\
  --tensor-parallel-size=1 \\
  --max-num-seqs=16 \\
  --max-model-len=4396 \\
  --additional-config='{{
    "override_neuron_config": {nc_json}
  }}' \\
  --no-enable-prefix-caching \\
  --port={c['port']}
"""

    bench_sh = f"""#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
MODEL="{MODEL}"
vllm bench serve \\
  --host 0.0.0.0 --port {c['port']} --backend vllm \\
  --endpoint /v1/completions --dataset-name random \\
  --model ${{MODEL}} --random-input-len 3700 \\
  --random-output-len 300 --random-range-ratio 0.1 \\
  --max-concurrency 16 --num-prompts 64 \\
  --request-rate inf --seed 42
"""

    with open(os.path.join(task_dir, "serve.sh"), "w") as f:
        f.write(serve_sh)
    with open(os.path.join(task_dir, "bench.sh"), "w") as f:
        f.write(bench_sh)
    os.chmod(os.path.join(task_dir, "serve.sh"), 0o755)
    os.chmod(os.path.join(task_dir, "bench.sh"), 0o755)
    print(f"Created {c['name']}")

PYEOF

TASKS=(
  "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_only"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_mlp_only"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_kcache"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_mlp"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_attn_mlp_kcache"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_qkv_only"
)

log ""
log "============================================"
log "NKI Kernels 4K v3 (individual kernel groups): ${#TASKS[@]} tasks"
log "============================================"

pids=()
for dir in "${TASKS[@]}"; do
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
log "NKI Kernels 4K v3 COMPLETE: Success=$((${#pids[@]} - failed)) Failed=$failed"
log "============================================"

echo ""
echo "=== Results (baseline: 425.46 tok/s) ==="
echo ""
printf "%-50s %10s %10s %10s %8s %s\n" "Config" "ITL(ms)" "TTFT(s)" "Tok/s" "Delta" "Status"
echo "-----------------------------------------------------------------------------------------------------------"

for dir in "${TASKS[@]}"; do
  bench="$BASEDIR/$dir/bench.log"
  if [[ -f "$bench" ]]; then
    toks=$(grep "Output token throughput" "$bench" | grep -oP '[\d.]+')
    itl=$(grep "Mean ITL" "$bench" | grep -oP '[\d.]+')
    ttft_ms=$(grep "Mean TTFT" "$bench" | grep -oP '[\d.]+')
    if [[ -n "$toks" ]]; then
      ttft_s=$(python3 -c "print(f'{float(\"$ttft_ms\")/1000:.2f}')")
      delta=$(python3 -c "print(f'{(float(\"$toks\")/425.46-1)*100:+.1f}%')")
      printf "%-50s %10s %10s %10s %8s %s\n" "$dir" "$itl" "$ttft_s" "$toks" "$delta" "OK"
    else
      printf "%-50s %s\n" "$dir" "BENCH FAILED"
    fi
  else
    # Check what error
    err=$(grep -oP "(?:SyntaxError|AssertionError|Error|Exception).*" "$BASEDIR/$dir/serve.log" 2>/dev/null | head -1)
    printf "%-50s %s\n" "$dir" "SERVER DIED: $err"
  fi
done
