#!/bin/bash
# Test NKI kernels on best 4K configs - v2 with fused_qkv fix
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
      log "[$dir] FAILED: server process died after ${elapsed}s"
      # Show error
      grep -E "Error|Exception|assert" "$task_dir/serve.log" 2>/dev/null | tail -3
      return 1
    fi
  done

  if [[ $elapsed -ge $TIMEOUT_SERVER ]]; then
    log "[$dir] FAILED: server timeout after ${TIMEOUT_SERVER}s"
    kill_server_tree $server_pid $port
    return 1
  fi

  log "[$dir] Running bench.sh..."
  bash bench.sh > bench.log 2>&1
  local bench_exit=$?

  if [[ $bench_exit -eq 0 ]]; then
    log "[$dir] Benchmark DONE"
  else
    log "[$dir] FAILED: bench exited with $bench_exit"
  fi

  kill_server_tree $server_pid $port
  return $bench_exit
}

# Create test configs
python3 << 'PYEOF'
import os, json

BASEDIR = "/home/ubuntu/test-bytedance"
MODEL = "/home/ubuntu/test-bytedance/Qwen2-1.5B/"

configs = [
    # Config 1: TP=1 LNC=1 BS=16 + kernels + seq_parallel (baseline: 425 tok/s)
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_v2",
        "bs": 16, "tp": 1, "lnc": 1, "cores": "0", "num_cores": 1, "port": 8130,
        "seq_parallel": True, "fuse_residual": False,
    },
    # Config 2: TP=1 LNC=1 BS=16 + kernels + fuse_residual (no seq_parallel) (baseline: 425)
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_fuse_v2",
        "bs": 16, "tp": 1, "lnc": 1, "cores": "1", "num_cores": 1, "port": 8131,
        "seq_parallel": False, "fuse_residual": True,
    },
    # Config 3: TP=2 LNC=2 BS=32 + kernels (baseline: 737 tok/s)
    {
        "name": "qwen2_1.5b_4k_bs32_tp2_lnc2_kernels_v2",
        "bs": 32, "tp": 2, "lnc": 2, "cores": "4-7", "num_cores": 4, "port": 8132,
        "seq_parallel": True, "fuse_residual": False,
    },
    # Config 4: TP=1 LNC=2 BS=16 + kernels (baseline: 447 tok/s)
    {
        "name": "qwen2_1.5b_4k_bs16_tp1_lnc2_kernels_v2",
        "bs": 16, "tp": 1, "lnc": 2, "cores": "2-3", "num_cores": 2, "port": 8133,
        "seq_parallel": True, "fuse_residual": False,
    },
]

for c in configs:
    task_dir = os.path.join(BASEDIR, c["name"])
    os.makedirs(task_dir, exist_ok=True)

    nc = {
        "async_mode": True,
        "batch_size": c["bs"],
        "ctx_batch_size": 1,
        "enable_bucketing": True,
        "is_continuous_batching": True,
        "logical_nc_config": c["lnc"],
        "seq_len": 4396,
        "torch_dtype": "bfloat16",
        "tp_degree": c["tp"],
        "context_encoding_buckets": [4096],
        "token_generation_buckets": [4396],
        # Required for QKV kernel
        "fused_qkv": True,
        # NKI Kernels
        "attn_kernel_enabled": True,
        "attn_tkg_nki_kernel_enabled": True,
        "mlp_kernel_enabled": True,
        "mlp_tkg_nki_kernel_enabled": True,
        "qkv_kernel_enabled": True,
        "k_cache_transposed": True,
    }

    if c["seq_parallel"]:
        nc["sequence_parallel_enabled"] = True

    if c["fuse_residual"]:
        nc["mlp_kernel_fuse_residual_add"] = True
        nc["qkv_kernel_fuse_residual_add"] = True

    nc_json = json.dumps(nc, indent=6)

    serve_sh = f"""#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="{c['cores']}"
export NEURON_RT_NUM_CORES={c['num_cores']}
export BASE_COMPILE_WORK_DIR="{task_dir}/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \\
  --model="{MODEL}" \\
  --tensor-parallel-size={c['tp']} \\
  --max-num-seqs={c['bs']} \\
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
  --max-concurrency {c['bs']} --num-prompts 64 \\
  --request-rate inf --seed 42
"""

    with open(os.path.join(task_dir, "serve.sh"), "w") as f:
        f.write(serve_sh)
    with open(os.path.join(task_dir, "bench.sh"), "w") as f:
        f.write(bench_sh)
    os.chmod(os.path.join(task_dir, "serve.sh"), 0o755)
    os.chmod(os.path.join(task_dir, "bench.sh"), 0o755)
    print(f"Created {c['name']} (cores={c['cores']}, port={c['port']})")

PYEOF

TASKS=(
  "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_v2"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_fuse_v2"
  "qwen2_1.5b_4k_bs32_tp2_lnc2_kernels_v2"
  "qwen2_1.5b_4k_bs16_tp1_lnc2_kernels_v2"
)

log ""
log "============================================"
log "NKI Kernels 4K v2 (with fused_qkv): ${#TASKS[@]} tasks"
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
log "NKI Kernels 4K v2 COMPLETE: Success=$((${#pids[@]} - failed)) Failed=$failed"
log "============================================"

# Print results comparison
echo ""
echo "=== 4K Kernel Results vs Baselines ==="
echo ""
printf "%-50s %10s %10s %10s %10s %8s\n" "Config" "ITL(ms)" "TTFT(s)" "Tok/s" "Baseline" "Delta"
echo "-----------------------------------------------------------------------------------------------------------"

baselines=(
  "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_v2|425.46"
  "qwen2_1.5b_4k_bs16_tp1_lnc1_kernels_fuse_v2|425.46"
  "qwen2_1.5b_4k_bs32_tp2_lnc2_kernels_v2|736.59"
  "qwen2_1.5b_4k_bs16_tp1_lnc2_kernels_v2|447.05"
)

for entry in "${baselines[@]}"; do
  dir="${entry%%|*}"
  base="${entry##*|}"
  bench="$BASEDIR/$dir/bench.log"
  if [[ -f "$bench" ]]; then
    toks=$(grep "Output token throughput" "$bench" | grep -oP '[\d.]+')
    itl=$(grep "Mean ITL" "$bench" | grep -oP '[\d.]+')
    ttft_ms=$(grep "Mean TTFT" "$bench" | grep -oP '[\d.]+')
    if [[ -n "$toks" ]]; then
      ttft_s=$(python3 -c "print(f'{float(\"$ttft_ms\")/1000:.2f}')")
      delta=$(python3 -c "print(f'{(float(\"$toks\")/float(\"$base\")-1)*100:+.1f}%')")
      printf "%-50s %10s %10s %10s %10s %8s\n" "$dir" "$itl" "$ttft_s" "$toks" "$base" "$delta"
    else
      printf "%-50s %10s\n" "$dir" "FAILED (check serve.log)"
    fi
  else
    printf "%-50s %10s\n" "$dir" "NO bench.log (server died)"
  fi
done
