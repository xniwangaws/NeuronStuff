#!/bin/bash
# Generate TP=4 and TP=2 supplementary task folders for Qwen3-0.6B
BASEDIR="/home/ubuntu/test-bytedance"
MODEL_PATH="/home/ubuntu/test-bytedance/Qwen3-0.6B/"

declare -A CTX_BUCKETS=(
  [4k]=4096 [8k]=8192 [16k]=16384 [32k]=32768 [64k]=65536
)
declare -A TKN_BUCKETS=(
  [4k]=4396 [8k]=8492 [16k]=16684 [32k]=33068 [64k]=65836
)
declare -A INPUT_LENS=(
  [4k]=3700 [8k]=7400 [16k]=14800 [32k]=29700 [64k]=59500
)
declare -A NUM_PROMPTS_MAP=(
  [1]=8 [2]=16 [4]=32 [16]=64 [32]=128 [64]=256
)

gen_task() {
  local len=$1 tp=$2 bs=$3
  local lnc=2
  local cores=$((tp * lnc))
  local dir="qwen3_0.6b_${len}_tp${tp}_lnc${lnc}_bs${bs}"
  local task_dir="$BASEDIR/$dir"
  local ctx=${CTX_BUCKETS[$len]}
  local tkn=${TKN_BUCKETS[$len]}
  local input_len=${INPUT_LENS[$len]}
  local num_prompts=${NUM_PROMPTS_MAP[$bs]}

  if [[ -d "$task_dir" ]]; then
    echo "SKIP (exists): $dir"
    return
  fi

  mkdir -p "$task_dir"

  cat > "$task_dir/serve.sh" << SERVE_EOF
#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES=${cores}
export BASE_COMPILE_WORK_DIR="/tmp/compile_${dir}/"

python3 -m vllm.entrypoints.openai.api_server \\
  --model="${MODEL_PATH}" \\
  --tensor-parallel-size=${tp} \\
  --max-num-seqs=${bs} \\
  --max-model-len=${tkn} \\
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": ${bs},
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": ${lnc},
      "seq_len": ${tkn},
      "torch_dtype": "bfloat16",
      "tp_degree": ${tp},
      "context_encoding_buckets": [${ctx}],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [${tkn}]
    }
  }' \\
  --no-enable-prefix-caching \\
  --port=8080
SERVE_EOF

  cat > "$task_dir/bench.sh" << BENCH_EOF
#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

MODEL="${MODEL_PATH}"
HOST="0.0.0.0"
PORT=8080
OUTPUT_LEN=300
INPUT_LEN=${input_len}
NUM_PROMPTS=${num_prompts}
SEED=42

echo "=========================================="
echo "Testing input_len=\${INPUT_LEN}, output_len=\${OUTPUT_LEN}"
echo "=========================================="

vllm bench serve \\
  --host \${HOST} \\
  --port \${PORT} \\
  --backend vllm \\
  --endpoint /v1/completions \\
  --dataset-name random \\
  --model \${MODEL} \\
  --random-input-len \${INPUT_LEN} \\
  --random-output-len \${OUTPUT_LEN} \\
  --random-range-ratio 0.1 \\
  --max-concurrency ${bs} \\
  --num-prompts \${NUM_PROMPTS} \\
  --request-rate inf \\
  --seed \${SEED}
BENCH_EOF

  echo "CREATED: $dir (TP=$tp, BS=$bs, cores=$cores)"
}

# TP=2 supplementary
gen_task 16k 2 16

# TP=4 full sweep
for len in 4k 8k; do
  for bs in 1 2 16 64; do
    gen_task $len 4 $bs
  done
done
for bs in 1 2 4 16; do
  gen_task 16k 4 $bs
done
for bs in 1 2; do
  gen_task 32k 4 $bs
done
gen_task 64k 4 1

echo ""
echo "Done generating tasks."
