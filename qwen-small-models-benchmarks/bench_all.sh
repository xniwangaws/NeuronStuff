#!/bin/bash
set -e

source /home/ubuntu/test-bytedance/vllm-env/bin/activate

HOST="0.0.0.0"
PORT=8080
OUTPUT_LEN=300
NUM_PROMPTS=64
SEED=42
RESULT_DIR="/home/ubuntu/test-bytedance/bench_results"
MODEL_DIR="/home/ubuntu/models"

mkdir -p ${RESULT_DIR}

# max_model_len set to cover 64K input + 300 output, capped by model's max_position_embeddings
# Qwen3-0.6B:   max_position_embeddings=40960
# Qwen3-1.7B:   max_position_embeddings=40960
# Qwen2-1.5B:   max_position_embeddings=131072
# Qwen2.5-7B:   max_position_embeddings=131072
declare -A MODEL_MAX_LEN
MODEL_MAX_LEN=(
  ["Qwen3-0.6B"]=40960
  ["Qwen3-1.7B"]=40960
  ["Qwen2-1.5B"]=131072
  ["Qwen2.5-7B"]=131072
)

# Input lengths to test (adjusted so max with ±10% range stays within power-of-2)
# 4K: 3700 → [3330, 4070] max < 4096
# 8K: 7400 → [6660, 8140] max < 8192
# 16K: 14800 → [13320, 16280] max < 16384
# 32K: 29700 → [26730, 32670] max < 32768
# 64K: 59500 → [53550, 65450] max < 65536
INPUT_LENS=(3700 7400 14800 29700 59500)

wait_for_server() {
  local max_wait=300
  local waited=0
  echo "Waiting for vllm server to be ready..."
  while [ $waited -lt $max_wait ]; do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
      echo "Server is ready! (took ${waited}s)"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "ERROR: Server did not become ready within ${max_wait}s"
  return 1
}

kill_server() {
  echo "Stopping vllm server..."
  if [ -n "$SERVER_PID" ]; then
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
  fi
  # Make sure no leftover vllm serve processes
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
}

run_bench_for_model() {
  local MODEL_NAME=$1
  local MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
  local MAX_LEN=${MODEL_MAX_LEN[$MODEL_NAME]}

  echo ""
  echo "############################################################"
  echo "# Model: ${MODEL_NAME} (max_model_len=${MAX_LEN})"
  echo "############################################################"

  # Start vllm serve in background
  # - gpu-memory-utilization=0.95: use most of GPU memory
  # - PagedAttention is the default, no special config needed
  # - No explicit batch size needed for GPU vllm
  echo "Starting vllm serve for ${MODEL_NAME}..."

  SERVE_CMD="vllm serve ${MODEL_PATH} \
    --host ${HOST} \
    --port ${PORT} \
    --max-model-len ${MAX_LEN} \
    --gpu-memory-utilization 0.95"

  echo "CMD: ${SERVE_CMD}"
  ${SERVE_CMD} > "${RESULT_DIR}/${MODEL_NAME}_serve.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "Failed to start server for ${MODEL_NAME}, check ${RESULT_DIR}/${MODEL_NAME}_serve.log"
    kill_server
    return 1
  fi

  # Run benchmarks for each input length
  for INPUT_LEN in "${INPUT_LENS[@]}"; do
    # Skip input lengths that exceed model's max context (with room for output)
    local TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
    if [ $TOTAL_LEN -gt $MAX_LEN ]; then
      echo ">>> SKIP input_len=${INPUT_LEN}: total ${TOTAL_LEN} > max_model_len ${MAX_LEN}"
      continue
    fi

    echo ""
    echo "=========================================="
    echo "Model: ${MODEL_NAME} | input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"
    echo "=========================================="

    RESULT_FILE="${RESULT_DIR}/${MODEL_NAME}_in${INPUT_LEN}_out${OUTPUT_LEN}"

    vllm bench serve \
      --host ${HOST} \
      --port ${PORT} \
      --backend vllm \
      --endpoint /v1/completions \
      --dataset-name random \
      --model ${MODEL_PATH} \
      --random-input-len ${INPUT_LEN} \
      --random-output-len ${OUTPUT_LEN} \
      --random-range-ratio 0.1 \
      --max-concurrency 8 \
      --num-prompts ${NUM_PROMPTS} \
      --request-rate inf \
      --num-warmups 3 \
      --seed ${SEED} \
      --result-dir "${RESULT_DIR}" \
      --result-filename "$(basename ${RESULT_FILE}).json" \
      2>&1 | tee "${RESULT_FILE}.log"

    echo ""
  done

  # Stop the server
  kill_server
}

# Trap to ensure cleanup on exit
trap kill_server EXIT

# Run benchmarks for all models
for MODEL_NAME in "Qwen3-0.6B" "Qwen3-1.7B" "Qwen2-1.5B" "Qwen2.5-7B"; do
  run_bench_for_model "${MODEL_NAME}"
done

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "Results saved in: ${RESULT_DIR}"
echo "============================================"
ls -la ${RESULT_DIR}/
