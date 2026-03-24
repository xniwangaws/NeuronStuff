#!/bin/bash
set -e

source /home/ubuntu/test-bytedance/vllm-env/bin/activate

HOST="0.0.0.0"
PORT=8080
OUTPUT_LEN=300
NUM_PROMPTS=256
CONCURRENCY=256
SEED=42
RESULT_DIR="/home/ubuntu/test-bytedance/bench_results"
MODEL_DIR="/home/ubuntu/models"

mkdir -p ${RESULT_DIR}

declare -A MODEL_MAX_LEN
MODEL_MAX_LEN=(
  ["Qwen3-0.6B"]=40960
  ["Qwen3-1.7B"]=40960
  ["Qwen2-1.5B"]=131072
  ["Qwen2.5-7B"]=131072
)

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
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
}

run_bench_for_model() {
  local MODEL_NAME=$1
  local MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
  local MAX_LEN=${MODEL_MAX_LEN[$MODEL_NAME]}

  echo ""
  echo "############################################################"
  echo "# Model: ${MODEL_NAME} (max_model_len=${MAX_LEN}) concurrency=${CONCURRENCY}"
  echo "############################################################"

  echo "Starting vllm serve for ${MODEL_NAME}..."
  SERVE_CMD="vllm serve ${MODEL_PATH} \
    --host ${HOST} \
    --port ${PORT} \
    --max-model-len ${MAX_LEN} \
    --gpu-memory-utilization 0.95"

  echo "CMD: ${SERVE_CMD}"
  ${SERVE_CMD} > "${RESULT_DIR}/${MODEL_NAME}_serve_c${CONCURRENCY}.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "Failed to start server for ${MODEL_NAME}"
    kill_server
    return 1
  fi

  for INPUT_LEN in "${INPUT_LENS[@]}"; do
    local TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
    if [ $TOTAL_LEN -gt $MAX_LEN ]; then
      echo ">>> SKIP input_len=${INPUT_LEN}: total ${TOTAL_LEN} > max_model_len ${MAX_LEN}"
      continue
    fi

    echo ""
    echo "=========================================="
    echo "Model: ${MODEL_NAME} | input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, concurrency=${CONCURRENCY}"
    echo "=========================================="

    RESULT_FILE="${RESULT_DIR}/${MODEL_NAME}_in${INPUT_LEN}_out${OUTPUT_LEN}_c${CONCURRENCY}"

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
      --max-concurrency ${CONCURRENCY} \
      --num-prompts ${NUM_PROMPTS} \
      --request-rate inf \
      --num-warmups 3 \
      --seed ${SEED} \
      --result-dir "${RESULT_DIR}" \
      --result-filename "$(basename ${RESULT_FILE}).json" \
      2>&1 | tee "${RESULT_FILE}.log"

    echo ""
  done

  kill_server
}

trap kill_server EXIT

for MODEL_NAME in "Qwen3-0.6B" "Qwen3-1.7B" "Qwen2-1.5B" "Qwen2.5-7B"; do
  run_bench_for_model "${MODEL_NAME}"
done

echo ""
echo "============================================"
echo "All c${CONCURRENCY} benchmarks complete!"
echo "Results saved in: ${RESULT_DIR}"
echo "============================================"
ls -la ${RESULT_DIR}/*c${CONCURRENCY}*
