#!/bin/bash

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

MODEL="/home/ubuntu/models/Qwen3-0.6B/"
HOST="0.0.0.0"
PORT=8080
OUTPUT_LEN=300
NUM_PROMPTS=64
SEED=42

for INPUT_LEN in 4000 8000 16000; do
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
    --max-concurrency 8 \
    --num-prompts ${NUM_PROMPTS} \
    --request-rate inf \
    --seed ${SEED}

  echo ""
done
