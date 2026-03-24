#!/bin/bash

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

MODEL="/home/ubuntu/test-bytedance/Qwen2-1.5B/"
HOST="0.0.0.0"
PORT=8102
OUTPUT_LEN=300
NUM_PROMPTS=64
SEED=42

echo "=========================================="
echo "Testing input_len=7400, output_len=${OUTPUT_LEN}"
echo "=========================================="

vllm bench serve \
  --host ${HOST} \
  --port ${PORT} \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name random \
  --model ${MODEL} \
  --random-input-len 7400 \
  --random-output-len ${OUTPUT_LEN} \
  --random-range-ratio 0.1 \
  --max-concurrency 32 \
  --num-prompts ${NUM_PROMPTS} \
  --request-rate inf \
  --seed ${SEED}
