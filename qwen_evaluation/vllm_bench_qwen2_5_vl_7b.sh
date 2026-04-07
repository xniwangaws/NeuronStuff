#!/bin/bash
# Benchmark Qwen2.5-VL-7B-Instruct on Neuron via vLLM
# Assumes server is already running on port 8080

MODEL_PATH="/home/ubuntu/models/Qwen2.5-VL-7B-Instruct"
HOST="0.0.0.0"
PORT=8080

# Text-only benchmark (no vision)
echo "=== Text-only benchmark ==="
vllm bench serve \
  --host ${HOST} \
  --port ${PORT} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --model ${MODEL_PATH} \
  --random-input-len 128 \
  --random-output-len 32 \
  --random-range-ratio 0.1 \
  --max-concurrency 1 \
  --num-prompts 32 \
  --request-rate inf \
  --seed 42

# Multimodal benchmark (100 images per request)
echo ""
echo "=== Multimodal benchmark (100 images) ==="
vllm bench serve \
  --host ${HOST} \
  --port ${PORT} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --model ${MODEL_PATH} \
  --random-input-len 128 \
  --random-output-len 32 \
  --random-range-ratio 0.1 \
  --random-mm-base-items-per-request 100 \
  --random-mm-bucket-config '{(640,320,1) : 1}' \
  --random-mm-limit-mm-per-prompt '{"image":100,"video":0}' \
  --max-concurrency 1 \
  --num-prompts 32 \
  --request-rate inf \
  --seed 42
