#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
MODEL="/home/ubuntu/test-bytedance/Qwen2-1.5B/"
vllm bench serve \
  --host 0.0.0.0 --port 8132 --backend vllm \
  --endpoint /v1/completions --dataset-name random \
  --model ${MODEL} --random-input-len 3700 \
  --random-output-len 300 --random-range-ratio 0.1 \
  --max-concurrency 32 --num-prompts 64 \
  --request-rate inf --seed 42
