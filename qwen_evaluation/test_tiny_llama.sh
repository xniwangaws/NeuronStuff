#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --tensor-parallel-size 2 \
    --max-model-len 128 \
    --max-num-seqs 4 \
    --block-size 32 \
    --port 8000
