#!/bin/bash

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

vllm serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128 \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --port 8000 \
  --additional-config '{
    "override_neuron_config": {
      "enable_bucketing": false
    }
  }'
