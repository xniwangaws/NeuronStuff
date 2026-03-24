#!/bin/bash

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

export NEURON_RT_NUM_CORES=1

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/models/Qwen3-0.6B/" \
  --tensor-parallel-size=1 \
  --max-num-seqs=8 \
  --max-model-len=16684 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 8,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 1,
      "sequence_parallel_enabled": true,
      "seq_len": 16684,
      "torch_dtype": "bfloat16",
      "tp_degree": 1,
      "context_encoding_buckets": [4096, 8192, 16384],
      "token_generation_buckets": [4396, 8492, 16684]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8080
