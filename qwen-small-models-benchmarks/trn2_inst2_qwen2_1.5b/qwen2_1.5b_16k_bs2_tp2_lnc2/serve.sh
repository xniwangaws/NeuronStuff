#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="28-31"
export NEURON_RT_NUM_CORES=4
export BASE_COMPILE_WORK_DIR="/home/ubuntu/test-bytedance/qwen2_1.5b_16k_bs2_tp2_lnc2/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen2-1.5B/" \
  --tensor-parallel-size=2 \
  --max-num-seqs=2 \
  --max-model-len=16684 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 2,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "sequence_parallel_enabled": true,
      "seq_len": 16684,
      "torch_dtype": "bfloat16",
      "tp_degree": 2,
      "context_encoding_buckets": [16384],
      "token_generation_buckets": [16684]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8117
