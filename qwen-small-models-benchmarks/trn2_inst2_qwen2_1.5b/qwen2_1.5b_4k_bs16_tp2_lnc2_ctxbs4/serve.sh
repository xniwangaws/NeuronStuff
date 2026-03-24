#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="4-7"
export NEURON_RT_NUM_CORES=4
export BASE_COMPILE_WORK_DIR="/home/ubuntu/test-bytedance/qwen2_1.5b_4k_bs16_tp2_lnc2_ctxbs4/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen2-1.5B/" \
  --tensor-parallel-size=2 \
  --max-num-seqs=16 \
  --max-model-len=4396 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 16,
      "ctx_batch_size": 4,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "sequence_parallel_enabled": true,
      "seq_len": 4396,
      "torch_dtype": "bfloat16",
      "tp_degree": 2,
      "context_encoding_buckets": [4096],
      "token_generation_buckets": [4396]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8131
