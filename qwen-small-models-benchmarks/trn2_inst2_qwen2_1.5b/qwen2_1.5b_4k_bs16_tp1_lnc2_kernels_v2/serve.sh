#!/bin/bash
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_VISIBLE_CORES="2-3"
export NEURON_RT_NUM_CORES=2
export BASE_COMPILE_WORK_DIR="/home/ubuntu/test-bytedance/qwen2_1.5b_4k_bs16_tp1_lnc2_kernels_v2/compile_cache"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen2-1.5B/" \
  --tensor-parallel-size=1 \
  --max-num-seqs=16 \
  --max-model-len=4396 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 16,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "seq_len": 4396,
      "torch_dtype": "bfloat16",
      "tp_degree": 1,
      "context_encoding_buckets": [
            4096
      ],
      "token_generation_buckets": [
            4396
      ],
      "fused_qkv": true,
      "attn_kernel_enabled": true,
      "attn_tkg_nki_kernel_enabled": true,
      "mlp_kernel_enabled": true,
      "mlp_tkg_nki_kernel_enabled": true,
      "qkv_kernel_enabled": true,
      "k_cache_transposed": true,
      "sequence_parallel_enabled": true
}
  }' \
  --no-enable-prefix-caching \
  --port=8133
