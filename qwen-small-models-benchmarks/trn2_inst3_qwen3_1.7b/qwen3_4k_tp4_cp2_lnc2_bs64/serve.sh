#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES=2
export BASE_COMPILE_WORK_DIR="/tmp/compile_qwen3_4k_tp4_cp2_lnc2_bs64/"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen3-1.7B" \
  --tensor-parallel-size=4 \
  --max-num-seqs=64 \
  --max-model-len=4396 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 64,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "cp_degree": 2,
      "seq_len": 4396,
      "torch_dtype": "bfloat16",
      "tp_degree": 4,
      "context_encoding_buckets": [4096],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [4396]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8080
