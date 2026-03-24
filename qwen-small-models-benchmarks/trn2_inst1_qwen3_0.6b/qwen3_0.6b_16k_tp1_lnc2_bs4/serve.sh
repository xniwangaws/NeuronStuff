#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES=2
export BASE_COMPILE_WORK_DIR="/tmp/compile_qwen3_0.6b_16k_tp1_lnc2_bs4/"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen3-0.6B/" \
  --tensor-parallel-size=1 \
  --max-num-seqs=4 \
  --max-model-len=16684 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 4,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "seq_len": 16684,
      "torch_dtype": "bfloat16",
      "tp_degree": 1,
      "context_encoding_buckets": [16384],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [16684]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8080
