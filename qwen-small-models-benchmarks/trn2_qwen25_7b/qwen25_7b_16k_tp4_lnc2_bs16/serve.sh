#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES=2
export BASE_COMPILE_WORK_DIR="/tmp/compile_qwen25_7b_16k_tp4_lnc2_bs16/"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen2.5-7B-Instruct" \
  --tensor-parallel-size=4 \
  --max-num-seqs=16 \
  --max-model-len=16684 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 16,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 2,
      "seq_len": 16684,
      "torch_dtype": "bfloat16",
      "tp_degree": 4,
      "context_encoding_buckets": [16384],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [16684]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8082
