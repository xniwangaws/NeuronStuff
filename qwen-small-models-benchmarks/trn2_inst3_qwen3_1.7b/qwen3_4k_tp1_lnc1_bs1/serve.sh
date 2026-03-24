#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_RT_NUM_CORES=1
export BASE_COMPILE_WORK_DIR="/tmp/compile_qwen3_4k_tp1_lnc1_bs1/"

python3 -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/test-bytedance/Qwen3-1.7B" \
  --tensor-parallel-size=1 \
  --max-num-seqs=1 \
  --max-model-len=4396 \
  --additional-config='{
    "override_neuron_config": {
      "async_mode": true,
      "batch_size": 1,
      "ctx_batch_size": 1,
      "enable_bucketing": true,
      "is_continuous_batching": true,
      "logical_nc_config": 1,
      "seq_len": 4396,
      "torch_dtype": "bfloat16",
      "tp_degree": 1,
      "context_encoding_buckets": [4096],
      "sequence_parallel_enabled": true,
      "token_generation_buckets": [4396]
    }
  }' \
  --no-enable-prefix-caching \
  --port=8080
