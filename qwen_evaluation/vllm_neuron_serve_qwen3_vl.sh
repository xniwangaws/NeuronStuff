#!/bin/bash

# Configuration
NUM_OF_IMAGES=50
TEXT_SEQ_LENGTH=32768
VISION_SEQ_LENGTH=40000
DTYPE="bfloat16"

# Text and Vision buckets
TEXT_BUCKETS="[2048, 15360, 32768]"
VISION_BUCKETS="[6, 50]"

# Build the override neuron config JSON
OVERRIDE_NEURON_CONFIG=$(cat <<EOF
{
    "override_neuron_config": {
        "text_neuron_config": {
            "batch_size": 1,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "seq_len": ${TEXT_SEQ_LENGTH},
            "max_context_length": ${TEXT_SEQ_LENGTH},
            "torch_dtype": "${DTYPE}",
            "skip_sharding": false,
            "save_sharded_checkpoint": true,
            "tp_degree": 8,
            "world_size": 8,
            "context_encoding_buckets": ${TEXT_BUCKETS},
            "token_generation_buckets": ${TEXT_BUCKETS},
            "flash_decoding_enabled": false,
            "fused_qkv": true,
            "qkv_kernel_enabled": true,
            "mlp_kernel_enabled": true,
            "attn_kernel_enabled": true,
            "enable_bucketing": true,
            "sequence_parallel_enabled": false,
            "cc_pipeline_tiling_factor": 2,
            "attention_dtype": "${DTYPE}",
            "rpl_reduce_dtype": "${DTYPE}",
            "cast_type": "as-declared",
            "logical_neuron_cores": 2,
            "async_mode": false
        },
        "vision_neuron_config": {
            "batch_size": 1,
            "seq_len": ${VISION_SEQ_LENGTH},
            "max_context_length": ${VISION_SEQ_LENGTH},
            "image_width": 640,
            "image_height": 320,
            "torch_dtype": "${DTYPE}",
            "skip_sharding": false,
            "save_sharded_checkpoint": true,
            "tp_degree": 4,
            "world_size": 8,
            "fused_qkv": true,
            "qkv_kernel_enabled": false,
            "attn_kernel_enabled": true,
            "mlp_kernel_enabled": false,
            "enable_bucketing": true,
            "buckets": ${VISION_BUCKETS},
            "cc_pipeline_tiling_factor": 2,
            "rpl_reduce_dtype": "${DTYPE}",
            "cast_type": "as-declared",
            "async_mode": false,
            "logical_neuron_cores": 2
        }
    }
}
EOF
)

# Environment Variables
export NEURON_RT_EXEC_TIMEOUT=600
export NEURON_RT_INSPECT_ENABLE=0
export NEURON_RT_NUM_CORES=8
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_LOG_LEVEL=ERROR
export XLA_DENSE_GATHER_FACTOR=0
#export XLA_HLO_DEBUG=1
#export XLA_IR_DEBUG=1
export DISABLE_NUMERIC_CC_TOKEN=1
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=146800640

# Run vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/HF-Qwen3-VL-8B-Instruct \
    --limit-mm-per-prompt "{\"image\": ${NUM_OF_IMAGES}}" \
    --tensor-parallel-size 8 \
    --max-model-len ${TEXT_SEQ_LENGTH} \
    --block-size 128 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "${OVERRIDE_NEURON_CONFIG}"
