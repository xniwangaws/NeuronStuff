#!/bin/bash
# Serve Qwen2.5-VL-7B-Instruct on Neuron via vLLM

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Configuration - matching reference vllm_bench_qwen2_vl.sh (100 images)
MODEL_PATH="/home/ubuntu/models/Qwen2.5-VL-7B-Instruct"
NUM_OF_IMAGES=100
TEXT_SEQ_LENGTH=27648
VISION_SEQ_LENGTH=102400
DTYPE="bfloat16"
TP_DEGREE=4

# Text and Vision buckets
TEXT_BUCKETS="[2048, 15360, 27648]"
VISION_BUCKETS="[1, 50, 100]"

# Build the override neuron config JSON
OVERRIDE_NEURON_CONFIG=$(cat <<EOF
{
    "override_neuron_config": {
        "text_neuron_config": {
            "batch_size": 1,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "seq_len": ${TEXT_SEQ_LENGTH},
            "max_new_tokens": 64,
            "max_context_length": ${TEXT_SEQ_LENGTH},
            "torch_dtype": "${DTYPE}",
            "skip_sharding": false,
            "save_sharded_checkpoint": true,
            "tp_degree": ${TP_DEGREE},
            "cp_degree": 1,
            "world_size": ${TP_DEGREE},
            "context_encoding_buckets": ${TEXT_BUCKETS},
            "token_generation_buckets": ${TEXT_BUCKETS},
            "flash_decoding_enabled": false,
            "fused_qkv": true,
            "qkv_kernel_enabled": true,
            "mlp_kernel_enabled": false,
            "enable_bucketing": true,
            "sequence_parallel_enabled": true,
            "attn_kernel_enabled": true,
            "cc_pipeline_tiling_factor": 2,
            "attention_dtype": "${DTYPE}",
            "rpl_reduce_dtype": "${DTYPE}",
            "cast_type": "as-declared",
            "logical_neuron_cores": 2,
            "on_device_sampling_config": null,
            "async_mode": false
        },
        "vision_neuron_config": {
            "batch_size": 1,
            "seq_len": ${VISION_SEQ_LENGTH},
            "max_context_length": ${VISION_SEQ_LENGTH},
            "torch_dtype": "${DTYPE}",
            "skip_sharding": false,
            "save_sharded_checkpoint": true,
            "tp_degree": ${TP_DEGREE},
            "cp_degree": 1,
            "world_size": ${TP_DEGREE},
            "fused_qkv": true,
            "qkv_kernel_enabled": false,
            "attn_kernel_enabled": true,
            "mlp_kernel_enabled": true,
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

export NEURON_RT_NUM_CORES=${TP_DEGREE}

echo "Starting vLLM server for Qwen2.5-VL-7B-Instruct (TP=${TP_DEGREE}, ${NUM_OF_IMAGES} images)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --limit-mm-per-prompt "{\"image\": ${NUM_OF_IMAGES}}" \
    --tensor-parallel-size ${TP_DEGREE} \
    --max-model-len ${TEXT_SEQ_LENGTH} \
    --block-size 128 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "${OVERRIDE_NEURON_CONFIG}" \
    --port 8080
