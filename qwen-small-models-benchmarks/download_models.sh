#!/bin/bash
set -e

source /home/ubuntu/test-bytedance/vllm-env/bin/activate

MODEL_DIR="/home/ubuntu/models"
mkdir -p ${MODEL_DIR}

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen2-1.5B"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-VL-7B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename ${MODEL})
  DEST="${MODEL_DIR}/${MODEL_NAME}"
  if [ -d "${DEST}" ] && [ "$(ls -A ${DEST})" ]; then
    echo ">>> ${MODEL_NAME} already downloaded, skipping."
  else
    echo ">>> Downloading ${MODEL} ..."
    huggingface-cli download ${MODEL} --local-dir ${DEST}
  fi
done

echo ""
echo "All models downloaded to ${MODEL_DIR}:"
ls -1 ${MODEL_DIR}/
