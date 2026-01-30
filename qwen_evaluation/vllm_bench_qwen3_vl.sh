vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --model /home/ubuntu/HF-Qwen3-VL-8B-Instruct \
  --random-input-len 128 \
  --random-output-len 32 \
  --random-range-ratio 0.1 \
  --random-mm-base-items-per-request 50 \
  --random-mm-bucket-config '{(640,320,1) : 1}' \
  --random-mm-limit-mm-per-prompt '{"image":50,"video":0}' \
  --max-concurrency 1 \
  --num-prompts 32 \
  --request-rate inf \
  --seed 42


