#!/bin/bash

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 50
  }' | python3 -m json.tool
