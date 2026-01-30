#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "/home/ubuntu/HF-Qwen3-VL-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://hips.hearstapps.com/hmg-prod/images/roa080117fea-fordgt-01-1501826042.jpg?resize=640:*"
                    }
                }
            ]
        }
    ],
    "top_k": 1
}'
