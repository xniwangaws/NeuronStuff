# Qwen2.5-VL-7B-Instruct on AWS Neuron (Trn2)

Run Qwen2.5-VL-7B-Instruct on Trn2 via vLLM + NeuronX Distributed Inference.

## Benchmark Results

### Multimodal (100 images 640x320, 100 output tokens, 32 prompts)

Trn2 TP=4 uses 1 Neuron device. trn2.48xlarge has 16 devices, so 2 devices ≈ H100 comparison.

| Metric | Trn2 TP=4 c=1 (1 device) | Trn2 ×2 devices | H100 80GB c=2 | H100 80GB c=4 | H100 80GB c=8 | Trn2 ×2 vs H100 best |
|--------|--------------------------|-----------------|---------------|---------------|---------------|----------------------|
| Output tok/s | 20.67 | **41.34** | **53.59** | 49.41 | 52.99 | 77% |
| TTFT (median) | **3471 ms** | **3471 ms** | 2417 ms | 3540 ms | 13238 ms | **143%** |
| TPOT (median) | 14.45 ms | 14.45 ms | **12.25 ms** | 38.80 ms | 12.26 ms | 85% |
| Request throughput | 0.20 req/s | **0.40 req/s** | **0.54 req/s** | 0.49 req/s | 0.53 req/s | 74% |

### Tested Environment

- **Trn2**: trn2.48xlarge, us-east-2, NxDI 0.8.0, vllm-neuron 0.4.1, vLLM 0.13, bfloat16
- **H100**: p5.4xlarge (1x H100 80GB HBM3), ap-northeast-1, vLLM 0.19, PyTorch 2.10, bfloat16

## Architecture Differences from Qwen2-VL

Qwen2.5-VL has key differences from Qwen2-VL that prevent direct reuse of NxDI's `qwen2_vl` model:

| Component | Qwen2-VL | Qwen2.5-VL-7B |
|-----------|----------|----------------|
| Vision Norm | LayerNorm (weight+bias) | RMSNorm (weight only) |
| Vision MLP | GELU (fc1/fc2) | SwiGLU (gate_proj/up_proj/down_proj) |
| PatchMerger ln_q | LayerNorm | RMSNorm |
| PatchMerger output dim | vision hidden_size (1280) | text hidden_size (3584) |
| tie_word_embeddings | true | false (separate lm_head) |
| vocab_size | 151936 | 152064 |

## Files

```
qwen2_5_vl_7b/
├── __init__.py                      # Package exports
├── modeling_qwen2_5_vl.py           # Main model: config + forward + state dict conversion
├── modeling_qwen2_5_vl_text.py      # Text model: handles tie_word_embeddings=false
└── modeling_qwen2_5_vl_vision.py    # Vision model: RMSNorm + SwiGLU MLP + PatchMerger2.5

qwen2_5_vl_7b_setup.sh              # One-click setup: patches config, vllm_neuron, verifies imports
vllm_neuron_serve_qwen2_5_vl_7b.sh  # Serve script (TP=4, seq_len=27648, 100 images, bfloat16)
vllm_bench_qwen2_5_vl_7b.sh         # Benchmark script (text-only + multimodal 100 images)
run_qwen2_5_vl_7b.py                # Standalone compile + test (without vLLM)
test_qwen2_5_vl_7b.py               # Smoke test (imports + config assertions)
```

## Quick Start

### Prerequisites

- **Instance**: trn2.48xlarge (or trn2n.32xlarge with TP=4)
- **AMI**: vLLM 0.13 Neuron AMI (e.g., `ami-0cb3b3dd424204539` in us-east-2)
- **venv**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`

### Step 1: Clone and setup

```bash
cd /home/ubuntu
git clone https://github.com/xniwangaws/NeuronStuff.git
cd NeuronStuff/qwen_evaluation

# Run setup (downloads model, patches config + vllm_neuron, verifies imports)
bash qwen2_5_vl_7b_setup.sh
```

### Step 2: Serve

```bash
bash vllm_neuron_serve_qwen2_5_vl_7b.sh
# First run compiles (~5-10 min), subsequent runs use cached NEFFs
```

### Step 3: Test

```bash
# Generate a test image with shapes and text
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (320, 240), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
draw.ellipse([170, 50, 270, 150], fill='blue', outline='black')
draw.text((100, 180), 'Hello AWS!', fill='black')
img.save('/tmp/test.png')
"

# Image understanding test
python3 -c "
import requests, base64, json
with open('/tmp/test.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post('http://localhost:8080/v1/chat/completions', json={
    'model': '/home/ubuntu/models/Qwen2.5-VL-7B-Instruct',
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
        {'type': 'text', 'text': 'What do you see in this image? Describe the shapes and text.'}
    ]}],
    'max_tokens': 100
})
print(json.dumps(resp.json()['choices'][0]['message']['content'], indent=2))
"
# Expected: identifies red square, blue circle, and "Hello AWS!" text

# Benchmark
bash vllm_bench_qwen2_5_vl_7b.sh
```

## How It Works

The adapter does **not** copy files into the venv. Instead:

1. **`qwen2_5_vl_7b_setup.sh`** patches three things:
   - **Model `config.json`**: adds `mlp_ratio`, `embed_dim`, `in_channels`, `out_hidden_size` fields that NxDI expects but Qwen2.5-VL doesn't provide
   - **`vllm_neuron` model loader**: adds `Qwen2_5_VLForConditionalGeneration` to supported models and routes it to our custom `NeuronQwen2_5_VLForCausalLM` class via `sys.path`
   - **`vllm_neuron` model runner**: adds `qwen2_5_vl` to multimodal data processing (same path as `qwen2_vl`)

2. **At runtime**, vLLM loads the model and calls `_get_neuron_model_cls("Qwen2_5_VLForConditionalGeneration")` which returns our `NeuronQwen2_5_VLForCausalLM` from `qwen2_5_vl_7b/modeling_qwen2_5_vl.py`. This class:
   - Reuses text model from NxDI `qwen2_vl` (architecture identical)
   - Uses our custom vision model (`NeuronQwen2_5_VLVisionModel`) with RMSNorm + SwiGLU
   - Handles `tie_word_embeddings=false` in state dict conversion

## Serve Configuration

Default config in `vllm_neuron_serve_qwen2_5_vl_7b.sh`:

| Parameter | Value |
|-----------|-------|
| TP degree | 4 |
| dtype | bfloat16 |
| max_model_len | 27648 |
| Text buckets | [2048, 15360, 27648] |
| Vision buckets | [1, 50, 100] |
| NUM_OF_IMAGES | 100 |
| max_num_seqs | 1 |
| block_size | 128 |
| Port | 8080 |
