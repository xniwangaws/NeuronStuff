# Qwen2-1.5B Benchmarks on AWS Trainium2 (trn2.48xlarge)

Benchmark scripts and results for Qwen2-1.5B on AWS Trainium2, comparing performance against NVIDIA H100.

## Prerequisites

- AWS Trainium2 instance (`trn2.48xlarge`)
- Neuron SDK 2.22+ with vLLM 0.13+
- Model downloaded to `/home/ubuntu/test-bytedance/Qwen2-1.5B/`

### Apply Neuron Patches

Before running any benchmarks, apply the required fixes to the Neuron virtual environment:

```bash
# Follow the patch instructions in:
# https://github.com/xniwangaws/NeuronStuff/blob/main/qwen-small-models-benchmarks/neuron_patches.md
# Patches are applied to: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
```

## How to Run

Each configuration folder (e.g., `qwen2_1.5b_4k_bs16_tp1_lnc1/`) contains two scripts:

- **`serve.sh`** — Starts the vLLM OpenAI-compatible server with the specified Neuron config
- **`bench.sh`** — Runs the vLLM benchmark against the running server

### Step 1: Start the vLLM Server

```bash
# Activate the Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Pick a configuration and start the server
cd qwen2_1.5b_4k_bs16_tp1_lnc1/
bash serve.sh
```

Wait for the server to finish compilation and show `Application startup complete` before proceeding.

### Step 2: Run Benchmark

In a separate terminal:

```bash
# Activate the benchmark virtual environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Run benchmark
cd qwen2_1.5b_4k_bs16_tp1_lnc1/
bash bench.sh
```

## Best Configuration Results

Performance comparison of the best Qwen2-1.5B configuration at each context length on Trn2 vs H100.

> **vs H100 formula**: `(Trn2 tok/s × 16) / (H100 tok/s × LNC × TP) × 100%`
>
> **per 2 Devices formula**: `tok/s × 16 / (TP × LNC)`

| Context | Best Config | vs H100 (%) | Trn2 tok/s per core | Trn2 tok/s per 2 Devices | H100 tok/s | TTFT Trn2 | TTFT H100 | Script Folder |
|---------|-------------|-------------|---------------------|--------------------------|------------|-----------|-----------|---------------|
| 4K | TP=1 BS=16 LNC=1 | 56% | 273.01 | 4,368.16 | 7,780.58 (c256) | 0.98s | 3.41s | [qwen2_1.5b_4k_bs16_tp1_lnc1](qwen2_1.5b_4k_bs16_tp1_lnc1/) |
| 8K | TP=1 BS=24 LNC=1 | 63% | 219.73 | 3,515.68 | 5,584.10 (c256) | 3.18s | 4.40s | [qwen2_1.5b_8k_bs24_tp1_lnc1](qwen2_1.5b_8k_bs24_tp1_lnc1/) |
| 16K | TP=2 BS=16 LNC=1 | 50% | 158.40 | 1,267.20 | 2,522.44 (c128) | 2.49s | 5.43s | [qwen2_1.5b_16k_bs16_tp2_lnc1](qwen2_1.5b_16k_bs16_tp2_lnc1/) |
| 32K | TP=1 BS=8 LNC=2 | 69% | 80.87 | 646.96 | 933.21 (c64) | 3.38s | 8.75s | [qwen2_1.5b_32k_bs8_tp1_lnc2](qwen2_1.5b_32k_bs8_tp1_lnc2/) |
| 64K | TP=2 BS=2 LNC=1 | 61% | 23.85 | 190.80 | 310.76 (c64) | 8.05s | 27.83s | [qwen2_1.5b_64k_bs2_tp2_lnc1](qwen2_1.5b_64k_bs2_tp2_lnc1/) |

### Key Observations

- **Throughput**: 32K at 69% of H100 is the best relative efficiency; 4K/8K/16K/64K range 50-63%
- **Trn2 uses a single NeuronCore (TP=1) for most configs**, maximizing per-device throughput

### ⚠️ TTFT Comparison Note (fair high C)

The `TTFT H100` column above reports Mean TTFT at the peak-throughput concurrency (c=64 / c=128 / c=256). For most rows these values include queueing delay because peak-throughput concurrency exceeds the H100 80GB KV capacity at long contexts. To compare TTFT fairly, use the lowest H100 concurrency that reaches ≥90% of peak throughput ("fair high C"):

| Ctx | H100 fair C | H100 TTFT | Trn2 TTFT | Comparison |
|-----|-------------|-----------|-----------|------------|
| 4K  | c=128 | 1.78s | 0.98s | Trn2 1.8× faster |
| 8K  | c=128 | 2.45s | 3.18s | H100 1.3× faster (parity) |
| 16K | c=128 | 5.43s | 2.49s | Trn2 2.2× faster |
| 32K | c=16  | 1.40s | 3.38s | H100 2.4× faster |
| 64K | c=16  | 3.93s | 8.05s | H100 2.0× faster |

**Summary**: Trn2 leads at 4K/16K, parity at 8K, H100 leads at 32K/64K. The previous claim "TTFT is consistently lower on Trn2" was based on H100 queue-saturated TTFT (c=256) and does not hold under fair high C comparison. Throughput ratios are unaffected.

## Configuration Naming Convention

```
qwen2_1.5b_{context}_{batch_size}_{tp}_{lnc}
```

- **context**: Max sequence length (4k, 8k, 16k, 32k, 64k)
- **batch_size**: `bs1`, `bs2`, `bs4`, `bs8`, `bs16`, `bs24`
- **tp**: Tensor parallel degree (`tp1`, `tp2`)
- **lnc**: Logical NeuronCore config (`lnc1`, `lnc2`)

## Hardware Info

- **Instance**: trn2.48xlarge
- **Devices**: 16 Neuron devices, 64 physical NeuronCores
- **System LNC**: logical-neuroncore-config = 2 (32 logical cores)
