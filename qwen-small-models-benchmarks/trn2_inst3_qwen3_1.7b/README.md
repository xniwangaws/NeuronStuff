# Qwen3-1.7B Benchmarks on AWS Trainium2 (trn2.48xlarge)

Benchmark scripts and results for Qwen3-1.7B on AWS Trainium2, comparing performance against NVIDIA H100.

## Prerequisites

- AWS Trainium2 instance (`trn2.48xlarge`)
- Neuron SDK 2.22+ with vLLM 0.13+
- Apply Neuron patches before testing: [neuron_patches.md](../neuron_patches.md)
  - Patches are applied to: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`

## How to Run

Each configuration folder contains two scripts:

- **`serve.sh`** — Starts the vLLM OpenAI-compatible server with the specified Neuron config
- **`bench.sh`** — Runs the vLLM benchmark against the running server

### Step 1: Start the vLLM Server

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

cd qwen3_4k_tp1_lnc2_bs16/
bash serve.sh
```

Wait for `Application startup complete` before proceeding.

### Step 2: Run Benchmark

In a separate terminal:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

cd qwen3_4k_tp1_lnc2_bs16/
bash bench.sh
```

## Best Configuration Results

| Context | Best Config | vs H100 (%) | Trn2 tok/s | Trn2 tok/s per 2 Devices | H100 tok/s | TTFT Trn2 | TTFT H100 | Script Folder |
|---------|-------------|-------------|------------|--------------------------|------------|-----------|-----------|---------------|
| 4K | TP=1 BS=16 LNC=2 | 86% | 422.98 | 3,383.84 | 3,946.36 (c256) | 0.53s | 6.31s | [qwen3_4k_tp1_lnc2_bs16](qwen3_4k_tp1_lnc2_bs16/) |
| 8K | TP=2 BS=16 LNC=2 | 70% | 388.38 | 1,553.52 | 2,210.66 (c64) | 0.80s | 1.72s | [qwen3_8k_tp2_lnc2_bs16](qwen3_8k_tp2_lnc2_bs16/) |
| 16K | TP=2 BS=16 LNC=2 | 92% | 222.25 | 889.00 | 969.53 (c64) | 1.88s | 6.43s | [qwen3_16k_tp2_lnc2_bs16](qwen3_16k_tp2_lnc2_bs16/) |
| 32K | TP=1 BS=2 LNC=2 | 83% | 42.85 | 342.80 | 410.80 (c256) | 3.13s | 89.89s | [qwen3_32k_tp1_lnc2_bs2](qwen3_32k_tp1_lnc2_bs2/) |

> **vs H100 formula**: `(Trn2 tok/s × 16) / (H100 tok/s × LNC × TP) × 100%`
>
> **per 2 Devices formula**: `tok/s × 16 / (TP × LNC)`

### Key Observations

- **16K context achieves 92% of H100** — the highest relative efficiency across all models and context lengths
- **32K TTFT is 29x faster than H100** (3.13s vs 89.89s)
- **4K TTFT is 12x faster than H100** (0.53s vs 6.31s)
- Trn2 shows strong scaling with LNC=2 across all configurations

## Configuration Naming Convention

```
qwen3_{context}_tp{tp}_lnc{lnc}_bs{batch_size}
```

## Hardware Info

- **Instance**: trn2.48xlarge
- **Devices**: 16 Neuron devices, 64 physical NeuronCores
- **System LNC**: logical-neuroncore-config = 2 (32 logical cores)
