# Qwen2.5-7B Benchmarks on AWS Trainium2 (trn2.48xlarge)

Benchmark scripts and results for Qwen2.5-7B on AWS Trainium2, comparing performance against NVIDIA H100.

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

cd qwen25_7b_4k_tp4_lnc2_bs64/
bash serve.sh
```

Wait for `Application startup complete` before proceeding.

### Step 2: Run Benchmark

In a separate terminal:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

cd qwen25_7b_4k_tp4_lnc2_bs64/
bash bench.sh
```

## Best Configuration Results

| Context | Best Config | vs H100 (%) | Trn2 tok/s | Trn2 tok/s per 2 Devices | H100 tok/s | TTFT Trn2 | TTFT H100 | Script Folder |
|---------|-------------|-------------|------------|--------------------------|------------|-----------|-----------|---------------|
| 4K | TP=4 BS=64 LNC=2 | 70% | 901.48 | 3,605.92 | 2,572.78 (c256) | 1.61s | 11.48s | [qwen25_7b_4k_tp4_lnc2_bs64](qwen25_7b_4k_tp4_lnc2_bs64/) |
| 8K | TP=4 BS=64 LNC=2 | 55% | 527.06 | 2,108.24 | 1,918.04 (c128) | 3.32s | 6.77s | [qwen25_7b_8k_tp4_lnc2_bs64](qwen25_7b_8k_tp4_lnc2_bs64/) |
| 16K | TP=4 BS=64 LNC=2 | 55% | 238.14 | 952.56 | 866.34 (c64) | 8.21s | 8.51s | [qwen25_7b_16k_tp4_lnc2_bs64](qwen25_7b_16k_tp4_lnc2_bs64/) |
| 32K | TP=4 BS=2 LNC=2 | 47% | 68.11 | 272.44 | 289.74 (c64) | 2.21s | 27.76s | [qwen25_7b_32k_tp4_lnc2_bs2](qwen25_7b_32k_tp4_lnc2_bs2/) |
| 64K | TP=4 BS=2 LNC=2 | 65% | 33.55 | 134.20 | 103.06 (c256) | 7.02s | 369.84s | [qwen25_7b_64k_tp4_lnc2_bs2](qwen25_7b_64k_tp4_lnc2_bs2/) |

> **vs H100 formula**: `(Trn2 tok/s × 16) / (H100 tok/s × LNC × TP) × 100%`
>
> **per 2 Devices formula**: `tok/s × 16 / TP`

### Key Observations

- **4K context TTFT is 7x faster than H100** (1.61s vs 11.48s)
- **64K context achieves 65% of H100** with TTFT 53x faster (7.02s vs 369.84s)
- **4K throughput reaches 901 tok/s** — highest absolute throughput among 7B-class models on Trn2
- All configs use TP=4 LNC=2, optimal for 7B model size

## Configuration Naming Convention

```
qwen25_7b_{context}_tp{tp}_lnc{lnc}_bs{batch_size}
```

## Hardware Info

- **Instance**: trn2.48xlarge
- **Devices**: 16 Neuron devices, 64 physical NeuronCores
- **System LNC**: logical-neuroncore-config = 2 (32 logical cores)
