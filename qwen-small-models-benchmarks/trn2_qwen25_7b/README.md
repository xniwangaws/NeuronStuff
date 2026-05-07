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
| 4K | TP=4 BS=64 LNC=2 | 70% | 901.48 | 1,802.96 | 2,572.78 (c256) | 1.61s | 11.48s | [qwen25_7b_4k_tp4_lnc2_bs64](qwen25_7b_4k_tp4_lnc2_bs64/) |
| 8K | TP=4 BS=64 LNC=2 | 55% | 527.06 | 1,054.12 | 1,918.04 (c128) | 3.32s | 6.77s | [qwen25_7b_8k_tp4_lnc2_bs64](qwen25_7b_8k_tp4_lnc2_bs64/) |
| 16K | TP=4 BS=64 LNC=2 | 55% | 238.14 | 476.28 | 866.34 (c64) | 8.21s | 8.51s | [qwen25_7b_16k_tp4_lnc2_bs64](qwen25_7b_16k_tp4_lnc2_bs64/) |
| 32K | TP=4 BS=2 LNC=2 | 47% | 68.11 | 136.22 | 289.74 (c64) | 2.21s | 27.76s | [qwen25_7b_32k_tp4_lnc2_bs2](qwen25_7b_32k_tp4_lnc2_bs2/) |
| 64K | TP=4 BS=2 LNC=2 | 65% | 33.55 | 67.10 | 103.06 (c256) | 7.02s | 369.84s | [qwen25_7b_64k_tp4_lnc2_bs2](qwen25_7b_64k_tp4_lnc2_bs2/) |

> **vs H100 formula**: `(Trn2 tok/s × 16) / (H100 tok/s × LNC × TP) × 100%`
>
> **per 2 Devices formula**: `tok/s × 16 / (TP × LNC)`

### Key Observations

- **4K throughput reaches 901 tok/s** — highest absolute throughput among 7B-class models on Trn2
- **64K context achieves 65% of H100 throughput** (33.55 tok/s per TP group vs 103 tok/s)
- All configs use TP=4 LNC=2, optimal for 7B model size

### TTFT Comparison Note (fair high C)

The `TTFT H100` column above reports Mean TTFT at the peak-throughput concurrency (c64/c128/c256). For long contexts these values include significant queueing delay. Fair comparison using the lowest H100 concurrency reaching 90% of peak throughput:

| Ctx | H100 fair C | H100 TTFT | Trn2 TTFT | Comparison |
|-----|-------------|-----------|-----------|------------|
| 4K  | c=128 | 5.80s | 1.61s | Trn2 3.6x faster |
| 8K  | c=128 | 6.77s | 3.32s | Trn2 2.0x faster |
| 16K | c=32  | 2.47s | 8.21s | H100 3.3x faster |
| 32K | c=32  | 7.64s | 2.21s | Trn2 3.5x faster |
| 64K | c=16  | 10.11s | 7.02s | Trn2 1.4x faster |

## Configuration Naming Convention

```
qwen25_7b_{context}_tp{tp}_lnc{lnc}_bs{batch_size}
```

## Hardware Info

- **Instance**: trn2.48xlarge
- **Devices**: 16 Neuron devices, 64 physical NeuronCores
- **System LNC**: logical-neuroncore-config = 2 (32 logical cores)
