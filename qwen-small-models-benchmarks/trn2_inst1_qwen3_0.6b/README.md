# Qwen3-0.6B Benchmarks on AWS Trainium2 (trn2.48xlarge)

Benchmark scripts and results for Qwen3-0.6B on AWS Trainium2, comparing performance against NVIDIA H100.

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

cd qwen3_0.6b_4k_tp2_lnc2_bs32/
bash serve.sh
```

Wait for `Application startup complete` before proceeding.

### Step 2: Run Benchmark

In a separate terminal:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

cd qwen3_0.6b_4k_tp2_lnc2_bs32/
bash bench.sh
```

## Best Configuration Results

| Context | Best Config | vs H100 (%) | Trn2 tok/s | Trn2 tok/s per 2 Devices | H100 tok/s | TTFT Trn2 | TTFT H100 | Script Folder |
|---------|-------------|-------------|------------|--------------------------|------------|-----------|-----------|---------------|
| 4K | TP=2 BS=32 LNC=2 | 85% | 1,031.45 | 8,251.60 | 4,837.13 (c256) | 0.38s | 4.64s | [qwen3_0.6b_4k_tp2_lnc2_bs32](qwen3_0.6b_4k_tp2_lnc2_bs32/) |
| 8K | TP=2 BS=16 LNC=2 | 81% | 502.95 | 4,023.60 | 2,489.73 (c64) | 0.50s | 1.40s | [qwen3_0.6b_8k_tp2_lnc2_bs16](qwen3_0.6b_8k_tp2_lnc2_bs16/) |
| 16K | TP=2 BS=16 LNC=2 | 79% | 226.87 | 1,814.96 | 1,149.30 (c256) | 1.42s | 30.22s | [qwen3_0.6b_16k_tp2_lnc2_bs16](qwen3_0.6b_16k_tp2_lnc2_bs16/) |
| 32K | TP=2 BS=1 LNC=1 | 84% | 25.11 | 200.88 | 477.87 (c128) | 5.06s | 36.79s | [qwen3_0.6b_32k_tp2_lnc2_bs1](qwen3_0.6b_32k_tp2_lnc2_bs1/) |

> **vs H100 formula**: `(Trn2 tok/s × 16) / (H100 tok/s × LNC × TP) × 100%`
>
> **per 2 Devices formula**: `tok/s × 16 / TP`

### Key Observations

- **Qwen3-0.6B achieves 79-85% of H100 throughput** — the highest efficiency among all small models tested
- **4K TTFT is 12x faster than H100** (0.38s vs 4.64s)
- **16K TTFT is 21x faster than H100** (1.42s vs 30.22s)
- Trn2 consistently delivers significantly lower TTFT across all context lengths

## Configuration Naming Convention

```
qwen3_0.6b_{context}_tp{tp}_lnc{lnc}_bs{batch_size}
```

## Hardware Info

- **Instance**: trn2.48xlarge
- **Devices**: 16 Neuron devices, 64 physical NeuronCores
- **System LNC**: logical-neuroncore-config = 2 (32 logical cores)
