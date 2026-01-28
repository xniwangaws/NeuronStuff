# MLP Kernel Benchmark

Fair performance comparison between nkilib MLP kernel and neuronxcc mlp_isa_kernel.

## Method

Uses `NEURON_FRAMEWORK_DEBUG=1` + `neuron-explorer capture` to measure actual kernel execution time.

## Usage

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Step 1: Generate NEFF files
python3 benchmark_mlp_step1_generate_neff.py

# Step 2: Profile NEFF files (must run separately, requires exclusive Neuron device access)
python3 benchmark_mlp_step2_profile.py
```

## Results

| Kernel | Config | Total (μs) | Tensor Eng (μs) | DMA (μs) |
|--------|--------|------------|-----------------|----------|
| neuronxcc | b=1, s=128, h=1024, i=512 | 39.36 | 14.66 | 19.80 |
| nkilib | b=1, s=128, h=1024, i=512 | 43.22 | 15.22 | 21.08 |
| neuronxcc | b=1, s=256, h=2048, i=1024 | 67.21 | 36.64 | 40.35 |
| nkilib | b=1, s=256, h=2048, i=1024 | 85.26 | 57.09 | 53.12 |

**Comparison:**
- Small config: neuronxcc is 1.10x faster than nkilib
- Large config: neuronxcc is 1.27x faster than nkilib

## Files

- `benchmark_mlp_step1_generate_neff.py` - Generate NEFF files
- `benchmark_mlp_step2_profile.py` - Profile NEFF files
- `benchmark_results.txt` - Detailed results and analysis

## Dependencies

- AWS Neuron SDK
- PyTorch NeuronX
- nkilib_standalone (must be installed)

## Key Parameters

`--profile-nth-exec=2`: Execute NEFF twice, only profile the second execution to avoid warmup overhead affecting results.
