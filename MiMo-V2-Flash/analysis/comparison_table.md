# Comparison Tables

## End-to-end throughput matrix

All configs use trn2.48xlarge, SDK 2.29, `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/`, vllm-neuron 0.5.0, input_len=900, output_len=90, random dataset.

| Experiment | Precision | BS | MOE_TP / EP | Runtime | c | Our tok/s | PR tok/s | Delta |
|---|---|---|---|---|---|---:|---:|---:|
| 1 | FP8 | 1 | 64/1 | NxDI smoke | 1 | 17.72 | — | — |
| 2 | FP8 | 32 | 1/64 | vLLM | 1 | 15.88 | — | — |
| 3 | FP8 | 32 | 1/64 | vLLM | 16 | 113.67 | — | — |
| 4 | FP8 | 32 | 1/64 | vLLM | 32 | 147.21 | — | — |
| 5 | FP8 | 32 | 1/64 | standalone (`shard_on_block`) | — | 137.87 | — | — |
| 6 | FP8 | 32 | 1/64 | standalone (`shard_on_intermediate`) | — | 146.29 | — | **+6% vs #5** |
| 7 | BF16 | 1 | 64/1 | NxDI smoke | 1 | 16.12 | 29.92 | **-46%** |
| **8** | **BF16** | **32** | **1/64** | **vLLM** | **1** | **26.61** | **27.98** | **-5%** ✅ |
| 9 | BF16 | 32 | 1/64 | vLLM | 16 | 159.94 | 224.57 | -29% |
| 10 | BF16 | 32 | 1/64 | vLLM | 32 | 196.58 | 302.61 | -35% |

## TPOT comparison at BF16 BS=32

| concurrency | our TPOT (ms) | PR TPOT (ms) | device NEFF time (ms) | host gap (ms) |
|---:|---:|---:|---:|---:|
| 1 | **33.72** | **33.65** | 34.67 | ≈0 ✅ |
| 16 | 89.00 | 64.95 | 34.67 | 54 |
| 32 | 135.35 | 90.23 | 34.67 | 100 |

**Key insight**: The 100 ms host gap at c=32 is where the optimization budget lives. It's not a compiler or kernel issue — it's Python/vLLM-side.

## Profile metric cross-product (BS=32)

| Metric | BF16 TGEN | FP8 TGEN | BF16 CTE | FP8 CTE |
|---|---:|---:|---:|---:|
| total_time (ms) | 34.7 | 58.7 | 348.8 | 400.0 |
| tensor_engine % | 47 | 50 | 69 | **75** |
| vector_engine % | 12 | 48 | 25 | 44 |
| scalar_engine % | 22 | 20 | 20 | 31 |
| dma % | **86** | 67 | 82 | 79 |
| mfu % | 5.8 | 3.4 | 42 | 38 |
| **mfu max achievable %** | **12.5** | 11.7 | 73 | **100** |
| mbu % | 46 | 29 | 58 | 31 |
| hbm read (GB) | 11 | 11 | 119 | 66 |
| mm intensity (flops/byte) | 28 | 26 | 160 | 265 |

Two headroom signals worth attacking:
- BF16 TGEN: mfu 5.8% vs max 12.5% (2×)
- FP8 CTE: mfu 38% vs max 100% (2.6×)
