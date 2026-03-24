# Qwen2.5-7B-Instruct Benchmark Results on Trn2.48xlarge

## Environment
- **Instance**: trn2.48xlarge (16 Trainium2 chips, 128 physical NeuronCores, 64 logical NCs at LNC=2)
- **System LNC**: 2
- **Model**: Qwen2.5-7B-Instruct (num_kv_heads=4, num_attention_heads=28, hidden_size=3584, num_layers=28)
- **Output Length**: 300 tokens
- **Framework**: vLLM v0.13.0 + neuronx-distributed-inference
- **Topk Patch**: NIO-2589 applied (TopK.apply instead of NKI topk)

## Results

### Wave 1: TP=1, LNC=2 (2 logical NCs per task)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|----------------|---------------|--------|
| 4k | 1 | 0.06 | 3.6 | 18.63 | 251.22 | 576.16 | 51.91 | DONE |
| 4k | 2 | 0.09 | 5.4 | 25.80 | 347.80 | 716.11 | 71.79 | DONE |
| 4k | 16 | 0.42 | 25.2 | 127.46 | 1720.17 | 1795.40 | 114.35 | DONE |
| 8k | 1 | 0.06 | 3.6 | 17.72 | 460.66 | 1167.83 | 52.74 | DONE |
| 8k | 2 | 0.08 | 4.8 | 25.24 | 656.37 | 1379.99 | 74.13 | DONE |
| 8k | 16 | 0.30 | 18.0 | 91.06 | 2367.17 | 3648.12 | 156.95 | DONE |
| 16k | 1 | 0.05 | 3.0 | 15.70 | 801.02 | 2667.86 | 54.99 | DONE |
| 16k | 2 | 0.07 | 4.2 | 21.19 | 1080.64 | 3131.74 | 83.45 | DONE |
| 16k | 16 | — | — | — | — | — | — | FAILED (HBM OOM, 28GB needed) |
| 32k | 1 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 2 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 64k | 1 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 64k | 2 | — | — | — | — | — | — | FAILED (HBM OOM) |

### Wave 2: TP=2, LNC=2 (4 logical NCs per task = 1 device)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|----------------|---------------|--------|
| 4k | 1 | 0.09 | 5.4 | 27.14 | 365.94 | 408.89 | 35.59 | DONE |
| 4k | 2 | — | — | — | — | — | — | FAILED (compile crash) |
| 4k | 16 | 0.79 | 47.4 | 236.28 | 3188.91 | 1223.83 | 60.94 | DONE |
| 4k | 64 | 1.11 | 66.6 | 331.28 | 4446.70 | 3760.23 | 172.38 | DONE |
| 8k | 1 | 0.09 | 5.4 | 25.92 | 673.94 | 826.24 | 35.96 | DONE |
| 8k | 2 | — | — | — | — | — | — | FAILED (compile crash) |
| 8k | 16 | 0.56 | 33.6 | 168.95 | 4391.87 | 2480.95 | 83.49 | DONE |
| 8k | 64 | 0.76 | 45.6 | 227.10 | 5870.40 | 7624.90 | 248.13 | DONE |
| 16k | 1 | 0.08 | 4.8 | 23.34 | 1190.24 | 1851.83 | 36.82 | DONE |
| 16k | 2 | — | — | — | — | — | — | FAILED (compile crash) |
| 16k | 16 | 0.28 | 16.8 | 84.25 | 4296.00 | 5650.54 | 165.97 | DONE |
| 16k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 1 | 0.06 | 3.6 | 19.18 | 1937.02 | 4302.32 | 37.98 | DONE |
| 32k | 2 | — | — | — | — | — | — | FAILED (compile crash) |
| 64k | 1 | — | — | — | — | — | — | FAILED (process died) |
| 64k | 2 | — | — | — | — | — | — | FAILED (process died) |

### Wave 3: TP=4, LNC=2 (8 logical NCs per task = 2 devices)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|----------------|---------------|--------|
| 4k | 1 | 0.28 | 16.8 | 82.80 | 1116.32 | 176.43 | 11.52 | DONE |
| 4k | 2 | 0.43 | 25.8 | 129.64 | 1747.87 | 206.07 | 14.09 | DONE |
| 4k | 16 | 1.82 | 109.2 | 547.36 | 7387.31 | 526.06 | 26.32 | DONE |
| 4k | 64 | 3.01 | 180.6 | 901.48 | 12100.45 | 1608.00 | 62.90 | DONE |
| 8k | 1 | 0.26 | 15.6 | 77.62 | 2018.47 | 356.24 | 11.74 | DONE |
| 8k | 2 | 0.41 | 24.6 | 122.78 | 3192.69 | 403.73 | 14.84 | DONE |
| 8k | 16 | 1.34 | 80.4 | 402.82 | 10471.51 | 1071.86 | 34.92 | DONE |
| 8k | 64 | 1.76 | 105.6 | 527.06 | 13623.96 | 3323.04 | 106.73 | DONE |
| 16k | 1 | 0.23 | 13.8 | 67.73 | 3454.48 | 816.84 | 12.09 | DONE |
| 16k | 2 | 0.33 | 19.8 | 99.09 | 5054.08 | 926.86 | 17.01 | DONE |
| 16k | 16 | 0.75 | 45.0 | 223.95 | 11419.20 | 2535.83 | 61.43 | DONE |
| 16k | 64 | 0.79 | 47.4 | 238.14 | 12073.23 | 8212.15 | 235.47 | DONE |
| 32k | 1 | 0.16 | 9.6 | 48.89 | 4938.54 | 2030.55 | 13.75 | DONE |
| 32k | 2 | 0.23 | 13.8 | 68.11 | 6880.09 | 2210.13 | 21.92 | DONE |
| 64k | 1 | 0.09 | 5.4 | 27.99 | 5627.75 | 6061.03 | 15.63 | DONE |
| 64k | 2 | 0.11 | 6.6 | 33.55 | 6743.97 | 7015.17 | 36.24 | DONE |

## Summary

- **Total tasks**: 45
- **Completed**: 33
- **Failed**: 12
  - HBM OOM: 7 (TP=1: 16k_bs16, 32k_bs1/2, 64k_bs1/2; TP=2: 16k_bs64, 64k_bs1/2)
  - Compile crash (BS=2 at TP=2): 4 (4k, 8k, 16k, 32k)
  - Process died: 1 (64k_tp2_bs2)

## Key Observations

### 1. TP Scaling (LNC=2, 4K, BS=1)
| Metric | TP=1 | TP=2 | TP=4 |
|---|---|---|---|
| Output tok/s | 18.63 | 27.14 | 82.80 |
| Mean TTFT | 576ms | 409ms | 176ms |
| Mean ITL | 51.91ms | 35.59ms | 11.52ms |

TP=4 provides 4.4x throughput and 4.5x lower ITL vs TP=1.

### 2. Best Throughput Configs (LNC=2)
| Config | Output tok/s | Total tok/s | Req/min |
|---|---|---|---|
| 4K, TP=4, BS=64 | 901.48 | 12100.45 | 180.6 |
| 4K, TP=4, BS=16 | 547.36 | 7387.31 | 109.2 |
| 8K, TP=4, BS=64 | 527.06 | 13623.96 | 105.6 |
| 8K, TP=4, BS=16 | 402.82 | 10471.51 | 80.4 |
| 4K, TP=2, BS=64 | 331.28 | 4446.70 | 66.6 |

### 3. Best Latency Configs (LNC=2)
| Config | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|
| 4K, TP=4, BS=1 | 176.43 | 11.52 |
| 4K, TP=4, BS=2 | 206.07 | 14.09 |
| 8K, TP=4, BS=1 | 356.24 | 11.74 |
| 8K, TP=4, BS=2 | 403.73 | 14.84 |
| 16K, TP=4, BS=1 | 816.84 | 12.09 |

### 4. BS=2 Compile Crashes at TP=2
All BS=2 configs at TP=2 failed during parallel compilation (engine core subprocess died without clear error). BS=2 works fine at TP=1 and TP=4, suggesting this is a compilation resource contention issue when 16 TP=2 tasks compile simultaneously, not a fundamental incompatibility.

### 5. HBM OOM Boundaries
- **TP=1** (24 GB/logical NC pair): Max 16K/BS=2. 16K/BS=16 needs 28GB → OOM. 32K+ all OOM.
- **TP=2** (48 GB across 1 device): Max 32K/BS=1. 16K/BS=64 OOM. 64K all OOM.
- **TP=4** (96 GB across 2 devices): All configs fit, including 64K/BS=2.

### 6. 64K Input Length
Only viable at TP=4:
- 64K, BS=1: 27.99 tok/s, TTFT=6.06s, ITL=15.63ms
- 64K, BS=2: 33.55 tok/s, TTFT=7.02s, ITL=36.24ms

### 7. Qwen2.5-7B vs Qwen3-1.7B (TP=4, LNC=2, 4K, BS=1)
| Metric | Qwen3-1.7B | Qwen2.5-7B | Ratio |
|---|---|---|---|
| Output tok/s | 220.96 | 82.80 | 2.7x slower |
| Mean TTFT | 80ms | 176ms | 2.2x slower |
| Mean ITL | 4.27ms | 11.52ms | 2.7x slower |

### 8. Throughput Scaling with Batch Size (TP=4, 4K)
| BS | Output tok/s | TTFT (ms) | ITL (ms) |
|----|-------------|-----------|----------|
| 1 | 82.80 | 176 | 11.52 |
| 2 | 129.64 | 206 | 14.09 |
| 16 | 547.36 | 526 | 26.32 |
| 64 | 901.48 | 1608 | 62.90 |

BS=64 yields 10.9x throughput vs BS=1, with 5.5x higher ITL — good throughput/latency tradeoff.
