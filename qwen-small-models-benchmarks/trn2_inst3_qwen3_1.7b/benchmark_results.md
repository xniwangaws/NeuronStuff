# Qwen3-1.7B Benchmark Results on Trn2.48xlarge

## Environment
- **Instance**: trn2.48xlarge (16 Trainium2 chips, 128 physical NeuronCores, 64 logical NCs at LNC=2)
- **System LNC**: 2
- **Model**: Qwen3-1.7B (num_kv_heads=8, num_attention_heads=16)
- **Output Length**: 300 tokens
- **Framework**: vLLM v0.13.0 + neuronx-distributed-inference
- **Topk Patch**: NIO-2589 applied (TopK.apply instead of NKI topk)

## Results

### Wave 1: TP=1, LNC=1 (1 logical NC per task)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (s) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|---------------|---------------|--------|
| 4k | 1 | 0.23 | 13.8 | 68.79 | 927.52 | 0.383 | 13.31 | DONE |
| 4k | 2 | 0.25 | 15.0 | 76.13 | 1026.36 | 0.418 | 23.81 | DONE |
| 4k | 16 | 0.42 | 25.2 | 126.45 | 1706.60 | 1.225 | 117.78 | DONE |
| 4k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 8k | 1 | 0.13 | 7.8 | 38.29 | 995.69 | 0.865 | 23.33 | DONE |
| 8k | 2 | 0.17 | 10.2 | 50.89 | 1323.21 | 0.937 | 35.97 | DONE |
| 8k | 16 | 0.21 | 12.6 | 63.86 | 1659.96 | 2.526 | 236.28 | DONE |
| 8k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 16k | 1 | 0.09 | 5.4 | 25.76 | 1313.87 | 2.049 | 32.12 | DONE |
| 16k | 2 | 0.09 | 5.4 | 26.91 | 1372.62 | 2.348 | 66.18 | DONE |
| 16k | 16 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 16k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 1 | 0.07 | 4.2 | 20.85 | 2106.04 | 5.925 | 28.36 | DONE |
| 32k | 2 | 0.04 | 2.4 | 13.50 | 1363.29 | 7.412 | 123.00 | DONE |
| 64k | 1 | — | — | — | — | — | — | FAILED (Timeout) |
| 64k | 2 | — | — | — | — | — | — | FAILED (Timeout) |
### Wave 2a: TP=1, LNC=2 (2 logical NCs per task)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (s) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|---------------|---------------|--------|
| 4k | 1 | 0.36 | 21.6 | 106.61 | 1437.40 | 0.188 | 8.80 | DONE |
| 4k | 2 | 0.57 | 34.2 | 170.40 | 2297.44 | 0.221 | 10.50 | DONE |
| 4k | 16 | 1.41 | 84.6 | 422.98 | 5708.64 | 0.531 | 34.52 | DONE |
| 4k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 8k | 1 | 0.28 | 16.8 | 84.26 | 2190.89 | 0.432 | 10.47 | DONE |
| 8k | 2 | 0.36 | 21.6 | 109.40 | 2844.68 | 0.486 | 16.56 | DONE |
| 8k | 16 | 0.60 | 36.0 | 181.60 | 4720.74 | 1.252 | 78.05 | DONE |
| 8k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 16k | 1 | 0.20 | 12.0 | 58.85 | 3001.47 | 1.103 | 13.34 | DONE |
| 16k | 2 | 0.16 | 9.6 | 47.41 | 2418.37 | 1.236 | 38.00 | DONE |
| 16k | 16 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 16k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 1 | 0.13 | 7.8 | 40.57 | 4098.47 | 2.772 | 15.49 | DONE |
| 32k | 2 | 0.14 | 8.4 | 42.85 | 4328.27 | 3.128 | 36.09 | DONE |
| 64k | 1 | — | — | — | — | — | — | FAILED (Timeout) |
| 64k | 2 | — | — | — | — | — | — | FAILED (Timeout) |
### Wave 2b: TP=2, LNC=2 (4 logical NCs per task = 1 device)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (s) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|---------------|---------------|--------|
| 4k | 1 | 0.51 | 30.6 | 152.39 | 2054.65 | 0.150 | 6.09 | DONE |
| 4k | 2 | 0.73 | 43.8 | 220.19 | 2968.70 | 0.145 | 8.22 | DONE |
| 4k | 16 | 2.48 | 148.8 | 744.52 | 10048.20 | 0.349 | 19.43 | DONE |
| 4k | 64 | 2.50 | 150.0 | 748.98 | 10053.52 | 1.122 | 77.03 | DONE |
| 8k | 1 | 0.47 | 28.2 | 141.72 | 3685.12 | 0.266 | 6.19 | DONE |
| 8k | 2 | 0.62 | 37.2 | 188.02 | 4889.17 | 0.303 | 9.57 | DONE |
| 8k | 16 | 1.29 | 77.4 | 388.38 | 10096.19 | 0.798 | 37.09 | DONE |
| 8k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 16k | 1 | 0.31 | 18.6 | 93.90 | 4789.20 | 0.655 | 8.50 | DONE |
| 16k | 2 | 0.31 | 18.6 | 94.69 | 4829.58 | 0.756 | 18.49 | DONE |
| 16k | 16 | 0.74 | 44.4 | 222.25 | 11332.50 | 1.880 | 63.47 | DONE |
| 16k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 1 | 0.21 | 12.6 | 64.25 | 6489.86 | 1.720 | 9.88 | DONE |
| 32k | 2 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 64k | 1 | 0.10 | 6.0 | 28.74 | 5777.02 | 5.539 | 16.45 | DONE |
| 64k | 2 | 0.11 | 6.6 | 32.26 | 6484.53 | 6.225 | 41.19 | DONE |
### Wave 2c: TP=4, LNC=2 (8 logical NCs per task = 2 devices)

| Input | BS | Req/s | Req/min | Output tok/s | Total tok/s | Mean TTFT (s) | Mean ITL (ms) | Status |
|-------|----|-------|---------|-------------|-------------|---------------|---------------|--------|
| 4k | 1 | 0.74 | 44.4 | 220.96 | 2979.05 | 0.080 | 4.27 | DONE |
| 4k | 2 | 1.12 | 67.2 | 334.96 | 4516.07 | 0.088 | 5.44 | DONE |
| 4k | 16 | 3.83 | 229.8 | 1149.04 | 15507.67 | 0.203 | 12.58 | DONE |
| 4k | 64 | 4.82 | 289.2 | 1444.38 | 19387.77 | 0.588 | 39.90 | DONE |
| 8k | 1 | 0.64 | 38.4 | 191.69 | 4984.42 | 0.158 | 4.71 | DONE |
| 8k | 2 | 1.04 | 62.4 | 312.48 | 8125.38 | 0.169 | 5.80 | DONE |
| 8k | 16 | 2.51 | 150.6 | 752.28 | 19555.89 | 0.424 | 19.01 | DONE |
| 8k | 64 | 2.16 | 129.6 | 645.99 | 16698.16 | 1.223 | 89.44 | DONE |
| 16k | 1 | 0.49 | 29.4 | 147.74 | 7535.72 | 0.388 | 5.50 | DONE |
| 16k | 2 | 0.59 | 35.4 | 176.08 | 8981.09 | 0.404 | 9.95 | DONE |
| 16k | 16 | 1.30 | 78.0 | 391.26 | 19950.92 | 1.037 | 36.10 | DONE |
| 16k | 64 | — | — | — | — | — | — | FAILED (HBM OOM) |
| 32k | 1 | 0.33 | 19.8 | 99.95 | 10096.58 | 0.922 | 6.96 | DONE |
| 32k | 2 | 0.29 | 17.4 | 87.15 | 8803.70 | 1.028 | 19.43 | DONE |
| 64k | 1 | 0.18 | 10.8 | 53.78 | 10811.86 | 2.628 | 9.89 | DONE |
| 64k | 2 | 0.18 | 10.8 | 52.88 | 10630.74 | 3.065 | 27.54 | DONE |
## Summary

- **Total tasks**: 64
- **Completed**: 48
- **Failed**: 16
  - HBM OOM: 12 (batch_size too large for available HBM per NC)
  - Timeout: 4 (64K + LNC=1/TP=1 compilation >30min)

## Key Observations

### 1. LNC=2 vs LNC=1 (TP=1)
LNC=2 provides significant improvements across all metrics:
| Metric (4K, BS=1) | LNC=1 | LNC=2 | Improvement |
|---|---|---|---|
| Output tok/s | 68.79 | 106.61 | +55% |
| Mean TTFT | 383ms | 188ms | -51% |
| Mean ITL | 13.31ms | 8.80ms | -34% |

### 2. TP Scaling (LNC=2)
Higher TP reduces latency and increases throughput:
| Metric (4K, BS=1) | TP=1 | TP=2 | TP=4 |
|---|---|---|---|
| Output tok/s | 106.61 | 152.39 | 220.96 |
| Mean TTFT | 188ms | 150ms | 80ms |
| Mean ITL | 8.80ms | 6.09ms | 4.27ms |

### 3. Best Throughput Configs (LNC=2)
| Config | Output tok/s | Total tok/s | Req/min |
|---|---|---|---|
| 4K, TP=4, BS=64 | 1444.38 | 19387.77 | 289.2 |
| 4K, TP=4, BS=16 | 1149.04 | 15507.67 | 229.8 |
| 8K, TP=4, BS=16 | 752.28 | 19555.89 | 150.6 |
| 4K, TP=2, BS=64 | 748.98 | 10053.52 | 150.0 |
| 4K, TP=2, BS=16 | 744.52 | 10048.20 | 148.8 |

### 4. HBM OOM Failures
- TP=1: BS=64 always OOM (needs ~33GB vs 24GB/NC available)
- TP=1: 16K+BS=16 also OOM
- TP=2: BS=64 OOM at 8K+ input lengths
- TP=4: BS=64 OOM only at 16K (4K/8K fit)
- CP (context parallelism) not viable for Qwen3-1.7B: requires TP >= 16 due to num_kv_heads=8 constraint

### 5. 64K Input Length
- LNC=1/TP=1: Compilation timeout (>30 min)
- LNC=2/TP=1: Compilation timeout (>30 min)
- TP=2/TP=4 with LNC=2: Works fine
  - TP=4, BS=1: 53.78 tok/s, TTFT=2.63s
  - TP=4, BS=2: 52.88 tok/s, TTFT=3.06s
