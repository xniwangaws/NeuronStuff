# Qwen3-0.6B Benchmark Results

## Test Config
- Model: Qwen3-0.6B
- Output Len: 300
- Dtype: BF16
- HW: Trn2 (trn2.48xlarge, logical-neuroncore-config: 2)
- sequence_parallel_enabled: true
- NKI topk fix applied (NIO-2589): neuronx_distributed/operators/topk.py patched to disable forced NKI topk kernel

## Results

| HW | Config Ver. | Dtype | Input Len | BS | TP | LNC | ITL (ms) | TTFT (s) | E2E (s) | Req/min | Tok/sec | Status |
|----|-------------|-------|-----------|----|----|-----|----------|----------|---------|---------|---------|--------|
| Trn2 | - | BF16 | 4K | 16 | 1 | 1 | 124.93 | 0.96 | 158.80 | 24.18 | 121.02 | DONE |
| Trn2 | - | BF16 | 4K | 32 | 1 | 1 | 139.70 | 1.42 | 180.12 | 42.60 | 212.45 | DONE |
| Trn2 | - | BF16 | 4K | 64 | 1 | 1 | - | - | - | - | - | HBM_OOM |
| Trn2 | - | BF16 | 4K | 16 | 1 | 2 | 34.53 | 0.39 | 45.04 | 85.21 | 426.68 | DONE |
| Trn2 | - | BF16 | 4K | 32 | 1 | 2 | 67.23 | 0.68 | 86.66 | 88.58 | 441.53 | DONE |
| Trn2 | - | BF16 | 4K | 64 | 1 | 2 | - | - | - | - | - | HBM_OOM |
| Trn2 | - | BF16 | 4K | 16 | 2 | 2 | 17.46 | 0.24 | 22.91 | 167.40 | 838.80 | DONE |
| Trn2 | - | BF16 | 4K | 32 | 2 | 2 | 28.56 | 0.38 | 37.10 | 207.01 | 1031.45 | DONE |
| Trn2 | - | BF16 | 4K | 64 | 2 | 2 | 67.07 | 0.76 | 88.26 | 173.18 | 866.01 | DONE |
| Trn2 | - | BF16 | 4K | 1 | 4 | 2 | 6.98 | 0.07 | 17.56 | 27.33 | 139.41 | DONE |
| Trn2 | - | BF16 | 4K | 2 | 4 | 2 | - | - | - | - | - | COMPILER_LIMIT |
| Trn2 | - | BF16 | 4K | 16 | 4 | 2 | 14.43 | 0.15 | 18.84 | 203.82 | 1020.09 | DONE |
| Trn2 | - | BF16 | 4K | 64 | 4 | 2 | 36.83 | 0.42 | 48.54 | 315.00 | 1574.68 | DONE |
| Trn2 | - | BF16 | 8K | 8 | 1 | 1 | 108.57 | 1.02 | 271.85 | 14.12 | 70.69 | DONE |
| Trn2 | - | BF16 | 8K | 16 | 1 | 1 | 119.21 | 1.74 | 157.23 | 24.30 | 122.23 | DONE |
| Trn2 | - | BF16 | 8K | 32 | 1 | 1 | - | - | - | - | - | HBM_OOM |
| Trn2 | - | BF16 | 8K | 8 | 1 | 2 | 98.56 | 0.53 | 243.54 | 15.80 | 78.91 | DONE |
| Trn2 | - | BF16 | 8K | 16 | 1 | 2 | 87.64 | 0.90 | 112.58 | 34.07 | 170.71 | DONE |
| Trn2 | - | BF16 | 8K | 32 | 1 | 2 | - | - | - | - | - | HBM_OOM |
| Trn2 | - | BF16 | 8K | 8 | 2 | 2 | 19.90 | 0.30 | 50.65 | 75.82 | 379.44 | DONE |
| Trn2 | - | BF16 | 8K | 16 | 2 | 2 | 28.78 | 0.50 | 38.21 | 100.24 | 502.95 | DONE |
| Trn2 | - | BF16 | 8K | 32 | 2 | 2 | 68.33 | 0.85 | 88.58 | 86.75 | 432.00 | DONE |
| Trn2 | - | BF16 | 8K | 1 | 4 | 2 | 7.33 | 0.12 | 18.81 | 25.52 | 130.13 | DONE |
| Trn2 | - | BF16 | 8K | 2 | 4 | 2 | - | - | - | - | - | COMPILER_LIMIT |
| Trn2 | - | BF16 | 8K | 16 | 4 | 2 | 19.65 | 0.29 | 25.97 | 147.86 | 740.05 | DONE |
| Trn2 | - | BF16 | 8K | 64 | 4 | 2 | 86.24 | 0.85 | 114.12 | 134.43 | 672.61 | DONE |
| Trn2 | - | BF16 | 16K | 1 | 1 | 1 | 25.62 | 1.51 | 147.14 | 6.53 | 32.72 | DONE |
| Trn2 | - | BF16 | 16K | 2 | 1 | 1 | 55.78 | 1.81 | 149.62 | 6.42 | 32.18 | DONE |
| Trn2 | - | BF16 | 16K | 4 | 1 | 1 | 112.27 | 2.07 | 292.09 | 6.57 | 33.18 | DONE |
| Trn2 | - | BF16 | 16K | 1 | 1 | 2 | 10.07 | 1.02 | 63.01 | 15.24 | 76.42 | DONE |
| Trn2 | - | BF16 | 16K | 2 | 1 | 2 | 15.85 | 0.96 | 46.16 | 20.80 | 104.31 | DONE |
| Trn2 | - | BF16 | 16K | 4 | 1 | 2 | 41.49 | 1.09 | 109.99 | 17.46 | 88.11 | DONE |
| Trn2 | - | BF16 | 16K | 1 | 2 | 2 | 7.16 | 0.52 | 42.70 | 22.48 | 112.76 | DONE |
| Trn2 | - | BF16 | 16K | 2 | 2 | 2 | 16.03 | 0.55 | 43.28 | 22.18 | 111.26 | DONE |
| Trn2 | - | BF16 | 16K | 4 | 2 | 2 | 16.87 | 0.63 | 46.42 | 41.26 | 208.76 | DONE |
| Trn2 | - | BF16 | 16K | 16 | 2 | 2 | 63.11 | 1.42 | 84.71 | 45.33 | 226.87 | DONE |
| Trn2 | - | BF16 | 16K | 1 | 4 | 2 | 7.92 | 0.30 | 21.70 | 22.12 | 112.83 | DONE |
| Trn2 | - | BF16 | 16K | 2 | 4 | 2 | - | - | - | - | - | COMPILER_LIMIT |
| Trn2 | - | BF16 | 16K | 4 | 4 | 2 | 19.81 | 0.36 | 51.50 | 37.28 | 188.16 | DONE |
| Trn2 | - | BF16 | 16K | 16 | 4 | 2 | 34.89 | 0.76 | 46.82 | 82.01 | 410.42 | DONE |
| Trn2 | - | BF16 | 32K | 1 | 1 | 1 | 23.09 | 5.06 | 191.73 | 5.01 | 25.11 | DONE |
| Trn2 | - | BF16 | 32K | 2 | 1 | 1 | 110.28 | 5.96 | 314.51 | 3.05 | 15.31 | DONE |
| Trn2 | - | BF16 | 32K | 1 | 1 | 2 | 12.30 | 2.46 | 98.34 | 9.77 | 48.96 | DONE |
| Trn2 | - | BF16 | 32K | 2 | 1 | 2 | 33.44 | 2.71 | 102.48 | 9.37 | 46.98 | DONE |
| Trn2 | - | BF16 | 32K | 1 | 2 | 2 | 8.40 | 1.41 | 62.91 | 15.26 | 76.54 | DONE |
| Trn2 | - | BF16 | 32K | 2 | 2 | 2 | 30.68 | 1.54 | 86.69 | 10.79 | 55.54 | DONE |
| Trn2 | - | BF16 | 32K | 1 | 4 | 2 | 9.80 | 0.72 | 29.70 | 16.16 | 82.41 | DONE |
| Trn2 | - | BF16 | 32K | 2 | 4 | 2 | - | - | - | - | - | COMPILER_LIMIT |
| Trn2 | - | BF16 | 64K | 1 | 1 | 1 | - | - | - | - | - | COMPILE_TIMEOUT |
| Trn2 | - | BF16 | 64K | 1 | 1 | 2 | - | - | - | - | - | COMPILE_TIMEOUT |
| Trn2 | - | BF16 | 64K | 1 | 2 | 2 | 22.64 | 4.88 | 94.27 | 5.09 | 25.97 | DONE |
| Trn2 | - | BF16 | 64K | 1 | 4 | 2 | 12.31 | 2.23 | 47.88 | 10.04 | 51.13 | DONE |

## Summary
- **42 DONE** out of 52 total configs
- **4 HBM_OOM**: TP=1 KV cache exceeds 24GB per NeuronCore (4K@bs64, 8K@bs32)
- **4 COMPILER_LIMIT**: TP=4 BS=2 fails with MATCH_REPLACE8 < 8 elements per partition
- **2 COMPILE_TIMEOUT**: 64K TP=1 compilation exceeds 1-hour timeout

## Notes
- HBM_OOM: KV cache exceeds single NeuronCore HBM (24GB). Needed ~30-31GB. Fix: use TP≥2.
- COMPILER_LIMIT: neuronx-cc `MATCH_REPLACE8` instruction requires ≥8 elements per partition. TP=4 with BS=2 shards tensors too small. Affects all input lengths. Avoid BS=2 with TP=4.
- COMPILE_TIMEOUT: neuronx-cc compilation for 64K input with TP=1 exceeds 3600s. TP=2/4 compile successfully.
- NKI topk fix (NIO-2589): Patched neuronx_distributed/operators/topk.py to disable forced NKI topk kernel on TRN2. neuronx-distributed version: 0.17.26814+4b18de63.
- CP (Context Parallelism): Not viable for Qwen3-0.6B. Requires num_kv_heads ≤ tp_degree; model has num_kv_heads=8.
- Req/min = Request throughput (req/s) × 60
- Tok/sec = Output token throughput (tok/s)
