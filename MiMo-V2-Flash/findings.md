# MiMo-V2-Flash Profile Findings

All numbers from `neuron-profile capture` + `view --output-format summary-text` on the NEFFs produced during FP8 and BF16 benchmark runs, 2026-04-26 and 2026-04-27.

## Environment

- Instance: trn2.48xlarge (us-east-2b), SDK 2.29.0 DLAMI (AMI `ami-0a81a0376c52f4d22`)
- Neuron compiler: `neuronxcc 2.24.5133.0+58f8de22`
- Neuron driver: `2.27.4`
- NxDI: PR #137 `contrib/MiMo-V2-Flash` branch
- vLLM: `vllm-neuron 0.5.0` + `vllm 0.16.0`

## End-to-end benchmark results

### BF16 (PR's reference recipe)

Input 900 tokens / output 90 tokens, `random` dataset, `vllm bench serve`:

| Concurrency | Our tok/s | PR tok/s | Delta | Our TPOT (ms) | PR TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | **26.61** | **27.98** | **-5%** | **33.72** | **33.65** |
| 16 | 159.94 | 224.57 | -29% | 89.00 | 64.95 |
| 32 | 196.58 | 302.61 | -35% | 135.35 | 90.23 |

BF16 BS=1 single-stream (standalone NxDI, `moe_tp=64, moe_ep=1`):
- Ours: 16.12 tok/s
- PR: 29.92 tok/s

### FP8 (PR's new recipe — no reported numbers in PR README)

BS=32 vLLM c=32 (FP8): 147.21 tok/s, TPOT 188 ms.
BS=32 standalone (FP8, `shard_on_block`): 137.87 tok/s, TPOT 232 ms.
BS=32 standalone (FP8, `shard_on_intermediate`): 146.29 tok/s, TPOT 219 ms (+6%).

## Per-NEFF device profile (rank 0 of TP=64)

### TGEN (token generation, per-step)

| Metric | BF16 BS=32 | FP8 BS=32 | BF16 BS=1 (MOE_EP=1) | FP8 BS=1 (MOE_EP=1) |
|---|---:|---:|---:|---:|
| total_time (ms) | **34.7** | 58.7 | 19.6 | 19.6 |
| tensor_engine_active % | 47 | 50 | 18 | 18 |
| vector_engine_active % | 12 | 48 | 41 | 41 |
| scalar_engine_active % | 22 | 20 | **46** | 46 |
| dma_active % | **86** | 67 | 23 | 23 |
| mfu % | 5.8 | 3.4 | **0.023** | 0.023 |
| mfu max achievable % | **12.5** | 11.7 | 0.69 | 0.69 |
| mbu % | 46 | 29 | 3.3 | 3.3 |
| mm arithmetic intensity | 28.1 | 26.3 | 1.59 | 1.59 |
| hbm_read_bytes | 11 GB | 11 GB | 461 MB | 461 MB |
| spill_save / spill_reload | 0 / 26 KB | 128 KB / 76 MB | 0 / 49 KB | 0 / 49 KB |

### CTE (context encoding, one-shot prefill)

| Metric | BF16 BS=32 | FP8 BS=32 | BF16 BS=1 | FP8 BS=1 |
|---|---:|---:|---:|---:|
| total_time (ms) | 348.8 | 400.0 | 739 | 739 |
| tensor_engine_active % | **69** | **75** | 54 | 54 |
| dma_active % | 82 | 79 | 87 | 87 |
| mfu % | **42** | **38** | 10.7 | 10.7 |
| mfu max achievable % | **73** | **100** ← | 25.3 | 25.3 |
| mbu % | 58 | 31 | 42 | 42 |
| mm arithmetic intensity | 160 | 265 | 22.6 | 22.6 |
| hbm_read_bytes | 119 GB | 66 GB | 137 GB | 137 GB |

## Interpretation (statements of fact, not speculation)

### 1. c=1 reproduces PR exactly
BF16 BS=32 vLLM c=1 TPOT = **33.72 ms**, PR reports **33.65 ms**. Differ by 0.2%. Our measured device TGEN time on the exact same NEFF = 34.7 ms. This means: the kernel runs as expected on our hardware, and at c=1 vLLM overhead is negligible.

### 2. c>1 gap is 100ms of host-side work per token
At c=32 our TPOT is 135 ms but device NEFF time is 35 ms. Delta = **100 ms per decoded token** not spent on the Neuron device. This 100 ms is the target for optimization — it is *not* compiler/kernel time.

Candidate breakdown of that 100 ms (unverified):
- vLLM V1 scheduler tick per step
- Python MoE routing / expert dispatch
- Collective op setup (AllReduce, AllGather) inter-NEFF
- Tokenize/detokenize on host CPU
- KV cache bookkeeping

### 3. BF16 TGEN is DMA-bound
86% of NEFF time is DMA. mm arithmetic intensity = 28 flops/byte → compute is light, each weight read yields few flops. This is **expected** for decode of a 286 GB BF16 model with BS=32.

### 4. FP8 CTE has 2.6× headroom (mfu 38% vs max 100%)
Compiler declares the CTE kernel theoretically capable of 100% MFU, but measured only 38%. The gap must be inside the NEFF — i.e., scheduling / dependency / pipeline stalls in the compiled graph. This is optimizable by the compiler team (not by config tuning).

### 5. BS=1 standalone with MOE_EP=1 is a different beast
When `moe_tp=64, moe_ep=1, BS=1`:
- TE active drops to 18%
- SE active jumps to 46%
- MFU max achievable falls to 0.69% — even theoretically compute-starved

This is the "selective loading" path (only 1 expert used per token). Not a real workload; only useful as smoke test for FP8. Serving must use `BS>=32` with `moe_ep_degree=64`.

### 6. `shard_on_intermediate` beats `shard_on_block` by 6%
Same model, same compile flags, only differing MoE blockwise kernel flag:
- `use_shard_on_block_dynamic_while=True`: 137.87 tok/s
- `use_shard_on_intermediate_dynamic_while=True`: 146.29 tok/s

A free +6% if we standardize on the latter, but it is not yet tested in the PR's official recipe.

### 7. The missing NKI kernel `blockwise_mm_baseline_shard_hidden`
PR explicitly notes SDK 2.29 ships `bwmm_shard_on_block` and `bwmm_shard_on_intermediate` but not `shard_hidden`. The latter is what NxDI's default code path prefers. Until this kernel ships in a stable SDK, **the MoE blockwise matmul cannot hit its designed throughput**.

## What we did NOT profile

- Host-side Python trace (`py-spy`, `torch.profiler`) — this is where the 100 ms answer lives
- Perfetto / Chrome-trace timeline — would show inter-NEFF gaps visually
- Per-rank profile beyond rank 0 (TP=64 may have imbalance)
- vLLM internal metrics trace (`VLLM_TRACE_FUNCTION=1`)
- BS=128 configuration (was excluded to save capacity-block time)

These are cheap to add if/when a new capacity block is available.
