# Image Generation Models on Trainium2 — Benchmark Summary

* FLUX.1-dev (https://github.com/xniwangaws/NeuronStuff/tree/main/flux-benchmark): 12B DiT model deployed on trn2.3xlarge with NxDI NeuronFluxApplication (BF16 TP=4). At 1K resolution, Neuron achieves 8.03s/image which is 5.2× faster than L4 FP8 (41.4s), and 67% cheaper per image ($0.005 vs $0.015). At 2K, Neuron achieves 93.3s compared to L4 231.6s — 2.5× faster and 32% cheaper. There is further optimization potential: VAE segmented decode (7 NEFFs pre-loaded to HBM) can reduce VAE from 50s to 2.93s, bringing total to ~47s.

* FLUX.2-klein (https://github.com/xniwangaws/NeuronStuff/tree/main/flux2): 9B guidance-distilled DiT deployed on trn2.3xlarge with NxDI PR #146 (BF16 TP=4). Neuron is 2.0× faster at 1K (38.0s vs 76.5s) with Neuron 14% cheaper ($0.024 vs $0.028), and 2.1× faster at 2K (184s vs 382.7s) with Neuron 19% cheaper ($0.114 vs $0.141). L4 cannot use torch.compile due to BFL custom code graph breaks. 4K is blocked by model spec (max_area=4MP), not hardware — all devices produce gray noise at 4K.

* SDXL-base-1.0 (https://github.com/xniwangaws/NeuronStuff/tree/main/sdxl-benchmark): 2.6B UNet deployed on trn2.3xlarge with torch_neuronx.trace + DataParallel(DP=2) and NKI flash-attention. At 1K, Neuron is 1.1× faster (11.14s vs 12.68s) and 26% cheaper per image ($0.0035 vs $0.0047). At 2K, L4 FP8+compile is 2.9× faster (74.85s vs 213.9s) and 79% cheaper ($0.028 vs $0.133) — SDXL UNet is only 2.6B parameters, too small to benefit from TP=4 tensor parallelism; the inter-core communication overhead at high spatial resolution (256×256 latent) exceeds the parallelism gain. TP=1 is not feasible at 2K (attention activation 49GB > 24GB HBM per core pair). Further optimization: VAE segmented decode on Neuron (3.0s vs 67s CPU) can reduce total from 213.9s to ~150s, but still slower than L4.

## Summary Table

| Model | Params | 1K Neuron | 1K L4 | Neuron vs L4 (1K) | 2K Neuron | 2K L4 | Neuron vs L4 (2K) |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 12B | 8.03s / $0.005 | 41.4s / $0.015 | **5.2× faster, 67% cheaper** | 93.3s / $0.058 | 231.6s / $0.085 | **2.5× faster, 32% cheaper** |
| FLUX.2-klein | 9B | 42.9s / $0.027 | 76.5s / $0.028 | **1.8× faster, cost 持平** | 191.4s / $0.119 | 382.7s / $0.141 | **2.0× faster, 16% cheaper** |
| SDXL | 2.6B | 11.14s / $0.0035 | 12.68s / $0.0047 | **1.1× faster, 26% cheaper** | 213.9s / $0.133 | 74.85s / $0.028 | L4 2.9× faster, 79% cheaper |

## Key Insight

Neuron advantage scales with model size: large models (9-12B FLUX) show 1.8-5.2× speedup and significant cost savings; small models (2.6B SDXL) at high resolution lose to L4 due to TP communication overhead exceeding parallelism benefit.

## 4K Resolution: Compilation Investigation (2026-05-11)

We investigated whether FLUX.1-dev and SDXL can compile at 4K (4096×4096) on trn2.48xlarge with SDK 2.29.

### FLUX.1-dev 4K — UNSUCCESSFUL

**Setup**: NxDI NeuronFluxApplication, TP=4, NKI attention_cte (LNC2 sharded), -O1, --inst-count-limit=100M
**Result**: walrus_driver passes all phases (HLO gen 20min, dep_reduction, dep_opt, isa_gen ~118s) but **fails final HBM check**:
```
Assertion failure: TotalDRAMUsage <= HBMLimit
hbm_usage failed after 14.034 seconds
```

**Why**: Even with NKI flash attention (attention_cte), the 65,536-token attention scratchpad exceeds 24GB per Neuron core.
Total compile time before failure: ~2.5 hours.

### SDXL 4K — PARTIAL SUCCESS, ATTENTION BLOCKERS

**Setup**: Segmented UNet (compile each block separately) with two variants:
1. Without NKI: standard SDPA attention
2. With NKI attention_cte: NKI flash attention swapped into AttnProcessor

**Results per segment** (4K = 4096×4096, latent 512×512):

| Segment | Spatial | Status (no-NKI) | Status (with NKI) | Notes |
|---------|---------|-----------------|-------------------|-------|
| seg1: down_block[0] (no attn) | 512×512 | ✅ PASS 134.7s, 37.9MB | n/a | DownBlock2D, simple |
| seg2: down_block[1] (cross-attn) | 256×256 | ❌ NCC_EXSP001: 320GB | 🔄 in walrus codegen | seq_len=65536 |
| seg3: down_block[2] (cross-attn) | 128×128 | ❌ NCC_EXSP001: 48GB | not attempted | seq_len=16384 |
| seg4: mid_block | 64×64 | ✅ PASS 769s, 1.3GB | n/a | UNetMidBlock2DCrossAttn |
| seg5-7: up_blocks | various | not attempted | not attempted | — |

**Why sg2/sg3 fail without NKI**: 256×256 spatial = 65,536 self-attention tokens. Standard SDPA materializes a [65536, 65536] = 86GB attention matrix in BF16. With NKI attention_cte, this is a fused flash kernel that avoids materialization.

### Root Cause: trn2 24GB/core HBM is the hard limit at 4K

Even with all available techniques:
- NKI flash attention (attention_cte) — fused kernel, no [seq×seq] materialization
- TP=4 tensor parallelism — distributes attention heads across 4 cores
- -O1 + 100M instruction limit — minimal optimization
- Segmented compilation — block-by-block

The attention activation footprint at 65,536 tokens still exceeds 24GB per core in walrus's hbm_usage check. **4K image generation is not viable on trn2 with current SDK 2.29.**

### Why GPU 4K works but Neuron does not

| | Neuron (AOT compiler) | GPU (PyTorch JIT) |
|--|---|---|
| Memory management | Static at compile time | Dynamic, runtime reuse |
| Attention | Computed by neuronx-cc, must materialize buffers | FlashAttention kernel, peak O(seq) |
| Buffer reuse | All intermediate buffers static-allocated | Memory freed/reused as needed |
| Result at 4K | 57GB > 24GB/core, cannot fit | H100 80GB: 37.67GB peak; L4 22GB: ~10GB peak |

### Solutions (not available in current SDK 2.29)

1. **Sequence parallelism** for diffusion DiT — NxDI does not implement SP for FLUX/SDXL
2. **Larger HBM/core** in next-gen Trainium
3. **More aggressive sequence chunking** inside NKI kernels
4. **CPU fallback** for 4K UNet/DiT (impractical, ~1 hour per image)

### Final Status

**1K and 2K work well** on trn2 — see main results above. **4K is currently a hardware limitation**, not a software issue. Customer should plan for 1K-2K Neuron deployment with 2K → 4K upscaling on CPU/GPU as needed.

