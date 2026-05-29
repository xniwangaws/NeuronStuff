# 2K Strategy

## 1K baseline (works)
- All Conv2d -> OutputChannelParallelConv2d(gather_output=True), all Linear -> ColumnParallelLinear(gather_output=True)
- 662 Linears + 34 Convs sharded
- Per-rank weights: 1.35 GB, warm 322ms, cos 0.999989

## 2K block
- NCC_EXSP001: 39.76 GB > 24 GB per rank. Activations dominate at 2048x2048.
- 320 ch * 2048^2 * 2 B = 2.5 GB per tensor (gathered). Sharded /4 = 625 MB.
- Highest-res self-attn at 128x128 = 16384 tokens, attn matrix 16384^2 * 2B = 512 MB unsharded.

## Plan for 2K
1. Resnet pattern: conv1 OutputChannel(gather_output=False) [sharded out] -> norm2 sharded GroupNorm(8 groups/rank from 32) -> SiLU -> conv2 InputChannel(input_is_parallel=True) [all-reduce to full]
2. shortcut conv (when in!=out): OutputChannel(gather=True). Block I/O is full.
3. NKI flash attn: patch BasicTransformerBlock.attn1/attn2 via diffusers AttnProcessor2_0.__call__ + Attention.get_attention_scores monkey-patches at trace time.

## NKI flash attn TP=4
- SDXL has heads in {5, 10, 20} -- 10/5 not divisible by 4.
- Keep attention projections gathered (gather_output=True). Apply NKI flash kernel on full heads.
- The flash kernel removes the materialized attn matrix peak; per-rank still has full Q,K,V but no 16384x16384 mat.

## Pattern summary for 2K
- Resnet conv chain: conv1(gather=False) -> ShardedGroupNorm -> SiLU -> conv2(input_is_parallel=True). Saves activation memory between conv1 and conv2.
- Other Conv2d: OutputChannel(gather=True) -- keeps boundaries full, simple.
- All Linear: ColumnParallelLinear(gather=True) -- same as 1K.
- NKI flash attn: monkey-patch at trace time.
