# SDXL UNet TP=4 at 2K — Attempted, Blocked

## Status
**Blocked.** TP=4 succeeds at 1K (322ms warm, single NEFF — see `../unet_tp4/`) but does not extend to 2K via the chained-shard pattern. Root cause: GroupNorm divisibility.

## Approach Tried

Chain Output→Input channel-parallel pattern in ResnetBlock2D:
- `conv1`: `OutputChannelParallelConv2d(gather_output=False)` — keep activation 1/4 channels
- `time_emb_proj`: `ColumnParallelLinear(gather_output=False)` — sharded temb to match
- `norm2 (GroupNorm 32)`: per-rank
- `conv2`: `InputChannelParallelConv2d(input_is_parallel=True)` — all-reduce to full

This pattern is the validated Plan B VAE recipe (`../../work_b/neuron_vae_decoder.py`, cos 0.9997 at 1K).

Plus NKI flash attention monkey-patch on `Attention.get_attention_scores` (from Plan A per-block UNet).

## Why It Fails

`SDXL UNet GroupNorm has num_groups=32 hardcoded`. Sharded channel count (channels / TP=4) often does not satisfy GroupNorm's two requirements simultaneously:

1. **Channel divisibility**: sharded ch must be divisible by 32. Smallest resnet block has 320 ch → 320/4 = **80, not divisible by 32** (80/32 = 2.5).
2. **Weight shape**: GroupNorm.weight is `[num_channels]` of length 320, but per-rank input has 80 channels — `Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [320] and input of shape [1, 80, ...]`.

Even adding a `MIN_CH_FOR_CHAINED_SHARD=512` filter (only chain-shard the bigger 640/1280 ch resnets) still hits the same GroupNorm errors at boundary sites.

## What Would Be Needed

A `ParallelGroupNorm` that:
- Shards weight + bias along channel dim
- Computes per-rank stats over its sharded channel slice (changes math vs reference, gives different numerics)
- Or: `gather → norm → split` in forward (correct math, adds collective ops, may not save HBM)

NxD doesn't ship one as of SDK 2.32. Writing it would take a few hours of correctness debugging.

## Iteration Summary
4 smoke compile attempts, all failed:
1. Initial: `RuntimeError: Shapes are not compatible for broadcasting: bf16[1,160,8,8] vs. bf16[1,640,1,1]` (sharded conv1 + non-sharded temb)
2. Add sharded `time_emb_proj`: same error class, different shapes
3. Lower SHARD_NUMEL_THRESHOLD: same broadcasting error class
4. Add MIN_CH_FOR_CHAINED_SHARD=512 filter to skip 320 ch blocks: GroupNorm weight shape error `bf16[1, 80, 16, 16]` vs `weight[640]`

## Recommendation

For SDXL 2K on Trn2:
- **Primary**: use Jim's tile-img2img approach at 4K (`../jim_tile_4k/`, validated 127.61s warm). Same approach works for 2K trivially.
- **Alternative**: trn2.48xlarge for direct 2K compile (8× HBM headroom).
- **Future**: implement `ParallelGroupNorm` once NxD has it (or write a custom one) to unlock TP=4 chained-shard at 2K and beyond.

## Files

- `STRATEGY_2K.md` — original plan
- `neuron_unet_2k.py` — final attempted implementation (chained shard + NKI flash)

Both kept for reference. Driver and validation scripts identical to working 1K version (`../unet_tp4/`).
