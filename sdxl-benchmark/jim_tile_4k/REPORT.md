# SDXL Tile-4K on Trn2 — Jim Burtoft pattern

**Goal:** 4096x4096 SDXL on trn2.3xlarge by reusing the 1K UNet via tiled img2img refinement (no UNet recompile at 2K/4K — sidesteps HBM and instruction-limit issues).

**Reference target:** Jim Burtoft 142.62 s end-to-end at 4096x4096.

**Result:** 127.6 s warm / 132.2 s cold at 4K — beats Jim by ~10%.

## Pipeline

prompt -> txt2img @ 1024 (traced 1K UNet/CLIP-L/CLIP-G/VAE-decoder) ~7s
       -> PIL bicubic upscale to 4096 ~0.12s
       -> tile grid (16 tiles @ 1024, no overlap)
       -> img2img per tile (strength=0.3, 6 steps, same 1K UNet) ~7.5s x 16 = 120s
       -> Gaussian-weighted blend ~0.18s
       -> 4096x4096 PNG

Same-shape UNet reused across base (txt2img CFG batch=2, latent 128x128) and refinement (img2img batch=2, latent 128x128). Zero recompile between stages.

## Configuration

| Knob | Value |
|---|---|
| Base resolution | 1024x1024 |
| Target resolution | 4096x4096 |
| Upscaler | PIL.Image.BICUBIC (CPU) |
| Tile size | 1024x1024 |
| Tile overlap | 0 px (16-tile fast) or 128 px (25-tile quality) |
| Refinement strength | 0.3 |
| Base steps | 20 (EulerDiscrete) |
| Refinement steps/tile | 6 |
| Guidance scale | 5.0 |
| Compiler args (UNet/VAE) | --model-type=unet-inference --lnc=2 |

## Latency breakdown

16-tile (no overlap):
- Run 0 cold: 132.16 s (txt2img 7.11 / tile-loop 120.26 / blend 0.19)
- Run 1 warm: 127.61 s (per-tile 7.50 s)

25-tile (overlap=128):
- Run 0 cold: 196.84 s (per-tile 7.37 s)
- Runs 1-3 warm: 191.76 / 192.13 / 191.85 s

## Cost per image (trn2.3xlarge, $2.235/hr)

| Config | Latency | $/image |
|---|---|---|
| 16-tile warm | 127.6 s | $0.0793 |
| 25-tile warm | 191.9 s | $0.1193 |

Half-chip projection (DP=2, single chip used = $1.118/hr): if UNet retraced at batch=1 and split across 2 logical cores, per-tile drops to ~3.7s and 4K total to ~70s = $0.0218/image. Currently single-core; future work.

## Visual sanity

- base_1024.png: clean SDXL 1K txt2img.
- final_4096_16tile_nooverlap.png: sharp 4K, single coherent astronaut on green horse, photo detail in suit and horse, no visible tile seams.
- final_4096_25tile_overlap128.png: same composition, slightly smoother boundaries.
- L4 reference (sdxl_astro_l4_4096/seed42_astro.png): repeated astronauts/horses (classic SDXL native-4K failure). Our tile-4K is qualitatively BETTER, not just faster.

## Implementation gotchas

1. Compiler OOM (F137): 1K UNet with --optlevel 1 exceeded 124 GiB RAM at link time. Added 96 GiB swap; switched compiler args to --model-type=unet-inference --lnc=2; UNet traced cleanly in ~37.5 min.
2. CLIP-L pooled-embed bug: diffusers SDXL encode_prompt picks the first 2D tensor it sees as pooled_prompt_embeds. The traceable wrapper had CLIP-L returning pooler_output (2D, 768-d) as text_embeds, polluting downstream. Fix: NeuronTextEncoder for kind="clip_l" returns last_hidden_state (3D) at index 0 so the ndim==2 check in encode_prompt only fires for CLIP-G.
3. Timestep shape: NEFF was compiled with a 0-dim scalar timestep. The wrapper must reshape(()) — whn09's original expand((sample.shape[0],)) produces 1-D tensor rejected as "Incorrect tensor shape at input #1: received 2, expected ".
4. DataParallel scatter != replicate: DP(unet, [0,1]) scatters dim-0 so CFG batch=2 -> batch-1 per core, mismatching a batch=2 trace. Either trace at batch=1 or run single-core. We chose single-core; LNC=2 gives 4 logical cores, only 1 used today.
