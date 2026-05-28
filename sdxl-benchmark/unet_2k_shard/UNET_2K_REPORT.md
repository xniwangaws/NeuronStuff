# SDXL UNet 2K — per-block trace on Trn2 (Plan A)

## Approach

**Per-block trace** (preferred path A from the brief). Split the SDXL UNet
into **9 independent NEFFs** along the natural Down/Mid/Up boundaries and
chain them at runtime. This dodges both the previous failure modes:

- `NCC_EVRF007` (instructions per graph > 5M typical limit) — was 11.9M for
  the monolithic UNet at latent 256² with the whn09 NKI flash attn pattern.
- `NCC_EOOM002` (HBM > 24 GB / Trn2) — DMA ring spills don't accumulate
  across NEFFs because each sub-NEFF is loaded/run independently.

Each sub-NEFF is compiled with `lnc=2` (4 logical cores on
`trn2.3xlarge`), `optlevel=1`, and `--inst-count-limit=15000000`
for the heavy attention blocks (p2..p6); the small stem/head/down0/up2 use
`optlevel=2` (default budget).

NKI flash attention (`attention_isa_kernel`,
`AttentionMMSoftmaxMMWithoutSwap`) is patched into diffusers
`AttnProcessor2_0` only at trace time — CPU dry runs use stock SDPA.

The chained wrapper threads diffusers' standard skip-connection stack:
after all down blocks the LIFO has 9 entries
`[h0, r0a, r0b, r0c, r1a, r1b, r1c, r2a, r2b]`; up_blocks[0] pops the top
3, up_blocks[1] the next 3, up_blocks[2] the bottom 3. `t_emb` (output of
`time_proj`) is computed on the host so the stem stays bf16-only.

## Sub-NEFF inventory (latent 256×256, BF16, batch=1)

| name      | block                     | input shape(s)                   | compile time | NEFF size |
|-----------|---------------------------|----------------------------------|--------------|-----------|
| p0_stem   | conv_in + time/add embed  | (1,4,256,256), (1,320), …        |     5.1 s   |    23.9 MB |
| p1_down0  | DownBlock2D               | (1,320,256,256), (1,1280)        |    44.5 s   |    23.9 MB |
| p2_down1  | CrossAttnDownBlock (2)    | (1,320,128,128), (1,1280), ehs   |   794.4 s   |   331.8 MB |
| p3_down2  | CrossAttnDownBlock (10)   | (1,640,64,64), (1,1280), ehs     |   680.8 s   |  2530.0 MB |
| p4_mid    | UNetMidBlock2DCrossAttn   | (1,1280,64,64), (1,1280), ehs    |   305.2 s   |  1324.7 MB |
| p5_up0    | CrossAttnUpBlock (10)     | (1,1280,64,64) + 3 res + emb,ehs |  1230.8 s   |  4035.1 MB |
| p6_up1    | CrossAttnUpBlock (2)      | (1,1280,128,128) + 3 res + …     |  1445.1 s   |   561.4 MB |
| p7_up2    | UpBlock2D                 | (1,640,256,256) + 3 res + emb    |   104.1 s   |    52.6 MB |
| p8_head   | conv_norm_out+SiLU+conv_out | (1,320,256,256)                |     3.7 s   |     0.5 MB |
| **total** |                           |                                  | **~80 min** | **8.8 GB** |

Per-NEFF HBM is no longer an issue — none of the 9 compiles hit
`NCC_EOOM002` and none hit `NCC_EVRF007`. The NEFF sizes here track model
constants (e.g. p5_up0 has 3 resnets at 1280→1280 plus 10 transformer layers
plus an upsample = the largest).

## Latency at 2048² latent (one full UNet forward)

- Cold: **1463.7 ms**
- Warm: 1438.8 / 1434.1 / 1434.1 / 1434.0 / 1434.2 ms
- Warm mean: **1435.0 ms**
- Warm min: **1434.0 ms**

## Numerical accuracy (Neuron chained vs CPU bf16 reference)

- **cos_sim = 0.999487**  (≥ 0.99 target met)
- max_abs_diff = 0.156, mean_abs_ref = 0.797
- The relative max-abs ~20% is the expected envelope when comparing
  whn09 NKI flash attention (PSum/HBM bf16 path) to stock CPU
  `scaled_dot_product_attention` in bf16 — same accuracy band as the VAE
  Plan A (SD3Diff PR #149) and the existing 1K UNet path.

## Artifacts on the instance

```
/home/ubuntu/work_e2e/
├── unet_neff/                              # 9 NEFFs (8.6 GB), torch.jit.ScriptModules
│   ├── traced_p0_stem.pt
│   ├── traced_p1_down0.pt
│   ├── traced_p2_down1.pt
│   ├── traced_p3_down2.pt
│   ├── traced_p4_mid.pt
│   ├── traced_p5_up0.pt
│   ├── traced_p6_up1.pt
│   ├── traced_p7_up2.pt
│   └── traced_p8_head.pt
├── scripts/
│   ├── trace_unet_2k_perblock.py           # the per-block tracer
│   ├── chained_unet.py                     # ChainedUNet wrapper
│   ├── validate_chained_unet.py            # numerical validation
│   └── dump_unet.py                        # block I/O shape capture
├── unet_shape_map.json                     # per-block shape map
├── UNET_2K_VALIDATION.json                 # cos_sim, latency, sizes
└── UNET_2K_REPORT.md                       # this file
```

## E2E integration notes

- The chained wrapper exposes
  `ChainedUNet.forward(sample, t_emb, text_embeds, time_ids,
  encoder_hidden_states)`. The diffusers `time_proj` should be called
  host-side once per timestep (cheap) — keeps the stem NEFF bf16-only.
- Combined with VAE Plan A (`vae_decoder/chained_vae_decoder.py`, warm
  3.10 s) the 2K E2E path is now unblocked at:
  `1.435 s × N_steps + ~3.1 s VAE`.
- For CFG (batch=2) the wrapper can be wrapped with `DataParallel` across
  the 4 logical cores (or run twice) — current trace is batch=1.
