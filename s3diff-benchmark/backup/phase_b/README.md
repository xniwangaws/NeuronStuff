# Phase B — Path B implementation: full custom NeuronS3DiffUNet

Preparatory implementation for the "NxDI-style port" of S3Diff. Given our
current eager-only Neuron SDK has no `torch_neuronx.trace` API, we can't
actually produce traced NEFFs today. But we can write the **full UNet from
scratch** using DeModLoRA modules, matching diffusers' block structure, so
when a trace API becomes available this is one `torch_neuronx.trace()` call
away from a single-NEFF UNet.

The full UNet (898M params, 257 LoRA sites, 4 down + mid + 4 up blocks, 16
transformer2d models, 22 resnets) is tested to `cosine=1.000000` vs the
diffusers reference on CPU fp32.

## TL;DR

| Stack | Warm (cat) | PSNR (cat) | 5-img mean warm | 5-img mean PSNR |
|---|---|---|---|---|
| Phase E baseline | 8.60s | 43.26 dB | 8.75s | 42.72 dB |
| Phase R (attr-replace) | 8.17s | 43.40 dB | 8.19s | 42.78 dB |
| Phase R4c (selective compile) | 7.93s | 43.41 dB | — | — |
| **Phase B (full custom UNet)** | **7.77s** | **43.40 dB** | **7.97s** | **42.78 dB** |
| Phase B + NKI attn1 | 8.31s | 43.36 dB | — | — |

Phase B (full-custom UNet, no compile) is the fastest **shippable** path on
this SDK — **7.97s warm mean, 5-image PSNR 42.78 dB** (matches Phase R
quality, identical to Phase E/R within noise). H100 bf16 = 1.26s, L4 = 2.34s;
we're still 6.3× / 3.4× slower, limited by host dispatch (42% of wall per
Phase E profile).

## What's new vs Phase R

Phase R **monkey-patches** peft LoraLayer in a live `diffusers.UNet2DConditionModel`
with `DeModLoRALinearAttr` / `DeModLoRAConv2dAttr`. It works but the module
tree is still diffusers' tree.

Phase B **builds the UNet from scratch**. Every block type (`NeuronS3DiffDownBlock2D`,
`CrossAttnDownBlock2D`, `UNetMidBlock2DCrossAttn`, `UpBlock2D`, `CrossAttnUpBlock2D`,
`Transformer2DModel`, `BasicTransformerBlock`, `Attention`, `FeedForward`, `ResnetBlock2D`,
`Downsample2D`, `Upsample2D`) is a small custom `nn.Module` that:
- Uses `DeModLoRALinear` / `DeModLoRAConv2d` (forward-arg de_mod, trace-friendly)
- Accepts `de_mod_map: Dict[str, Tensor]` keyed by full module path
- Mirrors diffusers' forward structure 1:1

A small **adapter** (`custom_unet_adapter.wrap_custom_unet`) keeps the outer
S3Diff forward loop unchanged: it assigns `module.de_mod = ...` on the
diffusers peft modules as before, and the adapter harvests those tensors
into a `de_mod_map` before calling our custom UNet.

Correctness: `cosine = 1.000000, max|diff| = 1.07e-6` vs diffusers reference
(random weights + random de_mod inputs, fp32).

## What this unblocks

The full UNet has **one explicit forward signature** that takes `(sample,
timestep, encoder_hidden_states, de_mod_map)`. All 257 de_mod tensors flow
through as arguments — no module attributes, no Python-level hooks.

When AWS ships `torch_neuronx.trace` in the eager SDK (or we port to
production NxDI), tracing this UNet is a one-liner:

```python
traced_unet = torch_neuronx.trace(
    custom_unet,
    example_inputs=(
        torch.randn(1, 4, 128, 128, dtype=torch.bfloat16, device="neuron"),
        torch.tensor([999], device="neuron"),
        torch.randn(1, 77, 1024, dtype=torch.bfloat16, device="neuron"),
        {name: torch.zeros(1, 32, 32, dtype=torch.bfloat16, device="neuron")
         for name in de_mod_keys},
    ),
    compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
)
```

At that point 7,344 per-image ops collapse to 1 NEFF launch, and latency
should approach H100 territory (estimated 1.5-3s based on NxDI Flux numbers).

## What doesn't work yet

1. **NKI attn2 (cross-attention) segfaults** — `nkilib.core.attention.attention_cte`
   crashes with SIGSEGV when fed asymmetric seqlen (16384 image × 77 text).
   Task 3 earlier hit the same bug. Our Phase B tries NKI only on attn1
   (symmetric self-attn, proven in Task 3); attn2 stays on SDPA. NKI attn1
   runs: 8.31s warm, 43.36 dB PSNR (within Phase B noise).

2. **Host dispatch is still the bottleneck**. The custom UNet does the same
   number of per-op submissions as the diffusers UNet — eager doesn't batch.
   Only `torch_neuronx.trace` fixes this.

3. **`torch.compile` on attention still breaks accuracy**. Tested in Phase
   R4/R4b/R4c — any compile unit containing `Attention` produces 13-24 dB
   garbage. Phase B doesn't try compile; it relies on the cleaner module
   tree being trace-ready once that API arrives.

## Files

- `modules/s3diff_attention.py` — `NeuronS3DiffAttention` with optional NKI
  kernel (env `S3DIFF_USE_NKI_ATTN=1`). Uses `F.scaled_dot_product_attention`
  by default; NKI `attention_cte` for attn1 only when enabled.
- `modules/s3diff_unet_blocks.py` — 5 block wrappers + Downsample2D/Upsample2D.
- `modules/s3diff_unet.py` — top-level `NeuronS3DiffUNet` (signature-compatible
  with `diffusers.UNet2DConditionModel.forward`).
- `modules/custom_unet_adapter.py` — bridges S3Diff's outer forward
  (`module.de_mod = tensor` assignment) to our custom UNet's argument-based
  de_mod.
- `scripts/phase_b_smoke.py` — single-image smoke.
- `scripts/phase_b_bench.py` — 5-image bench (cat/bus/bird/butterfly/woman).
- `tests/test_full_custom_unet.py` — end-to-end cosine test (also provides
  `copy_weights` helper, imported by smoke/bench).
- `images/` — 5 SR outputs + cat with NKI attn1.
- `logs/` — smoke, NKI-attn1 smoke, 5-image bench.

## How to run (on trn2.3xlarge with eager SDK)

```bash
source ~/workspace/native_venv/bin/activate
cd ~/workspace/s3diff_nxdi
python phase_b_smoke.py --device neuron --dtype bf16 --output_image /tmp/out.png
python phase_b_bench.py   # full 5-image bench
S3DIFF_USE_NKI_ATTN=1 python phase_b_smoke.py --device neuron --dtype bf16 --output_image /tmp/out_nki.png
```

## Next steps for actual NxDI deployment

1. **`torch_neuronx.trace` integration** — one line, once the API is
   available. Would also need the `Dict[str, Tensor]` as input type, which
   most trace APIs support via flattening.

2. **Shape normalization for tile NEFFs** — Phase E Task 2 proved padding
   VAE decoder tiles to uniform shapes eliminates the 10M-instruction
   compile bug. Same trick applies to UNet tiles: pad to 128×128 latent
   regardless of content size.

3. **`MultiLoraLinear` / `MultiLoraConv2d` compatibility** — NxDI's
   production LoRA (`neuronx_distributed_inference.modules.lora_serving`)
   selects one of N pre-compiled adapters. Ours modulates within one adapter
   via `de_mod`. A full port either (a) extends MultiLoraLinear with a
   de_mod input, or (b) uses our DeModLoRA* directly. (a) is the "proper"
   NxDI integration.

4. **TP sharding** — unnecessary on trn2.3xlarge (LNC=2, ~24GB/core, UNet
   fits), but a 1-line `ColumnParallelLinear`/`RowParallelLinear` swap in
   `NeuronS3DiffAttention` would enable TP=2 for future larger resolutions.
