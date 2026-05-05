# S3Diff Eager Dispatch Architecture — why 8.6s and how to fix

## The 8.77s decomposition (profile from bf16 warm run 2)

```
┌─ Wall time: 8.77s ──────────────────────────────────────────────┐
│  Host dispatch gap  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  3.72s (42%) │
│  Output copyout     ▓▓▓▓▓▓▓▓                      1.18s (13%) │
│  Device compute     ▓▓▓▓▓▓▓▓                      1.18s (13%) │
│  Kernel sync        ▓▓▓▓▓▓                        0.81s (9%)  │
│  Model submit       ▓▓▓▓                          0.62s (7%)  │
│  DMA + misc         ▓▓▓▓▓▓▓▓▓                     1.26s (15%) │
└─────────────────────────────────────────────────────────────────┘
```

**Device compute is 13% of wall**. The other 87% is host-side dispatch, sync, and data movement.

## The 7,344-submission problem

Each S3Diff inference submits **7,344 NEFFs** to the Neuron runtime. Each submission requires:
1. PyTorch eager dispatch → torch_neuronx custom backend
2. HLO construction for the op (or cache lookup)
3. `nrt_model_submit` call via the Neuron runtime
4. Synchronization fence before next op
5. Optional data copy to/from DRAM

Even if each submission took only 500µs of host time, 7344 × 500µs = **3.67s** just in overhead. That matches the observed 3.72s "host dispatch gap".

H100 avoids this through:
- **cuDNN graph capture**: attention / convolution / norm are each 1 CUDA kernel launch, not thousands
- **cuBLAS batched GEMM**: LoRA `A @ B @ x` fuses into a single kernel
- **Python-side graph mode** (`torch.compile`): 100× fewer Python interpreter round-trips

## Three paths to fix this

### Path A: Pipeline-level capture (estimated 3-5s warm)

Break S3Diff into 3 fixed-shape subgraphs and trace each **once**:

```python
# Graph 1: VAE encoder (tile body, 224×224 → 28×28 latent)
vae_enc_traced = torch_neuronx.trace(
    vae.encoder.original_forward,
    example_inputs=torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device="neuron"),
)

# Graph 2: UNet single forward (with de_mod as input, not attribute — this is the key change)
def unet_with_de_mod(z, t, text_embed, de_mod_tensor):
    # inject de_mod via functional closure instead of attribute
    ...
unet_traced = torch_neuronx.trace(
    unet_with_de_mod,
    example_inputs=(latent, timesteps, text_embed, de_mod),
)

# Graph 3: VAE decoder (tile body)
vae_dec_traced = torch_neuronx.trace(
    vae.decoder.original_forward,
    example_inputs=torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device="neuron"),
)
```

This is basically **Phase 3 trace mode, but with de_mod as an input tensor instead of a baked attribute**. Phase 3's attribute-bake workaround forced one NEFF per image; passing de_mod as an input means one NEFF total.

**Pros**: guaranteed to cut submissions from 7344 to ~100 (3 traced graphs × ~30 invocations for tiling).
**Cons**: re-introduces the compile-time cost of tracing (~20-40 min per graph), needs Phase 3's attribute-routing hack to be rewritten.
**Estimated warm**: 3-5s/image if the three traces hit the same NEFF cache.

### Path B: NxDI-style coarse kernels (estimated 1.5-3s warm)

`neuronx-distributed-inference` ships pre-fused kernels for diffusion-model building blocks:
- `FusedAttentionBlock` (Q/K/V projection + flash attention + output proj in one kernel)
- `FusedResNetBlock` (conv + norm + silu + conv + residual as one kernel)
- `FusedUpsampleBlock` (upsample + conv fused)

Adapting S3Diff to use NxDI's attention/resnet/upsample would:
- Reduce UNet from ~500 ops/forward to ~30 ops (one per transformer block and resnet)
- Preserve LoRA by passing `de_mod` into the fused attention kernel as an extra tensor
- Enable bf16 end-to-end without manual cast patches

**Pros**: matches H100's "few big kernels" dispatch pattern. Production-ready runtime.
**Cons**: major porting effort (~1-2 weeks). Need to map diffusers attention API → NxDI attention API.
**Estimated warm**: 1.5-3s/image based on NxDI's own SDXL-like benchmarks on trn2.

### Path C: Fix upstream torch.compile + dynamic shapes (estimated 4-8s warm)

The current Phase E2 failure modes are:
- `NCC_IXTP002 >10M instructions` on whole VAE decoder → fixable by shape padding (Task 2 proved this works for the decoder tile body)
- `NCC_IRPX901 RelaxPredicates` inside UNet LoRA proj_in → real SDK compiler bug, needs an AWS issue

Once those are fixed, `torch.compile(backend='neuron', dynamic=False, fullgraph=False)` on the UNet with shape-padded inputs should cut per-op dispatch down to ~one submission per UNet forward (vs the current thousands).

**Pros**: pure engineering + one upstream bug report. Keeps the eager codepath for non-hot sections.
**Cons**: depends on AWS fixing NCC_IRPX901. Compile cold start stays painful.
**Estimated warm**: 4-8s/image with full UNet compile, 6-7s with VAE-decoder-only compile (already shown in Phase E2 stage 1).

## Recommendation

Short term (1 week): **Path C**, specifically the VAE-decoder-only compile on top of Task 2's padding. Already has data showing it's feasible.

Medium term (1 month): **Path A**, rewriting de_mod as a functional input and tracing the three graphs. Re-uses Phase 3 infrastructure.

Long term (1 quarter): **Path B**, NxDI port. This is the only path that closes the H100 gap to within 2× instead of 7×.
