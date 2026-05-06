# Phase R — S3Diff NxDI-style Rewrite (DeModLoRA)

Follow-up to Phase E. Goal: reduce the 7,344 ops/image host-dispatch overhead by
rewriting S3Diff's peft LoRA layers (which produce `Linear→einsum→Linear→add`
per site) into a single `DeModLoRA*` module that folds the matmul and einsum
(Phase E C-1a algebra). All 257 UNet + 31 VAE-encoder LoRA sites are replaced.

Ship target: **eager bf16 with DeModLoRA**, 8.19s mean warm latency on 5 test
images, PSNR 42.78 dB ± 2.44 vs CPU fp32 reference.

## TL;DR comparison (cat 1K)

| Stack | Warm | Cold | PSNR vs CPU fp32 |
|---|---|---|---|
| H100 bf16 | **1.26s** | 9.6s | 45.10 dB |
| L4 bf16 | 2.34s | 6.4s | 45.15 dB |
| CPU eager fp32 | 54.0s | — | ref |
| Trn2 Phase 3 trace fp32 | 24.91s | ~30 min/image | 24.55 dB (seam) |
| Trn2 Phase E eager bf16 (baseline) | 8.60s | 73s | 43.26 dB |
| **Trn2 Phase R eager bf16 DeModLoRA** | **8.19s** (-4.8%) | 30s first, 8s cached | **43.40 dB** (+0.14) |
| **Trn2 Phase R + R4c selective compile** | **7.93s** (-7.8%) | 64s | **43.41 dB** (+0.15) |

Mean across 5 test images (cat/bus/bird/butterfly/woman): Phase E 8.75s /
42.72 dB → Phase R 8.19s / 42.78 dB (+0.06 dB mean). On cat specifically,
Phase R is 8.17s / 43.40 dB and **R4c is 7.93s / 43.41 dB**.

**Still 6.3× slower than H100.** The remaining gap is host dispatch and
attention-on-Neuron compile bugs (see section 4). Path B full NxDI port (4-6
person-weeks) remains the only known path to ~2s on Trn2.

## What changed

### 1. Weight conversion (R1)

`scripts/extract_state_dicts.py` + `scripts/convert_state_dict.py` convert the
S3Diff UNet checkpoint (1,200 keys, 898M params) and VAE decoder (138 keys, 49M
params) from the peft layout into a clean Neuron-shaped state dict. Mapping:

```
.base_layer.weight              ->  .base.weight
.base_layer.bias                ->  .base.bias
.lora_A.default.weight          ->  .lora_A.weight
.lora_B.default.weight          ->  .lora_B.weight
```

Round-trip verified — parameter count, shapes, and dtypes all preserved.

Key finding: **VAE decoder has no LoRA** in S3Diff. Only encoder (31 sites).
UNet has 257 sites. Both are known from `data/lora_targets.txt`.

### 2. DeModLoRA modules (R2 / R3)

`modules/de_mod_lora.py` + `modules/de_mod_lora_attr.py`:
- `DeModLoRALinear{,Attr}` — Linear with de_mod-modulated LoRA; folds into
  2 matmuls instead of 3 via `bd = einsum('or,bkr->bok', lora_B.weight, de_mod)`
  then `out_adapter = einsum('bok,...k->...o', bd, lora_A(x))`.
- `DeModLoRAConv2d{,Attr}` — Conv2d variant, same idea on channel dim.
- The `Attr` variant reads `self.de_mod` as a module attribute; S3Diff's outer
  `forward` sets these attributes per image (original semantics preserved).
- `replace_lora_modules_in_unet()` walks a diffusers UNet, swaps every
  `peft.tuners.lora.layer.Linear` / `Conv2d` for our DeMod equivalent, copies
  weights, preserves scaling (UNet = 0.25, VAE = 0.5).

Correctness gates:
- Unit test (random weights, fp64): `max|diff| = 1e-14` — algebraic identity.
- Single layer (loaded S3Diff weights, fp32): `max|diff| < 1e-7`.
- BasicTransformerBlock (all 10 LoRA sites × weights loaded): **cosine = 1.000001**.
- Transformer2DModel wrapper: **cosine = 1.000000**.
- ResnetBlock2D with conv_shortcut: **cosine = 1.000376**.
- Full UNet (257 modules replaced): **cosine = 1.000000**, max|diff| = 1.07e-6.

### 3. Why this is (only) +6% faster

Every LoRA forward was `Linear(base) + Linear(lora_A) + einsum + Linear(lora_B)
+ mul + add` = ~6 eager ops. Our folded version has `Linear(base) +
Linear(lora_A) + einsum(fold) + einsum(apply) + mul + add` = ~6 ops still —
**same op count, different arrangement**. The 6% win is probably from:
- One less kernel launch per LoRA site (2 einsums replace 1 matmul + 1 einsum)
- Cleaner fused dispatch pattern in the Neuron eager backend

**The 7,344 ops/image host dispatch gap (42% wall) is still there**. Only a
real trace (which this `torch_neuron_eager` SDK doesn't expose) can attack that.

### 4. torch.compile partial works — selective compile (R4c)

Tried `torch.compile(backend="neuron")` at multiple granularities. Results:

| Compile scope | Warm | PSNR | Status | Log |
|---|---|---|---|---|
| Full UNet | 4.83s | **13.41 dB** | ❌ Hallucinated output | `phase_r4.log` |
| `down_blocks` (4 CrossAttnDownBlock2D) | 7.33s | **24.53 dB** | ❌ Garbage | `phase_r4b_down.log` |
| `mid_block` (UNetMidBlock2DCrossAttn) | 7.98s | 43.24 dB | ⚠️ -0.16 dB | `phase_r4b_mid.log` |
| `conv_in + conv_out` only | 8.13s | 43.40 dB | ✅ Safe | `phase_r4b_convs.log` |
| **R4c: non-attention (conv_in, conv_out, time_emb, DownBlock2D[3], UpBlock2D[0])** | **7.93s** | **43.41 dB** | ✅ **Ship candidate** | `phase_r4c.log` |

**Finding**: any compile unit that contains `Attention` (diffusers' `Attention`
class under `BasicTransformerBlock`) silently produces wrong output. The log
shows `CPU fallback failed for 'model_default': No CPU implementation found
for operation: model_default` — some op in the attention graph has no Neuron
kernel, and fallback fails without raising. `NCC_IRPX901` from Phase E didn't
fire this time (DeModLoRA changed the IR enough) but a different compile path
is broken.

**R4c (non-attention compile)** compiles only the 5 submodules without
attention. Warm drops from 8.19s (eager DeMod) to **7.93s** with identical
PSNR. This is the best _shippable_ compile scope.

Artifact images: `cat_phase_r4c.png` (safe, 43.41 dB), `cat_phase_r4b_cat.png`
(convs only), `cat_phase_r4b_mid.png` (mid, 43.24 dB), `cat_phase_r4b_down.png`
(visibly broken, 24.5 dB).

## 5-image bench (1K SR, cat_LQ_256 size)

Images, CPU fp32 references, and compute mirror Phase E multi-image bench.
Same seed, same pipeline config, same image set.

| image | Phase E warm | **Phase R warm** | Phase E PSNR | **Phase R PSNR** |
|---|---|---|---|---|
| cat | 8.78s | **8.17s** | 43.26 dB | **43.40 dB** |
| bus | 8.85s | 8.09s | 38.54 dB | 38.64 dB |
| bird | 8.66s | 8.31s | 45.65 dB | 45.63 dB |
| butterfly | 8.80s | 8.22s | 41.71 dB | 41.71 dB |
| woman | 8.69s | 8.16s | 44.46 dB | 44.52 dB |
| **mean** | 8.75s | **8.19s** (-6.4%) | 42.72 dB | **42.78 dB** |

PSNR floor (bus image, 38.6 dB) is unchanged — it's a content-specific bf16 ULP
accumulation issue (see `phase_e/data/dispatch_architecture.md` §diff analysis).
The bird image at 45.6 dB is already near the Neuron fp32 ceiling.

## Conclusion

Phase R delivers a **small but clean engineering win** (+6% warm, +0.06 dB
PSNR, -59% cold on first image) without breaking correctness. The real
bottleneck remains host dispatch, which this eager SDK cannot address without
a proper trace API. If AWS ships `torch_neuronx.trace` in a future release, or
we port to NxDI's Flux-style architecture (4-6 person-weeks per Path B
research), latency could drop to 3-5s.

The `torch.compile(UNet)` experiment is an open item — the 4.83s warm number
is tantalizing, but the 13 dB PSNR is unusable. Next step there: bisect which
sub-module produces the `model_default` fallback error.

## Files

- `README.md` — this file
- `scripts/` — R1 tooling (dump, convert) + R4 smoke/compile drivers
- `modules/` — DeModLoRALinear/Conv2d, Attention, TransformerBlock,
  Transformer2DModel, ResnetBlock2D (all correctness-tested)
- `tests/` — unit + block-level + full UNet correctness tests
- `data/` — state_dict key dumps, LoRA targets, UNet class summary
- `images/` — 5-image SR outputs (cat/bus/bird/butterfly/woman)
- `logs/` — `phase_r_bench.log` (production bench), `phase_r4.log` (compile attempt)
