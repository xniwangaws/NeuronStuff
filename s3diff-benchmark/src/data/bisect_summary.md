# S3Diff UNet torch.compile Bisection — Phase R4c follow-up

Reference: pure-eager Phase R UNet, `/tmp/phase_r_cat.png` (same seed/input).
Target: find the finest-grained compile scope whose `torch.compile(backend="neuron")`
produces the silent-garbage / compiler-crash cited in R4.

## Results

| Trial | Compile scope                                              | # units | PSNR vs eager  | Cold / Warm s | Error?                                                             |
|:-----:|------------------------------------------------------------|:-------:|:--------------:|:-------------:|--------------------------------------------------------------------|
|   1   | `BasicTransformerBlock.attn1.forward`                      |   16    | **45.08 dB**   | 66.22 / 7.74  | clean                                                              |
|   2   | `BasicTransformerBlock.attn2.forward`                      |   16    | **44.79 dB**   | 43.69 / 7.92  | clean                                                              |
|   3   | `BasicTransformerBlock.ff.forward`                         |   16    | **45.28 dB**   | 42.70 / 7.78  | clean                                                              |
|   4   | `Transformer2DModel.proj_in/out.forward` (DeModLoRALinearAttr) |   32    | **47.21 dB**   | 35.83 / 8.05  | `Cannot process non-contiguous tensors` — silent fallback to eager |
|   5   | Whole `BasicTransformerBlock.forward`                      |   16    | **44.81 dB**   | 85.75 / 6.60  | clean                                                              |
|   6   | Whole `Transformer2DModel.forward`                         |   16    | **43.79 dB**   | 96.08 / 6.14  | clean                                                              |
|   7   | Whole `CrossAttn{Up,Down}Block2D` + `UNetMidBlock2DCrossAttn` (R4 repro) | 7 | **CRASH**   | — / —         | `NCC_IRPX901 RelaxPredicates` → CPU fallback fails `vector::reserve`, `No CPU implementation found for operation: model_default` |

All PSNRs pass. There is **no sub-module inside `Transformer2DModel` that silently breaks.**
Every finer-grained scope — including the whole `Transformer2DModel.forward` — compiles
cleanly and matches eager within ~0.5 dB of the reference (43.40 dB baseline, ±noise).

## The finest-grained broken unit

**The `CrossAttnDownBlock2D` / `CrossAttnUpBlock2D` / `UNetMidBlock2DCrossAttn` compile
scope itself is the smallest unit that reproduces the R4 failure.** Breaking it apart
into (a) its `Resnet` children (known good from R4c) + (b) its `Transformer2DModel`
child (trial 6, clean) eliminates the bug. The bug is not inside `Transformer2DModel`;
it is caused by compiling the **fused graph that spans Resnet + Transformer2DModel +
Downsampler/Upsampler** as a single unit.

## Best hypothesis on the failing op

Compiler trace from trial 7 pinpoints the failure:
```
attentions.1|Transformer2DModel>attentions.1.proj_in|DeModLoRALinearAttr>
  attentions.1.proj_in.lora_A|Linear_dot.63
[INTERNAL_ERROR] [NCC_IRPX901] RelaxPredicates assertion error:
  inst should be valid after relaxing predicates
```
The op that trips the compiler is a matmul inside `DeModLoRALinearAttr.lora_A`
(the rank-32 projection in `proj_in`). In isolation (trial 4) the SAME module
compiles fine (it falls back on "non-contiguous tensors" — a separate issue, but
recovers), and at `Transformer2DModel` scope (trial 6) it is fully clean.
It only fails when the larger `CrossAttn*Block2D` graph is fused — meaning a
**scheduling / predicate pass** (`RelaxPredicates`) that runs on the combined
Resnet+Attention graph hits an internal assert. Most likely a broadcast / stride
constraint introduced by the `einsum("or,bkr->bok", ...)` materializing a (B,out,r)
tensor whose producer is GroupNorm+Resnet output and whose consumer is the attention
matmul — the fused graph crosses a tile-shape boundary the compiler can't fix up.

## Recommendation

- **User workaround — already found.** Compile at `Transformer2DModel.forward` (or
  finer) scope, not at `CrossAttn*Block2D` scope. Combined with R4c's non-attention
  compile (conv_in/out, time_embedding, DownBlock2D, UpBlock2D), this gives full-UNet
  coverage without the crash. Warm runs in trials 5-6 are 6.1-6.6 s (≈23 % faster
  than eager 8.19 s).
- **Compiler issue for AWS.** `NCC_IRPX901 RelaxPredicates assertion` is an internal
  compiler assert, not a user-fixable bug. The error message itself asks users to
  file a support ticket. The underlying issue is compiler scheduling of the fused
  Resnet+Attention graph, reproducible with the attached trial 7 log.
- **Non-contiguous-tensor issue (trial 4)** is a separate, smaller issue in
  `DeModLoRALinearAttr` proj compile — user could `.contiguous()` inside the
  `lora_A` output, or live with the transparent eager fallback (PSNR unaffected).

## Artifacts
- Images: `/tmp/bisect_trial_{1..6}.png` (trial 7 crashed before save)
- Logs: `/tmp/bisect_trial_{1..7}.log` (on laptop + on trn2)
- Script: `~/workspace/s3diff_bisect/phase_bisect.py` on trn2
- Reference: `/tmp/phase_r_cat.png` on trn2
