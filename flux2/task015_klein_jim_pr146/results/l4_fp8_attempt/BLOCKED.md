# L4 FLUX.2-klein-9b-fp8 benchmark: BLOCKED

## Outcome
Not runnable on L4 with current OSS stack. Did not produce benchmark numbers.

## Blocker
`black-forest-labs/FLUX.2-klein-9b-fp8` is a single-file FP8 checkpoint
(`flux-2-klein-9b-fp8.safetensors`, 9.4 GB, dtype mix of `float8_e4m3fn`,
`float32` scale scalars, `bfloat16` norms). diffusers 0.38.0 (latest pip
release as of 2026-05-06, and main branch) cannot load it:

1. `Flux2Transformer2DModel.from_single_file(...)` raises
   `RuntimeError: chunk expects at least a 1-dimensional tensor` inside
   `convert_flux2_double_stream_blocks` — the converter renames every
   `.scale` suffix to `.weight`, then unconditionally `torch.chunk`s any
   key containing "qkv" into three. The 0-dim `input_scale` scalar hits
   this path and crashes.

2. Even if that were patched, the converter has no branch for FP8:
   it writes FP8 weights and scalar scales into `attn.to_q.weight` etc.
   `nn.Linear` has no dispatch for `float8_e4m3fn @ bfloat16`, so
   inference would fail at the first matmul. Proper FP8 inference
   requires a quant backend (TensorRT, torchao FP8 linear, FBGEMM-FP8)
   wired into the transformer — not present in diffusers 0.38.

No FP8 Flux2 support exists in diffusers main (2026-05-06). Searched
for `float8|fp8|e4m3|input_scale` in single_file_utils.py — zero hits.
Likely BFL expects users of this checkpoint to use their own inference
stack (or TensorRT). See diffusers issue #11648 for related FP8 loading
discussion.

## What was proved
- HF gated access granted: download succeeded (8.8 GB, 7 files).
- Pipeline scaffolding works: BF16 `Flux2KleinPipeline` loads fine from
  `/home/ubuntu/flux2_klein` (already cached from prior runs).
- The single-file FP8 format is BFL-original layout
  (`double_blocks.*`, fused qkv with `input_scale`/`weight_scale`
  scalars), not the HF-distilled layout the diffusers converter was
  written for.

## Files on instance (i-0e60eb3b74d82c2f3 @ 15.228.197.184, sa-east-1)
- `/home/ubuntu/flux2_klein_fp8/` — downloaded FP8 checkpoint
- `/home/ubuntu/bench_klein_fp8_l4.py` — bench script (unrunnable)
- `/home/ubuntu/bench_l4_fp8.log` — contains the traceback

## Next steps (if ever resumed)
Option A: Skip L4 FP8 — compare H100 FP8 vs L4 BF16-offload / NF4
(already collected) against trn2.
Option B: Install `torchao` FP8 linear + monkey-patch `Flux2Transformer2DModel`
to use torchao's `Float8Linear` (non-trivial, not a 1h task).
Option C: Wait for diffusers to ship Flux2 FP8 support natively.

## Cost impact
L4 g6.4xlarge @ $1.323/hr ran ~1 h across setup + download +
debugging = ~$1.35. Instance terminated after diagnosis.
