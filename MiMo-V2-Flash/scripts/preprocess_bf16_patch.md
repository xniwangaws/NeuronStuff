# Preprocess FP8 → BF16 (patch of PR's FP8 preprocess)

PR `contrib/models/MiMo-V2-Flash/src/conversion_script/preprocess_mimo_v2_flash_fp8.py` is the streaming FP8 → Neuron-FP8 converter. To produce a **Neuron-BF16** checkpoint (needed to reproduce PR's BF16 bench numbers), patch the three quantization primitives and the MoE assembly logic to skip requantization.

## Patch summary

1. `convert_bf16_to_fp8_per_row(w)` → return `(w.bfloat16(), None)` (skip FP8 cast).
2. `rescale_fp8_to_per_row(w, scale)` → dequantize blockwise FP8 to fp32, return `(dequant.bfloat16(), None)` (skip re-FP8).
3. `rescale_fp8_weight_blockwise(w, scale)` → dequantize 2D/3D blockwise FP8 (with per-expert scale broadcast), return `(bf16_weight, None)`.
4. In `process_layer`, the MoE assembly block must skip all `gate_up_scale[...]` / `down_scale[...]` writes and the final `out[...gate_up_proj.scale]` / `out[...down_proj.scale]` keys.
5. Output dir will be ~617 GB (vs ~311 GB Neuron-FP8), so ensure ~800 GB free on `/opt/dlami/nvme`.

## Gotcha

When we deleted the HF FP8 source directory to free disk during the BF16 preprocess (thinking it was no longer needed), preprocess crashed because it streams layer-by-layer — layer N reads need to happen after layer 0 writes. **Keep the HF source until preprocess completes.**

## Where to find the full patched file

This session wrote the patch into `~/preprocess_bf16.py` on the remote instance. The diff against the PR's `preprocess_mimo_v2_flash_fp8.py` is ~60 lines. The patched script has been archived to S3:

```
s3://xniwang-neuron-models-us-east-2/mimo-v2-flash/profile/mimo_bf16_results_20260427T1004Z.tgz
```

Extract `preprocess_bf16.py` from the tarball to reuse. The output of this script is the ~617 GB Neuron-BF16 directory required by the PR's old-recipe bench script.
