# Jim's S3Diff Port (torch_neuronx.trace) — Attempt Archive

Archive of the `torch_neuronx.trace()` port of S3Diff based on
[jimburtoft/NeuronStuff/S3Diff](https://github.com/jimburtoft/NeuronStuff/tree/main/S3Diff)
and [PR aws-neuron/neuronx-distributed-inference#149](https://github.com/aws-neuron/neuronx-distributed-inference/pull/149).

## Outcome

**128×128 → 512×512 (0.5K output)**: reproduces Jim's claim (0.482s warm, 2.07 img/s).
**BUT** the output is silently **all-NaN** under `--auto-cast=matmult` flag
(`diag_nan2.py` confirms VAE encoder returns NaN). Jim's notebook has no
sanity check on output, so this failure was invisible in his own demo.

**256×256 → 1024×1024 (1K output)**:
- `--auto-cast=matmult` flag: UNet compiles, output is all-NaN.
- `--model-type=unet-inference -O1` flag (per Jim's PR 149): UNet recompiled,
  output is no longer NaN but has **visible artifacts** (PSNR 32.98 dB vs CPU
  fp32 — significantly lower than Trial 6's 43.10 dB).

## Why

Root cause hypothesis (not confirmed with Jim): his wrapper sets
`module.de_mod = de_mod_all[:, block_idx]` inside forward, which may be baked
into the traced graph at the `latent=64` calibration shape but not correctly
tracked at `latent=128`.

## Files

- `run_jim_s3diff.py` — command-line port of Jim's notebook
- `run_jim.py` — earlier version
- `recompile_unet_256_nocast.py` — UNet recompile with `--model-type=unet-inference`
- `resume_256_vae_enc.py` — VAE encoder recompile (NCC_IBIR182 retry)
- `diag_nan.py`, `diag_nan2.py`, `diag_nan3.py`, `diag_nan4.py` — NaN diagnostics
- `psnr_compare.py` — cross-stack PSNR comparison
- `logs_*.log` — all compile/run logs for LR=128 and LR=256
- `*_jim_neuron_256.png` — actual inference outputs (mostly black/artifact)
- `psnr.json`, `images_bench.json`, `global_metric_store.json` — raw metrics

## Replication

Not recommended as-is. If re-attempting:
1. Start from Jim's upstream PR 149 code (cleaner than notebook), not this archive.
2. Fix `--auto-cast=matmult` → `--model-type=unet-inference -O1` for all LoRA components.
3. Test output numerically (`torch.isnan(output).any()`) after first inference.
4. If LR=256 still produces artifacts, investigate stacked-de_mod tracing behavior
   vs Phase B's per-site `Dict[str, Tensor]` approach.
