# SDXL UNet TP=4 — Single-NEFF Result

## Summary
TP=4 single NEFF for SDXL UNet works at 1K (1024x1024). cos=0.99999 vs CPU bf16. 2K hit NCC_EXSP001 (39.76 GB > 24 GB per-rank HBM).

## 1K headline (latent 128x128)
| Metric | Value |
|---|---|
| TP | 4 |
| Compile | 13.8 min |
| Cold | 556 ms |
| Warm avg | 322 ms |
| cos_sim | 0.999989 |
| max_abs | 0.0312 (rel 0.9%) |
| Per-rank weights | 1.35 GB (vs 5.13 GB) |
| Linears sharded | 662/743 |
| Conv2d sharded | 34/51 |
| Param sharded | 98.2% |
| NEFFs | 1 (vs 9) |

## 16x16 sanity (validation)
cos=0.999996, warm 46 ms.

## 2K blocker
NCC_EXSP001: 39.76 GB > 24 GB per rank. Activations dominate: 320 ch * 2048^2 * 2 B = 2.5 GB per tensor; sharded /4 = 625 MB but many concurrent across U-Net residuals. Untried levers: -O2; NKI flash attention; chained sharded-activation conv pattern (work_b VAE InputChannelParallelConv2d); half-UNet 2-NEFF fallback.

## Strategy
In-place module replacement: ColumnParallelLinear / OutputChannelParallelConv2d, gather_output=True at every boundary. Threshold 1M params. Diffusers forward unchanged.

## Dead ends
parallel_model_trace hangs in xm.rendezvous after compile - switched to ModelBuilder (same as VAE). --internal-max-instruction-limit does not exist in neuronxcc 2.25.

## Artifacts on 56.124.82.50
/home/ubuntu/work_unet_tp4/{neuron_unet.py, trace_unet_tp4_mb.py, test_unet_tp4.py, UNET_TP4_VALIDATION.json, REPORT.md}
NEFF: /home/ubuntu/work_unet_tp4/compile_tp4_1k/unet_tp4/.../model.neff (~1.3 GB)
