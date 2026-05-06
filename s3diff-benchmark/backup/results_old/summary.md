# S3Diff benchmark — 9-cell matrix (3 devices × 3 resolutions, BF16)

| Device | Res | Load (s) | Cold (s) | Steady (s) | P50 (s) | P95 (s) | Peak device (GB) | Notes |
|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 | 1K | 4.0238 | 1.8071 | 1.2392 | 1.2394 | 1.2486 | 9.0322 |  |
| NVIDIA L4 | 1K | 3.9953 | 2.6916 | 2.3351 | 2.3353 | 2.3408 | 7.9277 |  |
| Trainium2 (LNC=2) | 1K | 42.4748 | 24.6502 | 24.7222 | 24.6142 | 24.8302 | None | unet (tiled, B=1, latent 96x96) |
| NVIDIA H100 80GB HBM3 | 2K | 4.0237 | 25.1914 | 24.3546 | 24.248 | 24.7453 | 15.6963 |  |
| NVIDIA L4 | 2K | 4.0017 | 28.3129 | 27.9134 | 27.9174 | 28.0303 | 15.1797 |  |
| Trainium2 (LNC=2) | 2K | 42.7026 | 133.5739 | 140.2958 | 140.2551 | 140.3366 | None | unet (tiled, B=1, latent 96x96) |
| NVIDIA H100 80GB HBM3 | 4K | 4.0443 | 109.0239 | 106.9482 | 106.7373 | 107.931 | 42.1807 |  |
| NVIDIA L4 | 4K | 4.0011 | 129.329 | 126.4549 | 126.6563 | 126.993 | 16.4688 |  |
| Trainium2 (LNC=2) | 4K | 42.4323 | 596.0912 | 609.1862 | 608.1636 | 610.2089 | None | unet (tiled, B=1, latent 96x96) |

## Configuration
- S3Diff: 1-step SR on SD-Turbo UNet, x4 upscale
- BF16 everywhere (official --mixed_precision bf16 on GPU; Neuron UNet traced with --auto-cast=none)
- Neuron: UNet traced on trn2.3xlarge (LNC=2); VAE encode/decode + text encoder + DEResNet on CPU
- Neuron traced at latent tile 96x96 with baked de_mod (valid for the specific LQ image's degradation score)
- GPU baseline uses london2 bus-grid input; Neuron uses Golden Gate Bridge input (input differs — see note below)

## Notes
- Neuron end-to-end latency dominated by **CPU VAE encode/decode** (~90% of time); Neuron UNet is fast (<5s at 1K tiled).
- For a full apples-to-apples, VAE should also be traced on Neuron (future work).
- Neuron SDK 2.29.0, DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`, venv `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- Accuracy: Neuron vs CPU eager on bridge LQ 1K: PSNR 19.7 dB, mean |diff| 14.8/255. Dominated by tile blending seam; raw UNet cosine ~0.9955.
