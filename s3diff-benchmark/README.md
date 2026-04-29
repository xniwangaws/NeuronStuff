# S3Diff benchmark — Neuron vs H100 vs L4

Port [S3Diff](https://github.com/ArcticHare105/S3Diff) (1-step super-resolution on SD-Turbo) to AWS Trainium2, compare against H100 and L4 at 1K / 2K / 4K output resolution in BF16.

Weights: `zhangap/S3Diff` + `stabilityai/sd-turbo`.

## 9-cell matrix (BF16 steady-state latency, seconds)

| Device | 1K | 2K | 4K |
|---|---|---|---|
| H100 80GB (p5.4xlarge) | **1.24** | 24.4 | 107 |
| L4 24GB (g6.4xlarge) | 2.34 | 27.9 | 126 |
| Trn2 v1 (UNet traced, CPU VAE) | 24.7 | 140.3 | 609 |
| **Trn2 v2 (UNet + VAE traced)** | **14.7** | 58.9 (stdout only) | — (SSH dropped mid-run) |

Peak device memory (nvidia-smi sampled):
- H100: 9 / 15.7 / 42.2 GB
- L4: 7.9 / 15.2 / 16.5 GB

v2 numbers: 1K JSON in `results/trn2v2_1K.json`; 2K lost to instance shutdown (only captured from stdout log). 4K not completed.

## Why Trn2 is still slower than GPU

| Res | Trn2 v1 total | Neuron UNet compiled | CPU VAE + text enc + misc | VAE % |
|---|---|---|---|---|
| 1K | 24.7s | 4.3s | 20.4s | 83% |
| 2K | 140.3s | 17.1s | 123.2s | 88% |
| 4K | 609s | 68.5s | 540.5s | 89% |

v1 was dominated by CPU VAE, not Neuron compute. v2 adds traced VAE encoder + decoder:
- 1K v2 = 14.7s: UNet 4.3s + VAE enc 0.5s + VAE dec 9.9s (decoder is now the bottleneck)
- For reference: Neuron UNet tile 0.535s (compiled), VAE enc 0.032s/tile, VAE dec 2.48s/tile

Per-component accuracy vs CPU eager:
- UNet tile cosine 0.9955, max|diff| 1.76
- VAE encoder max|diff| 0.40
- VAE decoder **max|diff| 0.00000** (bit-exact)

## Known accuracy issue

End-to-end PSNR on 1K = 19.7 dB vs CPU eager. **Not** a compute bug — it's **tile-blending seam + per-tile UNet drift**:
- S3Diff's default Gaussian tile blend uses `var=0.01` (edge weight → 0), so a tiny per-tile numerical drift (0.5% cosine gap) becomes a visible horizontal band at tile boundaries.
- Mitigated to `var=0.1` — seam shrank from 83.6 → 71 luma discontinuity, but still visible.
- VAE decoder tile boundaries also show slight per-channel fringing (BF16 accumulator + replicate padding).
- **Content is correct** (mean RGB matches CPU eager within 0.3%).

**Root-cause fix (not implemented yet)**: rewrite `my_lora_fwd` to take `de_mod` as a tensor argument (today it's a runtime attribute, so trace bakes it as a constant valid only for one specific LQ image). Then trace UNet **at full latent size per bucket** instead of tiled at 96×96. Eliminates tile blending entirely at 1K and reduces tile count at 2K/4K. Math unchanged; purely an implementation refactor (~30 LOC in `src/model.py` + `src/s3diff_tile.py`). Estimated 2h dev + 1h compile per resolution bucket.

## What's in this folder

```
s3diff-benchmark/
├── README.md                    # this file
├── results/
│   ├── summary.md               # markdown 9-cell table + context
│   ├── summary.csv              # csv for analysis
│   ├── {H100,L4}_*_bf16.json    # 10-run GPU benchmarks
│   ├── trn2_{1K,2K,4K}.json     # v1: UNet traced, CPU VAE
│   └── trn2v2_1K.json           # v2: UNet + VAE traced
├── scripts/
│   ├── bench.py                 # GPU benchmark harness
│   ├── neuron_unet_trace_v2.py  # trace UNet with de_mod baked
│   ├── trace_vae.py             # trace VAE encoder + decoder
│   ├── neuron_e2e.py            # v1 e2e pipeline
│   ├── neuron_e2e_v2.py         # v2 e2e pipeline (UNet + VAE traced)
│   ├── make_report.py           # generate summary.md + comparison grids
│   └── setup_and_trace_vae.sh   # one-shot setup on fresh trn2
└── docs/
    └── s3diff.md                # project registry entry with port notes
```

## Reproduce

**Hardware used:**
- H100: p5.4xlarge in ap-northeast-1c (sa-east-1 / us-east-* had no p5.4xlarge capacity on 2026-04-24)
- L4: g6.4xlarge in sa-east-1a
- Trn2: trn2.3xlarge in sa-east-1b, capacity block required ($17-54)

**Neuron SDK 2.29** (DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`, AMI `ami-0b0749742fb2391dc` in sa-east-1). Pre-installed venv `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` has `torch-neuronx 2.9.0.2.13`, `neuronx-cc 2.24.5133`, `NxDI 0.9.0`.

**S3Diff deps** (compatible with Neuron SDK 2.29's torch 2.9):
- `diffusers==0.34.0`, `peft>=0.15.0`, `setuptools<81`
- Patch `torchvision.transforms.functional_tensor` shim (removed in tv 0.17+)
- Patch `.cuda()` → `.cpu()` in `s3diff_tile.py`, `model.py`, `my_utils/devices.py`

Full setup in `scripts/setup_and_trace_vae.sh`.

**Key gotcha — compile OOM at CFG B=2**: trn2.3xlarge has 124 GB system RAM; `neuronx-cc` was killed compiling SD-Turbo UNet at CFG batch=2. Compile at B=1 and do CFG pos/neg as two separate Python-level calls to the traced UNet. 64 GB swap helps but B=1 is required.

## Cost summary

- GPU: ~$587 (3-day unexpected runtime)
- Trn2 capacity block × 2: $17 + $17 = $34
- **Total: ~$621**

## Status

- Phase 1 (v1, UNet only): complete, 9-cell matrix populated
- Phase 2 (v2, UNet + VAE): 1K done; 2K stdout only; 4K lost; **needs another block to finish**
- Phase 3 (fix tile seam via `de_mod` refactor): designed, not started

---

## Phase 3 update (2026-04-29)

Full re-run with a natural cat photo on all 3 devices. Added attribute-routing
UNet NEFF so the compiled graph is image-agnostic (Phase 1/2 NEFF was tied to
the LQ image used at compile time). Added traced VAE encoder + decoder.

**Phase 3 Trn2 v3 on cat image, BF16, N=10:**

| Res | Steady | PSNR vs CPU | Seam luma (GPU: 1-3) |
|---|---|---|---|
| 1K | 24.91s | 24.55 dB | 39.0 |
| 2K | 82.67s | — | 33.3 |
| 4K | 360.10s | — | 18.6 |

See `phase3/README.md` for full matrix + GPU cat benchmark.
