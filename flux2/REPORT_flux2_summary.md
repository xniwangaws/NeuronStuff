# FLUX.2 klein 9B Benchmark

**Prompt** `"A cat holding a sign that says hello world"` · 50 steps · guidance 4.0 · 10 seeds (42–51) · [source](https://github.com/xniwangaws/NeuronStuff/tree/main/flux2)

## Results

| Device | 1K Mean | 1K $/img | 2K Mean | 2K $/img | 4K | Pass |
|---|---:|---:|---:|---:|---|---:|
| **Neuron trn2.3xl BF16 TP=4** | **42.90s** | **$0.02664** | **191.44s** | **$0.1189** | N/A | 10/10 |
| H100 BF16 | 24.10s | $0.02896 | 107.05s | $0.1286 | GRAY | 10/10 |
| H100 FP8 (torchao) | 21.18s | $0.02545 | 106.20s | $0.1276 | GRAY | 10/10 |
| L4 FP8 (BFL official + torchao shim) | 77.25s | $0.02839 | 385s (1-seed) | $0.1416 | OOM | 10/1 |

## Key findings

- **Neuron $/img 最低**:比 H100 BF16 便宜 8% (1K) / 7% (2K)。trn2.3xl $2.235/hr vs p5.4xl $4.326/hr
- **修复 PR #146 timestep bug**:`modeling_flux2_klein.py` line 247 `half_dim-1` → `half_dim`(FLUX.2 用 `downscale_freq_shift=0`)。DiT step-0 cos 0.9635→0.999936,R 33.1→0.989
- **HBM**:Neuron 24/40 GB (1K/2K),H100 BF16 37/45 GB,低 35%
- **L4 FP8**(BFL 官方 `FLUX.2-klein-9b-fp8` + 手写 torchao FP8 shim):1K 77s / 13GB VRAM
- **Capacity**:p5 全 region `InsufficientInstanceCapacity`,trn2 更容易锁

## Problems

- **4K 全平台废**:klein `max_area=4MP`,4K=16MP 超规格,GRAY 噪声。Neuron 另有 HLO gen timeout
- **Bug 定位路径**:CPU R-ratio + weight audit + block bisect + NKI kernel isolate 都 PASS,最后在 attention input 7% inflation 追到非 DiT 本体的 timestep sinusoidal scalar
- **PSNR 7-9 dB** 同 prompt 跨 stack 的正常 bf16 漂移;SSIM 0.37(修复前 0.25)
- **trn2 compile 可缓存**:1K 157s / 2K 795s,热启 45-300s
- **L4 FP8 via diffusers BLOCKED**:diffusers 0.38 `from_single_file` 在 BFL `FLUX.2-klein-9b-fp8` 的 `input_scale` 0-dim scalar 上 `torch.chunk` 崩;只能用 BFL flux2 repo + 手写 FP8 Linear shim (~80 LoC)

## Raw data

- `flux2/task015_klein_jim_pr146/results/klein_fixed_{1k,2k}_50step/` — 10 PNG + results.json
- `h100_{1024,2048}_{bf16,fp8}/` · `l4_{1024,2048}_{fp8,nf4,bf16_offload}/`
- `klein_fix_step0_metrics.json` — fix before/after R-ratio
- `klein_fix_pixel_diff.json` — PSNR/SSIM vs H100
- `l4_fp8_attempt/BLOCKED.md` — diffusers 0.38 FP8 `from_single_file` 崩溃 traceback
