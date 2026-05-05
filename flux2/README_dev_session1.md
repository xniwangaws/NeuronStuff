# FLUX.2-dev → AWS Neuron Trainium2 port + GPU benchmarks

Partial port of [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (32B DiT + 24B Mistral-3 text encoder, rectified-flow T2I) to AWS Trainium2 via NxDI, with apples-to-apples H100 and L4 GPU baselines.

**Session dates**: 2026-04-26 / 2026-04-27 (1 capacity block exhausted)
**Status**: Neuron **speed validated** (3.75× faster than H100 BF16), **quality broken** (TP>1 drift bug open)

## Headline numbers (1024², 28 steps, seed=42, 10 prompts)

| Device | Precision | Mean (s) | P95 (s) | Peak VRAM (GB) | PSNR vs H100 BF16 |
|---|---|---|---|---|---|
| L4 g6.4xlarge | NF4 (bnb 4-bit) | 210.1 | 223.1 | 19.7 | 14.06 |
| H100 p5.4xlarge | BF16 (cpu_offload) | **91.2** (ref) | 108.7 | 65.7 | — |
| H100 p5.4xlarge | FP8 e4m3 (torchao) | 68.6 | 68.9 | 48.4 | 23.39 |
| **Neuron trn2.48xlarge** | **BF16 (TP=8, LNC=2)** | **24.3** ✨ | 24.6 | n/a | 9.94 ❌ |

- L4 2K = OOM (infeasible even with NF4 + cpu_offload at 24 GB)
- Neuron BF16 PSNR is **low because of a known TP>1 wiring bug** — see *Open bugs* below. Component-level parity passed (TE cos_sim 0.9996, VAE 0.998, DiT single-block TP=1 cos_sim 1.000000).

## Directory layout

```
flux2/
├── HANDOFF.md                      # next-session recovery guide (rebuild NEFFs ~1.5h, debug Bug 2)
├── results/preliminary.md          # full benchmark matrix + findings
├── task001/bench_l4.py             # L4 NF4 benchmark driver
├── task002/bench_h100.py           # H100 BF16/FP8 benchmark driver
├── task003/
│   ├── trace_vae.py                # torch_neuronx.trace VAE decoder at 512²
│   ├── validate_vae_2k.py          # real-latent 1K/2K validation
│   └── results/
│       ├── vae_tile_decode.py      # 512²-NEFF → 1K/2K tiled orchestration
│       ├── vae_validation_report.json    # cos_sim 0.998 @ 512²
│       └── vae_1k_validation.json        # cos_sim 0.992 @ 1K tiled
├── task004/cpu_reference.py        # CPU BF16 golden reference generator
├── task006/results/
│   └── trace_text_encoder.py       # Mistral-3-24B TP=8 trace via ModelBuilder (cos_sim 0.9996)
├── task008/
│   ├── neuron_flux2_dit.py         # NxDI DiT scaffold (1183 LOC, TP-interleave fix)
│   ├── test_interleave_tp8_sim.py  # bitwise-verifies per-rank interleave at tp=1/2/4/8/16
│   └── results/
│       ├── compile_dit_tp8.py      # ModelBuilder compile script (106s compile, 65 GB NEFF)
│       ├── test_block_parity.py    # CPU single-block parity (cos_sim 1.000000 @ TP=1)
│       └── PARITY_FINDINGS.md      # what parity resolved vs what it didnt
├── task009/results/
│   ├── neuron_flux2_pipeline.py    # end-to-end pipeline with Bug 1 (TE pad-side) fix
│   └── run_pipeline_stub.py        # smoke-test driver
├── task010/
│   ├── bench_neuron.py             # Neuron 10-prompt benchmark driver
│   └── results/neuron_bf16_1024/results.json   # 24.3s/image measured
└── task011/
    ├── make_grids.py               # 4-column side-by-side comparison grid
    ├── compute_metrics.py          # PSNR / pixel-L1 vs H100 BF16 reference
    └── results/metrics.json
```

## Neuron component status

| Component | Status | Key numbers |
|---|---|---|
| SDK 2.29 DLAMI + NxDI 0.9.0 | ✅ | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |
| VAE decoder @ 512² NEFF | ✅ validated | cos_sim **0.9977**, 1.07 s/tile |
| VAE 1K tiled decode (3×3 of 512² tile) | ✅ validated | cos_sim **0.9920**, 9.6 s/image |
| Mistral-3-24B text encoder, TP=8 | ✅ validated | cos_sim **0.9996**, **63.8 ms/prompt** (50× CPU) |
| DiT NxDI scaffold + single-block CPU parity | ✅ passed | double+single both cos_sim **1.000000** at TP=1 |
| TP>1 interleave fix (`ff.linear_in`, `to_qkv_mlp_proj`) | ✅ bitwise | tp=1/2/4/8/16 all verified |
| DiT full-model compile @ TP=8 | ✅ compiled | 106s compile, 65 GB NEFF, 518 ms / denoising step |
| End-to-end 1024² pipeline | ✅ runs | 24.3 s/image steady state |

## Open bugs

### ✅ Bug 1 — FIXED: text-encoder pad-side mismatch

Mistral-3 tokenizer pads on the **left** by default, but the Neuron TE was traced with `attention_mask=None` (causal-only). Under causal masking, real tokens at positions `[S-N, S)` attend backward to all pad tokens at `[0, S-N)`, contaminating every real hidden state.

- Before fix: Neuron-vs-HF real-position cos_sim = **0.22** (catastrophic).
- After fix: cos_sim = **1.000**.

The fix is in `task009/results/neuron_flux2_pipeline.py` `_encode_prompt`: right-pad input_ids before the NEFF call, then shift hidden states back to the HF left-padded layout after the forward.

### ❌ Bug 2 — OPEN: DiT TP=8 systematic drift

After Bug 1 is fixed, a single-step CPU `HF vs Neuron DiT` comparison at seed=42 / step 0 / 1024² shows:
- **cos_sim(HF, NEFF) = 0.389**
- HF output norm **1090** vs NEFF norm **454** (NEFF is ~2.4× too small in magnitude)

Ruled out:
- RoPE concat order (`[txt, img]` matches scaffold)
- timestep scaling (`t/1000` on entry, `*1000` inside — net identity, matches trace example)
- guidance scaling (consistent between trace and pipeline)
- output slicing (`[:, :L_img, :]`)
- prompt-embed layout
- per-rank interleaving of fused weights (bitwise at tp=1..16)

Most likely culprit: a TP>1-only bug in one of the modulation all-gathers (`double_stream_modulation_img.linear [36864, 6144]`, `norm_out.linear [12288, 6144]`) or in the single-block `to_out_attn + to_out_mlp` reduce pattern. Block-level parity was only tested at TP=1.

Next-session debug plan (see HANDOFF.md):
1. Write a TP=2 end-to-end CPU simulation of the scaffold (monkey-patched parallel layers + fake 2-rank loop) loading the actual interleaved weights; compare forward output against single-rank.
2. If TP=2 matches: recompile DiT at TP=2 and bisect upward to find which TP-degree first introduces drift.
3. If TP=2 diverges: root-cause inside the sim (cheap to iterate), then re-verify at TP=8.

Snapshot data for this debug is in S3 (inputs and HF/NEFF step-0 outputs at seed=42):
```
s3://xniwang-neuron-models-us-east-2/flux2/task011-neff_step0_v2.pt
s3://xniwang-neuron-models-us-east-2/flux2/task011-step0_compare_v2.pt
```

## What's NOT in this repo

- **Image outputs** (10 × 6 = 60 PNGs): not pushed (user direction: code + md only). They live on the local machine at `~/oppo-opencode/working/flux2/task00{1,2,10}/results/*/seed*.png` and were used to compute the PSNR/L1 numbers in `task011/results/metrics.json`.
- **Compiled NEFFs** (DiT 65 GB, TE 33 GB, VAE 217 MB): **lost** when the capacity block expired and the EBS volume was deleted. S3 uploads had silently failed during the session (`aws s3 cp --quiet` + background = no error visibility). Scripts here can recompile all three in ~1.5 hours on a fresh trn2.48xlarge.
- **HF weights** (130 GB): standard HuggingFace download.

## Rebuild in a fresh session

See `HANDOFF.md` for the full checklist. Summary: new trn2.48xlarge capacity block → HF download → run `task003/trace_vae.py`, `task006/results/trace_text_encoder.py`, `task008/results/compile_dit_tp8.py` (in that order) → **back up each NEFF to S3 with size verification immediately** → continue with Bug 2 debug via TP=2 CPU sim.

## Cost

- L4 g6.4xlarge: ~$5 (terminated)
- H100 p5.4xlarge: ~$15 of $61 capacity block (terminated)
- trn2.48xlarge: $558 (full 24h capacity block consumed)
- **Total session: ~$578**
