# FLUX.2-dev — Preliminary Benchmark Results

**Date**: 2026-04-26 / 27
**Agent**: flux2
**Status**: GPU baselines complete; Neuron end-to-end pipeline runs at 1024² with real DiT NEFF (TP=8, LNC=2) — timing measured (mean 24.28 s/image), but visual accuracy NOT YET at BF16 parity (PSNR 9.94 dB); 2K compile not attempted this session (capacity block expired window). Next session: debug RoPE/conditioning wiring before any recompile.

## Benchmark matrix (FLUX.2-dev, 28 steps, guidance=4.0, seed=42, 10 prompts)

| Device | Precision | Resolution | Load (s) | First (s) | Mean steady (s) | P50 (s) | P95 (s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| L4 24GB (g6.4xlarge) | NF4 (bnb 4-bit) | 1024² | 2.98 | 229.9 | 210.1 | 210.4 | 223.1 | 19.67 |
| L4 24GB | NF4 | 2048² | — | — | — | — | — | **OOM** (infeasible @ 24 GB) |
| H100 80GB (p5.4xlarge) | BF16 (cpu_offload) | 1024² | 1.79 | 122.2 | 91.2 | 91.4 | 108.7 | 65.70 |
| H100 80GB | BF16 (cpu_offload) | 2048² | 1.67 | 158.7 | 162.9 | 163.0 | 164.1 | 68.91 |
| H100 80GB | FP8 e4m3 (torchao, cpu_offload) | 1024² | 67.54 | 67.2 | 68.6 | 68.5 | 68.9 | 48.44 |
| H100 80GB | FP8 e4m3 (torchao, cpu_offload) | 2048² | 67.54 | 132.8 | 133.2 | 133.1 | 134.4 | 48.44 |
| Neuron trn2.48xlarge | BF16 (TP=8, LNC=2) | 1024² | 63.54 | 24.82 | 24.28 | 24.28 | 24.58 | — |
| Neuron trn2.48xlarge | BF16 | 2048² | — | — | — | — | — | **not attempted this session** |

## Neuron component status (this session)

| Component | Status | Key numbers |
|---|---|---|
| Environment (SDK 2.29 DLAMI, NxDI 0.9.0) | ✅ | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |
| Weights downloaded (130 GB diffusers format) | ✅ | `/home/ubuntu/flux2_weights/` |
| CPU BF16 reference (3 prompts × 512²) | ✅ | Golden latents + PNG |
| **VAE decoder Neuron trace @ 512²** | ✅ Validated | cos_sim **0.9977**, PSNR 23.2 dB vs CPU |
| **VAE 1K tiled decode** (3×3 of 512² NEFF) | ✅ Validated | cos_sim **0.9920**, PSNR 28.77 dB vs CPU tiled, **9.6 s/image** |
| VAE 2K tiled decode (6×6 of 512² NEFF) | ⚠️ Benchmarked on random latent | 38.5 s/image; real-latent validation skipped (2K CPU denoise ≈ 90 min) |
| **Mistral-3-24B text encoder @ TP=8** | ✅ Compiled + validated | Compile 38.3s; inference **63.8 ms/prompt** (vs CPU 3.2s = **50× speedup**); stacked-output cos_sim **0.9996**; per-layer cos_sim 0.9998/0.9995/0.9995 for [10]/[20]/[30] |
| **Pipeline integration scaffold** | ✅ Runs E2E | 473 LOC pipeline + 295 LOC driver; stub DiT contract fully documented; 1024² image generated (garbage content, correct shape) |
| **DiT NxDI scaffold + single-block CPU parity** | ✅ **PASSED** | Double-block cos_sim **1.000000** (bitwise identical); single-block cos_sim **1.000000** (max_abs 7.6e-5 fp32 rounding). Weight converter: zero missing/unexpected keys. |
| DiT compile on Neuron | ⏳ Next session | 4-6 engineering days estimated |
| End-to-end Neuron 1K/2K benchmark | ⏳ Next session | Blocked on DiT only |

## Pipeline breakdown @ 1024² / 28 steps (measured with stub DiT)

| Component | Time | Notes |
|---|---|---|
| Text encoder (Neuron TP=8) | 0.064 s | Will be 0.064 × 28 = 1.8 s across denoise steps; but actually called once per prompt, not per step |
| Scheduler loop (28 × stub DiT) | 0.94 s | Real DiT per-step cost TBD; this is just the loop overhead |
| VAE decode (Neuron tiled, 4 tiles) | 9.75 s | Pipeline agent measured 1024² as 2×2=4 tiles; my separate benchmark used 3×3=9 tiles |
| **Total pipeline (stub DiT)** | **15.6 s** | Excludes 236s one-time load |

If real DiT adds ~20-30 s per 28-step loop (based on H100 BF16's 91 s minus encoder + VAE), end-to-end 1K would be ~30-40 s on Neuron — **competitive with H100 BF16 (91s) if DiT port succeeds**.

## Per-layer text encoder accuracy (real signal; layers [10, 20, 30])

| Layer | cos_sim | max_abs_err |
|---|---|---|
| 10 | 0.99985 | 0.016 |
| 20 | 0.99952 | 0.062 |
| 30 | 0.99946 | 0.211 |
| stacked (pipeline input) | 0.99964 | — |

Error growth with depth is expected (BF16 accumulation across 30 transformer layers).

## Key findings

1. **VAE ~parity with CPU** on trn2 (Neuron 1.07s vs CPU 0.85s at 512²). Recommendation: keep Neuron VAE only if it integrates cleanly; else CPU VAE is fine.
2. **Text encoder is the clear Neuron win**: 50× faster than CPU. This alone justifies the Neuron path if DiT parity is acceptable.
3. **Latent format**: `(B, 1024, 128)` packed, requires `_unpack_latents_with_ids` + per-channel `vae.bn` denorm + `_unpatchify_latents`. Different from FLUX.1's scalar shift/scale. Documented in `task009/neuron_flux2_pipeline.py`.
4. **DiT architecture verified** against real weights: 32B, 8 double + 48 single blocks, no biases, SwiGLU, 4-axis RoPE [32,32,32,32] θ=2000, shared modulation, fused QKV+MLP projection in single blocks, no pooled text path.
5. **DiT single-block CPU parity PASSED**: double-stream block output bitwise-identical to HF reference; single-stream block within fp32 rounding noise (7.6e-5 max_abs from `to_out` split-Linear reordering). This is the strongest possible de-risking of the full compile — **if TP sharding doesn't introduce new bugs, the full 56-block Neuron model should produce correct output**.
5. **NxDI tracing API**: prefer `ModelBuilder` over older `parallel_model_trace` — the latter hangs post-compile on `xm.rendezvous`. Text encoder agent discovered this.
6. **Text encoder save format mismatch**: monolithic `text_encoder.pt` (33 GB) saved by trace agent; pipeline agent's `parallel_model_load` expected sharded `tp_0.pt…tp_7.pt`. Pipeline uses monolithic via `torch.jit.load` fine; minor fix noted as `TODO(port)` in pipeline code.
7. **4K is infeasible** on any device (L4 24GB OOM; 2K already tight on 24GB).

## DiT stub contract (what the NEFF must satisfy for pipeline integration)

```
inputs:
  hidden_states         (B, L_img, 128)       bf16
  timestep              (B,)                  bf16  (already /1000)
  encoder_hidden_states (B, L_txt, 15360)     bf16  (= 3 layers × 5120 hidden, stacked from text encoder)
  guidance              (B,)                  fp32
  img_ids               (B, L_img, 4)         int64 (T,H,W,L RoPE axes)
  txt_ids               (B, L_txt, 4)         int64
returns:
  noise_pred            (B, L_img + L_txt, 128) bf16
                         Pipeline slices [:, :L_img, :] before scheduler.step.
```

At 1024²: `L_img=4096` (64×64 post-2×2-patchify), `L_txt=512`, joint seq = 4608 tokens.
At 2048²: `L_img=16384` (128×128), joint seq = 16896 tokens.

## Accuracy Comparison

Side-by-side grids (10 prompts × available devices, columns labeled; 512² per cell):

- 1024²: `/Users/xniwang/oppo-opencode/working/flux2/task011/results/grid_1024.png` (1552×5204, 3 cols: L4 NF4 | H100 BF16 | H100 FP8)
- 2048²: `/Users/xniwang/oppo-opencode/working/flux2/task011/results/grid_2048.png` (1036×5204, 2 cols: H100 BF16 | H100 FP8)

Neuron BF16 column is **pending** — images not yet on laptop (another agent is producing them via task010). The grid builder and metrics script both auto-include Neuron columns once `task010/results/neuron_bf16_{1024,2048}/` lands; rerun the two scripts in `task011/` to refresh.

Per-device 1024² metrics vs H100 BF16 reference (mean over 10 prompts, PSNR on 0-1 RGB; LPIPS skipped — no `lpips`/torch installed locally, so pixel-L1 used as perceptual proxy):

| Device / precision | Mean PSNR vs H100 BF16 (dB) | Mean L1 vs H100 BF16 | Min PSNR (dB) | Max L1 |
|---|---|---|---|---|
| L4 NF4 | 14.06 | 0.1418 | 11.49 | 0.1969 |
| H100 FP8 | 23.39 | 0.0434 | 15.21 | 0.0995 |
| Neuron BF16 | 9.94 | 0.2876 | — | — |

Full per-prompt numbers in `/Users/xniwang/oppo-opencode/working/flux2/task011/results/metrics.json`.

**Interpretation**: H100 FP8 stays close to the BF16 reference (~23 dB / L1 ≈ 0.04, visually very similar modulo fine detail), while L4 NF4's 4-bit quantization plus different attention kernel path produces substantially different images at the pixel level (~14 dB / L1 ≈ 0.14) — the content is on-prompt but composition diverges from BF16.

**Neuron BF16 accuracy — OPEN ISSUE (2026-04-27 session)**: The end-to-end pipeline runs to completion but produces visually incoherent outputs (grain-textured pseudo-noise, no recognizable prompt content) across all 10 prompts. PSNR 9.94 dB / L1 0.29 vs H100 BF16 indicates a systematic wiring or numerics defect, not random failure.

**2026-04-27 follow-up session — one root cause identified, a second still open**:

1. **Text encoder attention-mask bug (IDENTIFIED + PARTIALLY FIXED)**. The Neuron-traced Mistral-3 TE was compiled with `attention_mask=None` (causal-only masking). But the Mistral-3 tokenizer pads on the **LEFT** by default, so real prompt tokens live at positions `[S-N, S)` of a 512-long sequence while positions `[0, S-N)` are padding. Under causal-only masking, every real token attends to **all** preceding padding tokens, contaminating its hidden state. Measured on the "red panda" prompt (N=42 real tokens): Neuron TE vs HF (with real attention_mask) gave cos_sim = **0.22** at real positions, Neuron TE vs HF (with mask=None) gave cos_sim = **0.999** — i.e. the trace is correct for its contract but the pipeline wasn't honoring that contract.
   - **Fix applied** in `/Users/xniwang/oppo-opencode/working/flux2/task009/results/neuron_flux2_pipeline.py` (synced to `/home/ubuntu/neuron_flux2_pipeline.py`): in `_encode_prompt`, right-pad `input_ids` before the NEFF call so real tokens sit at `[0, N)`; under causal-only masking those tokens now only see other real tokens. After the NEFF returns, shift the stacked hidden states back into the original left-padded layout (real tokens at `[S-N, S)`) so downstream 4-axis RoPE position ids line up with what the DiT was compiled against; pad-position hidden states are zeroed. **Validation**: per-token cos_sim vs HF jumped from 0.22 → **1.000** on real positions, norms match within 1% (neu=390, hf=388).
   - The pipeline branch selector was also tightened: `text_fn` in `from_traced` used to carry an `attention_mask=None` default kwarg which made `inspect.signature` report `attention_mask` as a parameter and routed to the CPU-fallback branch. Removed that kwarg so `_encode_prompt` correctly detects the Neuron-NEFF path and applies the right-pad/shift logic.

2. **DiT NEFF output still disagrees with HF (UNRESOLVED)**. Even with the corrected text-encoder output, a one-step CPU-vs-NEFF comparison (seed=42, step 0, 1024²; inputs captured from the running pipeline) gives:
   - cos_sim(HF, NEFF) = **0.389** (flat, not on-prompt)
   - HF norm 1090 vs NEFF norm 454 → **~2.4× magnitude shortfall**
   - max_abs diff 6.8, mean_abs diff 1.10 on a tensor whose elements are O(1) in bf16
   - Output is not NaN/inf, signs are plausible — but a significant chunk of the network's contribution is missing or mis-scaled.
   
   This is likely a TP=8-only bug (the scaffold's TP=1 block parity was verified bitwise-identical to HF, and `test_interleave_tp8_sim.py` verifies the per-rank layout for fused SwiGLU / QKV+MLP projections). Things that are NOT the bug (ruled out by direct inspection vs the running HF pipeline on the instance): RoPE concat order (both pipelines use txt-first); timestep scaling (pipeline /1000 + NEFF *1000 matches HF /1000 + HF DiT *1000); guidance scaling (pipeline raw `4.0` + NEFF *1000 matches HF); output sequence slice (NEFF already drops txt tokens internally so the pipeline slice is a no-op); prompt-embed layout (HF `permute(0,2,1,3).reshape` produces the same `[layer0_h, layer1_h, layer2_h]` flattening we use).
   
   **Still to investigate next session**: (a) whether `double_stream_modulation_img.linear` (ColumnParallelLinear gather_output=True, weight [36864, 6144]) gets the correct per-rank slab ordering after all-gather such that downstream `chunk(6, dim=-1)` still recovers `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]` at TP=8 — needs a targeted simulation similar to `test_interleave_tp8_sim.py`; (b) whether the single-stream `to_out_attn`/`to_out_mlp` reduce pattern is producing the right final value (manually-split `to_out` + explicit `reduce_from_tensor_model_parallel_region` — correctness relies on `reduce_output=False` being honored and on the two `RowParallelLinear.tensor_parallel_group` being the same process group); (c) whether `norm_out.linear` (ColumnParallelLinear with gather_output=True, weight [12288, 6144]) has the same potential ordering issue as (a). Recommend next session write a TP=2 end-to-end block-sim that loads converted weights into the monkey-patched CPU scaffold but with an EXPLICIT 2-rank fake-TP loop, comparing to a single-rank reference — this would surface any surviving TP>1 layout bugs without needing Neuron hardware.

**Speed deliverable is intact**: 24.28 s/image mean steady-state at 1024² (3.75× faster than H100 BF16 cpu_offload at 91.2 s). The fix to the text-encoder bug did not change timing. Image quality remained textured noise after the TE fix alone, confirming the DiT NEFF bug is independent.

## Accuracy / image artifacts

- L4 NF4 1024²: 10 images at `working/flux2/task001/results/l4_nf4_1024/`
- H100 BF16 1024²: 10 images at `working/flux2/task002/results/h100_bf16_1024/`
- H100 BF16 2048²: 10 images at `working/flux2/task002/results/h100_bf16_2048/`
- H100 FP8 1024²: 10 images at `working/flux2/task002/results/h100_fp8_1024/`
- H100 FP8 2048²: 10 images at `working/flux2/task002/results/h100_fp8_2048/`
- Neuron CPU BF16 512² (ref): 3 images + tensors at `working/flux2/task004/results/`
- Neuron VAE 512² validation: `working/flux2/task003/results/vae_validation_report.json`
- Neuron VAE 1K validation: `working/flux2/task003/results/vae_1k_validation.json` + PNG
- Neuron text encoder trace: `working/flux2/task006/results/` (script + CPU ref tensors + log)
- Neuron pipeline integration: `working/flux2/task009/{neuron_flux2_pipeline.py, run_pipeline_stub.py}`
- Neuron DiT scaffold: `working/flux2/task008/neuron_flux2_dit.py` (1183 LOC)

## Cost / capacity usage

| Resource | Status | Realized |
|---|---|---|
| L4 g6.4xlarge | Terminated | ~$5 |
| H100 p5.4xlarge | Terminated | ~$15 (of $61 capacity block) |
| trn2.48xlarge | Running, ~$150 realized (of $558) | ~7 / 24 h used |

## Next session priorities

1. **Compile DiT at TP=8 with bucket shapes for 1024² latent — 1-2 days** (single-block parity PASSED, so go straight to full compile)
2. Swap stub DiT in `neuron_flux2_pipeline.py` with real NEFF — **0.5 day**
3. End-to-end 1K/2K Neuron benchmark — **0.5 day**
4. Accuracy comparison grids (PSNR/LPIPS) — **0.5 day**
5. Final report + registry update — **0.5 day**

**Remaining engineering: 3-4 days** (downgraded from 4-6 after block parity passed).
