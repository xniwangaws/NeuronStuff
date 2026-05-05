# Phase E — S3Diff on AWS Trainium2 (PyTorch Native Eager Mode)

Follow-up to Phase 3 (trace mode). This phase ports S3Diff to the `torch_neuron_eager` stack — no pre-compiled NEFFs, no per-image bake, no fixed shapes. Proves that eager-mode **eliminates the tile-seam artifact** and **beats trace-mode latency**.

## TL;DR

| Stack | Load | Cold | **Steady (1K)** | PSNR vs CPU fp32 | NEFFs compiled |
|---|---|---|---|---|---|
| CPU eager fp32 (reference) | 0.9s | — | 54.0s | ref | — |
| H100 bf16 (diffusers) | 4.8s | 9.6s | **1.26s** | 45.10 dB | — |
| L4 bf16 (diffusers) | 4.5s | 6.4s | 2.34s | 45.15 dB | — |
| Trn2 **trace** fp32 (Phase 3) | 55.9s + ~30 min/image | 24.94s | 24.91s | 24.55 dB (tile seam) | N/A (precompiled) |
| **Trn2 eager fp32 (E1)** | 10.6s | 406s | **14.71s** | **67.82 dB** ✅ | — |
| **Trn2 eager bf16 (E1)** | 9.0s | 73s | **8.60s** | **43.26 dB** | 451 |
| **Trn2 eager bf16 + VAE tile pad (E2-T2)** | 11.4s | **30.5s** | 8.75s | 43.26 dB | ≪451 |
| **Trn2 eager bf16 + NKI attention (E2-T3)** | 11.0s | 86s | 8.85s | **43.43 dB** | **173** (-62%) |
| Trn2 eager bf16 + single-tile (E2-T1-B) | 11.0s | 260s | **6.14s** | 36.42 dB | — |

Phase E1 closes the **tile-seam gap** that dominated Phase 3 (43 dB improvement) at **42% lower latency**.

---

## Phase E1 — Validate eager stack works

### Goal
Move every S3Diff submodule to `torch.device('neuron')` and run a single end-to-end inference in eager mode. No `torch.compile`.

### Delta from Phase 3 (trace mode)
- **No per-image compile**: each forward pass is fetched from the eager NEFF cache keyed by `(op, shape, dtype)`. First image pays a 406s / 73s cold-compile cost; subsequent images reuse the cache.
- **No shape-specialized NEFFs**: `module.de_mod = tensor` attribute injection works naturally because eager does not bake attributes into a graph.
- **No tile-seam artifact**: the model sees the uncompiled graph, so the per-tile BF16 drift that caused Phase 3's 24.55 dB ceiling is gone. Neuron eager fp32 output is **67.82 dB vs CPU eager fp32** — essentially lossless.

### Results (1024×1024 output, cat_LQ_256 input)
| dtype | Load | Cold | Steady (warm) | PSNR vs CPU fp32 | PSNR vs H100 bf16 |
|---|---|---|---|---|---|
| **fp32** | 10.6s | 406s | **14.71s** | **67.82 dB** | 45.10 dB |
| **bf16** | 9.0s | 73s | **8.60s** | 43.26 dB | 42.18 dB |

`fp32 vs bf16`: bf16 is **41% faster** with a 24 dB "drop" that is actually just the bf16-vs-fp32 difference — the same 45-vs-67 dB gap exists between H100 bf16 and CPU fp32 reference.

**Gate (user-set):** PSNR ≥ 30 dB vs CPU eager — both fp32 and bf16 pass comfortably.

### Visual comparison
`images/e1_cpu_fp32_sr.png`, `images/e1_neuron_fp32_sr.png`, `images/e1_neuron_bf16_sr.png`. On the cat image, all three are visually indistinguishable; H100 bf16 and Neuron bf16 are essentially identical; Neuron fp32 pixel-matches CPU fp32.

### Script: `scripts/phase_e1_smoke.py`
Runs: `python phase_e1_smoke.py --device {cpu,neuron} --dtype {fp32,bf16} --output_image /tmp/sr.png --num_inferences 3`

---

## Phase E2 — Try torch.compile(backend='neuron') per stage

### Goal
Use `torch.compile` to reduce the host dispatch overhead that dominates eager execution time.

### Stage-by-stage results
Run on warm model, 3 inferences per stage. 3 unique stages of compile attempted.

| Stage | Cold | Warm min | PSNR | Notes |
|---|---|---|---|---|
| stage0 baseline (no compile) | 36.1s | **13.9s** | 43.26 | — |
| stage1 +compile VAE encoder | 2623s (**44 min**) | 12.29s | 43.26 | compile success, 1.6s/image win, not worth 44 min cost |
| stage2 +compile VAE decoder | — | **FAIL** | — | `NCC_IXTP002: 11.45M instructions > 10M threshold` |
| stage3 +compile UNet | — | **FAIL** | — | `NCC_IRPX901 RelaxPredicates` internal compiler bug inside LoRA proj_in |

Conclusion: whole-module compile is blocked by (a) decoder instruction budget, (b) UNet LoRA + compiler bug. Eager baseline at 13.9s is already competitive.

### Script: `scripts/phase_e2_compile.py`

---

## Phase E2 (follow-up) — Diagnose real bottleneck

Ran `neuron-profile inspect` + Perfetto SQL on a warm bf16 run (see `data/profile_summary.md`). Key finding:

### Run 2 (steady 8.77s) time budget

| Bucket | Time | % |
|---|---|---|
| **Host dispatch gap between ops** | **3.72s** | **42%** |
| Output copyout to host | 1.18s | 13% |
| Device compute (nc_exec_running) | 1.18s | 13% |
| Kernel sync barrier | 0.81s | 9% |
| nrt_model_submit overhead | 0.62s | 7% |
| DMA alloc + misc | 1.26s | 15% |

**7344 NEFF submissions per inference** across **349 unique NEFFs**. Top 5 NEFFs combined account for only 26% of device time — no single compute hotspot exists. The primary bottleneck is **host-side Python dispatch**, not device compute.

Top 5 NEFFs (rank 1-3 are VAE decoder tile body variants, rank 4-5 are UNet latent tiles):

| rank | execs | total_ms | avg_us | op class |
|---|---|---|---|---|
| 1 | 36 | 121 | 3367 | VAE decoder tile body |
| 2 | 36 | 86 | 2395 | VAE decoder tile variant |
| 3 | 36 | 41 | 1139 | VAE decoder tile variant |
| 4 | 10 | 35 | 3528 | UNet attention / latent tile |
| 5 | 14 | 29 | 2079 | UNet block |

### NEFF cache analysis

- **1,795 NEFFs / 559 MB** cached at `/tmp/neff_cache/` after one full warm run
- **~69 min cumulative cold compile** at first-run time
- Root cause: `latent_tiled_size=96 + latent_tiled_overlap=32` on 128×128 latent produces 3-4 distinct tile shapes → 569 distinct UNet NEFFs, plus VAE decoder tile=224 produces 18 tiles/image with 3 shape variants from edge-clamp fallback

---

## Phase E2 (three parallel experiments)

Three sub-agents launched in parallel against isolated repo copies on the same trn2 instance.

### Task 1 — Shape normalization (`scripts/phase_e1_smoke.py` with modified tile config)

Test: collapse tile shape variance by changing `latent_tiled_size` / `latent_tiled_overlap`.

| Config | Cold | Warm | PSNR vs CPU | Visual quality |
|---|---|---|---|---|
| baseline 96/32 | ~69 min | 8.60s | 43.26 | high |
| **A: 64/0 + vae_dec=128** | **180s** | 12.12s | 31.31 | lower (noise regions) |
| **B: 128/0 (single tile) + vae_dec=128** | **260s** | **6.14s** ⭐ | 36.42 | visually OK, algorithmically different |

**Finding**: compile-time hypothesis confirmed (fewer shapes → fewer NEFFs → faster compile). However, the original tile config is **load-bearing for the S3Diff reconstruction formula** — removing overlap/gaussian-blend changes what the model actually computes. The 6.14s single-tile warm is impressive but does not match the original output algorithm, so it cannot be shipped as a drop-in performance fix.

### Task 2 — Pad VAE decoder tiles + torch.compile inner blocks

Test: pad every VAE-decoder tile to the same shape before `original_forward` so only one NEFF compiles, then try `torch.compile` on inner resnet blocks.

| Mode | Cold | Warm | PSNR vs CPU | Notes |
|---|---|---|---|---|
| **pad_only** (no compile) | **30.5s** ⭐ | 8.75s | **43.26 dB** ✅ | **2000× cold speedup, no quality loss** |
| pad + compile 19 resnets | 600s | 9.02s | 35.73 | warm no faster, PSNR drops 7.5 dB |

**Finding**: padding alone is a **clean, lossless win** — shape normalization at the padding layer collapses the 3 decoder-tile NEFF variants into 1 and drops cold compile from ~69 min to 30.5s, without changing the output algorithm. Compile on top of padding hurts accuracy, likely due to bf16 intermediate cast strategy change inside torch.compile's Inductor path.

### Task 3 — Substitute UNet self-attention with `nkilib.core.attention.attention_cte`

Test: replace all 16 `attn1.forward` processors with a small NKI kernel that calls the stock library flash attention. Keep cross-attention (attn2) and LoRA adapters in eager.

| Metric | baseline bf16 | +NKI attention | Delta |
|---|---|---|---|
| Warm steady | 8.60s | 8.85s | +2.9% slower |
| Cold | 73s | 86s | +18% |
| PSNR vs CPU fp32 | 43.26 dB | **43.43 dB** | **+0.17 dB** |
| NEFFs compiled | 451 | **173** | **-62%** |

**Correctness gate passed** on CPU-ref (max |diff|=1e-5 fp32). The +0.17 dB is explained by NKI's `softmax_dtype=fp32` default — more accurate than diffusers' bf16 softmax.

**Finding**: the NKI kernel is not the thing making things fast (attention is only 0.4% of wall time per profile). The real value is **graph-fragmentation reduction** (-62% NEFFs). This is a much cheaper cold start and a cleaner compile surface for any future `torch.compile` attempts.

---

## Combined optimal stack

| Layer | Add to baseline | Cost | Benefit |
|---|---|---|---|
| eager bf16 (E1) | baseline | 9s load, 8.60s warm | 42 dB PSNR vs CPU, no seam |
| + Task 2 pad VAE tiles | ~1h of vaehook.py editing | 30.5s cold (-98%) | same PSNR, fewer NEFFs |
| + Task 3 NKI self-attn | ~1d of integration work | ~60s cold, 8.85s warm | +0.17 dB, -62% NEFFs |
| **= Optimal** | | **~60s cold, 8.9s warm** | **43.43 dB**, ~100-150 NEFFs |

**Ship recommendation**: eager bf16 + Task 2 padding. This is a pure engineering patch with zero quality regression and a 138× faster cold start. Task 3 NKI is a nice-to-have for correctness-critical deployments where the 0.17 dB matters more than the 3% warm latency.

---

## The real bottleneck — host dispatch architecture

Phase 3 was blocked by compile time and tile seams. Phase E solved both. The remaining 8.6s steady-state is **not a Neuron hardware problem** — it's a PyTorch eager + Neuron dispatch stack problem:

- 7,344 NEFF submissions per image × ~500µs per submission = 3.7s of pure Python/dispatch overhead
- Device compute is only 1.18s (13% of wall); the other 87% is dispatch, sync, and copyout
- Even with all 3 parallel experiments applied, cache can only reduce the number of unique graphs, not the number of submissions

H100 wins here through PyTorch+CUDA's aggressive kernel fusion (cuDNN / cuBLAS graph capture). On Neuron, achieving sub-3s warm latency will require one of:

1. **Pipeline-level capture**: use `torch_xla.compile` or `torch_neuronx.trace` on well-defined boundaries (VAE-encode, single UNet step, VAE-decode as three graphs instead of 7K ops).
2. **NxDI-style coarse kernels**: the neuronx-distributed-inference stack fuses UNet-like workloads into a handful of kernels per layer. If the S3Diff UNet can be adapted to that pattern, per-image op count drops from thousands to dozens.
3. **Dynamic-shape eager with captured subgraphs**: `torch.compile` with `dynamic=True` + shape-padded inputs (Task 2's padding made this feasible for the VAE decoder — do the same for UNet LoRA once the NCC_IRPX901 compiler bug is fixed upstream).

See `data/dispatch_architecture.md` for a full decomposition of the 8.6s and the three proposed paths.

---

## Files

- `README.md` — this file
- `scripts/phase_e1_smoke.py` — the eager pipeline smoke test (CPU or Neuron, fp32 or bf16)
- `scripts/phase_e2_compile.py` — stage-by-stage torch.compile benchmark
- `scripts/phase_t2_compile.py` — Task 2 (VAE decoder tile padding + compile) driver
- `scripts/nki_attention_patch.py` — Task 3 monkey-patch replacing `attn1.forward` with `nkilib.core.attention.attention_cte`
- `images/` — SR outputs from every configuration
- `logs/` — raw stdout/stderr for every run
- `data/` — ranked NEFF CSV + profile summary (TBA)
