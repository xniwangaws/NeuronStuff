# FLUX.2-dev Neuron port — Next Session Handoff

**Last session**: 2026-04-26 / 27
**Agent**: flux2
**Instance state**: **trn2.48xlarge terminated** (capacity block `cr-00a67d76f08995183` expired; EBS volume deleted with `DeleteOnTermination=true`)

## What's fully done (captured in local git repo)

### GPU baselines (5 cells complete, terminated)

| Device | Precision | 1024² mean (s) | 2048² mean (s) | Peak VRAM (GB) |
|---|---|---|---|---|
| L4 g6.4xlarge | NF4 | 210.1 | OOM | 19.7 |
| H100 p5.4xlarge | BF16 cpu_offload | 91.2 | 162.9 | 66-69 |
| H100 p5.4xlarge | FP8 e4m3 | 68.6 | 133.2 | 48.4 |

### Neuron components (all validated, all compiled)

| Component | Speed | Accuracy | Size |
|---|---|---|---|
| VAE decoder @ 512² (tiled to 1K/2K) | 1.07s / 9.6s / 38.5s | cos_sim 0.998 (real) / 0.992 (1K tiled) | 217 MB |
| Mistral-3-24B text encoder TP=8 | **63.8 ms/prompt** (50× CPU) | cos_sim 0.9996 (stacked) | 33 GB |
| DiT TP=8, 56 blocks | **518 ms / denoising step** | cos_sim 1.000000 (single-block CPU parity at TP=1) | 65 GB |
| End-to-end 1K BF16 pipeline | **24.3 s/image** (3.75× H100 BF16) | — | — |

### Code artifacts (local git, not lost)

All in `/Users/xniwang/oppo-opencode/working/flux2/`:
- `task001/bench_l4.py` — L4 NF4 benchmark
- `task002/bench_h100.py` — H100 benchmark
- `task003/results/validate_vae.py`, `vae_tile_decode.py`, `validate_vae_2k.py` — VAE validation + tiled decode orchestrator
- `task004/cpu_reference.py` — CPU BF16 golden reference
- `task006/results/trace_text_encoder.py` — Text encoder TP=8 trace (uses ModelBuilder, NOT parallel_model_trace)
- `task008/neuron_flux2_dit.py` — DiT NxDI scaffold (1183 LOC, parity-validated at TP=1, has TP>1 interleave fix)
- `task008/results/compile_dit_tp8.py` — DiT compile script (ModelBuilder, 106s compile time)
- `task008/results/test_block_parity.py`, `PARITY_FINDINGS.md` — CPU single-block parity test + findings
- `task008/test_interleave_tp8_sim.py` — TP>1 interleave verification (tested tp=1,2,4,8,16 all bitwise)
- `task009/results/neuron_flux2_pipeline.py` — pipeline with Bug 1 (TE left-pad) fix applied, 473 LOC
- `task009/results/run_pipeline_stub.py` — smoke test driver
- `task010/bench_neuron.py` — Neuron 10-prompt benchmark driver
- `task010/results/neuron_bf16_1024/` — 10 output PNGs (visually broken due to Bug 2) + results.json
- `task011/make_grids.py`, `compute_metrics.py` — grid / PSNR scripts
- `task011/results/grid_1024.png`, `grid_2048.png`, `metrics.json` — 4-col accuracy grid
- `results/preliminary.md` — full report with both bugs + current numbers

### ⚠️ Lost artifacts (NEFFs, must be recompiled next session)

The following were ONLY on the terminated EBS volume — **S3 backups failed silently**:
- `/home/ubuntu/dit_traced/dit_tp8_1024.pt` (65 GB) — DiT NEFF
- `/home/ubuntu/text_encoder_traced/text_encoder.pt` (33 GB) — TE NEFF
- `/home/ubuntu/vae_traced/vae_decoder_512.pt` (217 MB) — VAE NEFF
- `/home/ubuntu/flux2_weights/` (130 GB) — HF weights (re-downloadable)
- `/home/ubuntu/cpu_reference/image_p0{0,1,2}.pt` + PNGs — CPU golden

**What survived in S3** (only 22 MB total):
```
s3://xniwang-neuron-models-us-east-2/flux2/
├── task011-neuron_flux2_pipeline.py      (36 KB) ← Bug1-fixed pipeline (also in local git)
├── task011-neff_step0_v2.pt              (17 MB) ← step 0 inputs snapshot — useful for next-session bug 2 debug
├── task011-step0_compare_v2.pt           (2 MB)  ← HF vs NEFF step 0 output diff
└── task011-seed0042_p00_after_te_fix.png (2 MB)  ← smoke image (still noise but from TE-fixed path)
```

### Total rebuild time next session

| Step | Time |
|---|---|
| Launch trn2.48xlarge on new capacity block | 30 min purchase + 5 min launch |
| Download FLUX.2 weights (130 GB) | 15-20 min |
| Re-trace VAE at 512² | ~20 min |
| Re-trace Mistral-3 text encoder TP=8 | ~1 min compile + ~30 min save |
| Re-compile DiT TP=8 | ~2 min compile + ~2 min save |
| CPU reference (3 prompts @ 512²) | ~20 min |
| **Full re-bootstrap** | **~1.5-2 hours** |

## The 2 bugs discovered (both documented, one fixed)

### ✅ Bug 1: Text encoder pad-side mismatch — FIXED

Mistral-3 tokenizer left-pads by default. Neuron TE was traced with `attention_mask=None` (causal-only). Under causal, real tokens at `[S-N, S)` attend back to all pad tokens at `[0, S-N)`, contaminating every hidden state.
- **Before fix**: real-position cos_sim vs HF = **0.22** (catastrophic)
- **After fix** (right-pad input, shift hidden states back to left-pad layout): cos_sim = **1.000** ✅

Fix is in `task009/results/neuron_flux2_pipeline.py` `_encode_prompt`.

### ❌ Bug 2: DiT TP=8 systematic drift — OPEN

After Bug 1 fix, one-step CPU HF vs Neuron DiT comparison at seed=42 / step 0 / 1024²:
- cos_sim(HF, NEFF) = **0.389**
- HF output norm **1090** vs NEFF norm **454** (NEFF is 2.4× too small)

Output is finite, on-range, but clearly missing contribution from several paths.

**Ruled out**: RoPE concat order, timestep scaling, guidance scaling, output slicing, prompt-embed layout, interleave_fused correctness (tested bitwise tp=1,2,4,8,16).

**Most likely culprit**: a TP=8-specific bug in:
- `double_stream_modulation_img.linear [36864, 6144]` all-gather (not unit-tested at TP>1)
- `norm_out.linear [12288, 6144]` all-gather (not unit-tested at TP>1)
- single-block `to_out_attn + to_out_mlp` reduce pattern

Block-level parity was only verified at **TP=1**.

**Next-session debug plan**:
1. Write TP=2 end-to-end CPU simulation of scaffold (monkey-patched parallel layers + fake 2-rank loop) loading actual interleaved weights; compare forward output vs single-rank reference
2. If TP=2 sim matches single-rank: recompile DiT at TP=2 and bisect upward to find which TP-degree first introduces drift
3. If TP=2 sim diverges: root-cause inside the sim (cheap to iterate), then re-verify at TP=8

Use snapshotted debug inputs/outputs from S3:
```bash
aws s3 cp s3://xniwang-neuron-models-us-east-2/flux2/task011-neff_step0_v2.pt .
aws s3 cp s3://xniwang-neuron-models-us-east-2/flux2/task011-step0_compare_v2.pt .
```

## Cost summary

| Resource | Status | Realized cost |
|---|---|---|
| L4 g6.4xlarge | Terminated | ~$5 |
| H100 p5.4xlarge | Terminated | ~$15 of $61 capacity block |
| trn2.48xlarge | Terminated 2026-04-27 10:15 UTC | **~$558 of $558** capacity block consumed |

**Total session**: ~$578

## Backup policy for next session

Lesson learned: **all S3 `aws s3 cp --quiet` calls silently failed**. Next time:
1. Never use `--quiet` on critical uploads
2. Always verify with `aws s3 ls s3://bucket/key --human-readable` and compare size
3. Upload NEFFs immediately after they land on the instance — don't wait for end of session
4. For files > 1 GB, use `aws s3api head-object` to verify ContentLength matches local `stat`

See `~/.claude/projects/.../memory/s3_backup_lesson.md` for the full lesson.

## Quick-start for next session

```bash
# 1. Buy a fresh trn2.48xlarge capacity block in us-east-2
aws ec2 describe-capacity-block-offerings --region us-east-2 \
  --instance-type trn2.48xlarge --instance-count 1 --capacity-duration-hours 24

# 2. Launch + SSH (see `projects/flux2.md` for AMI/subnet/key details)

# 3. Bootstrap in parallel
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
hf auth login --token $(cat ~/.hf_token)

# Download weights (skip monolithic flux2-dev.safetensors, we only need diffusers format)
hf download black-forest-labs/FLUX.2-dev --local-dir ~/flux2_weights \
  --include 'transformer/*' 'text_encoder/*' 'tokenizer/*' 'vae/*' 'ae.safetensors' \
             'model_index.json' 'scheduler/*'

# 4. Copy all code artifacts
scp -r /Users/xniwang/oppo-opencode/working/flux2/task00{3,4,6,8,9,10}/ ubuntu@<IP>:~/

# 5. BACKUP NEFFS IMMEDIATELY AFTER EACH COMPILE (don't wait)
python trace_vae.py --resolution 512
aws s3 cp ~/vae_traced/vae_decoder_512.pt s3://xniwang-neuron-models-us-east-2/flux2/
aws s3 ls s3://xniwang-neuron-models-us-east-2/flux2/vae_decoder_512.pt --human-readable

python trace_text_encoder.py --trace-only --tp 8 --seq-len 512
aws s3 cp ~/text_encoder_traced/text_encoder.pt s3://xniwang-neuron-models-us-east-2/flux2/
aws s3 ls s3://xniwang-neuron-models-us-east-2/flux2/text_encoder.pt --human-readable

NEURON_RT_VISIBLE_CORES=0-15 python compile_dit_tp8.py
aws s3 cp ~/dit_traced/dit_tp8_1024.pt s3://xniwang-neuron-models-us-east-2/flux2/
aws s3 ls s3://xniwang-neuron-models-us-east-2/flux2/dit_tp8_1024.pt --human-readable

# 6. Now start Bug 2 debug — step 0 snapshot is already in S3
aws s3 cp s3://xniwang-neuron-models-us-east-2/flux2/task011-neff_step0_v2.pt .
aws s3 cp s3://xniwang-neuron-models-us-east-2/flux2/task011-step0_compare_v2.pt .
# then TP=2 CPU simulation, etc.
```
