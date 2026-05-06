# sdxl (code dir)

Code + harness for the SDXL-base-1.0 basic validation (trn2.3xlarge vs H100 vs L4, all BF16).

Registry: `/Users/xniwang/Neuron-steering-docs/projects/sdxl.md`
Management: `/Users/xniwang/oppo-opencode/projects/sdxl/`
Plan: `/Users/xniwang/.claude/plans/zany-honking-pumpkin.md`

## Files

| File | Purpose |
|------|---------|
| `prompts.json` | 5 fixed prompts + seed=0 (subset of AWS notebook's 8) |
| `trace_sdxl.py` | Compile 5 NEFF-wrapped TorchScript modules on trn2.3xlarge (migrated from `aws-neuron-samples` SDXL notebook) |
| `benchmark_neuron.py` | Load compiled modules, run 5 prompts × N steps, save PNGs + summary.csv |
| `benchmark_gpu.py` | Same harness for H100 (p5.4xlarge) and L4 (g6.4xlarge) via diffusers native BF16 |
| `make_grids.py` | Build 3-device side-by-side comparison grids for human judgment |
| `compute_metrics.py` | Informational PSNR (+ optional LPIPS) vs H100 — not a gate |
| `results/` | Per-device PNGs, summary.csv, grids, final Chinese report |

## Run order (end-to-end)

```bash
# 1. GPU baselines (can run in parallel)
# --- L4 on g6.4xlarge ---
python benchmark_gpu.py --device cuda:0 --dtype bf16 \
    --model /home/ubuntu/models/sdxl-base --prompts prompts.json \
    --steps 25 --out /home/ubuntu/sdxl_out_25 --device_label l4
python benchmark_gpu.py ... --steps 50 --out /home/ubuntu/sdxl_out_50 --device_label l4

# --- H100 on p5.4xlarge --- (same commands, different instance)

# 2. Neuron on trn2.3xlarge
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python trace_sdxl.py 2>&1 | tee compile.log       # ~15–30 min
python benchmark_neuron.py \
    --compile_dir /home/ubuntu/sdxl/compile_dir \
    --model /home/ubuntu/models/sdxl-base \
    --prompts prompts.json --steps 25 --out /home/ubuntu/sdxl_out_25
python benchmark_neuron.py ... --steps 50 --out /home/ubuntu/sdxl_out_50

# 3. scp outputs back to results/images/{l4,h100,neuron}/ and run locally:
python make_grids.py --root results --steps 25 50 --n_prompts 5
python compute_metrics.py --root results --steps 25 --n_prompts 5   # add --lpips if installed
```

## Customer principles (shape the harness)

1. Align to GPU (H100 BF16), not ground-truth
2. Save PNGs for human judgment — no pixel-level gating
3. Qwen-VL `image_url` chat payload does not apply to SDXL (text-to-image)

## Notebook migration deltas vs `aws-neuron-samples` commit f532a05

See header of `trace_sdxl.py` and `projects/sdxl/steering/context.md` for the 6 changes:
FP32→BF16, drop pinned diffusers/transformers, DataParallel 2→4 cores, absolute workdir,
`--auto-cast=matmult`, warmup=5 per-prompt timing.
