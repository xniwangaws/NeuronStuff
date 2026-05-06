#!/usr/bin/env python3
"""FLUX.1-dev Neuron bench on trn2.3xlarge — resolution-parametric version.

Supports 1024 / 2048 / 4096. For 2K/4K, model is super-res (spec max_area=4MP),
so outputs >= 2K may be noise-quality (GRAY) — still useful for timing / HBM / BLOCKED tests.
"""
import argparse, json, os, time
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import NeuronFluxApplication
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import CLIPInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import T5InferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import FluxBackboneInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import VAEDecoderInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=1024, choices=[1024, 2048, 4096])
ap.add_argument("--ckpt", default="/home/ubuntu/FLUX.1-dev")
ap.add_argument("--compile_dir", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--guidance", type=float, default=3.5)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--world", type=int, default=4)
ap.add_argument("--backbone_tp", type=int, default=4)
args = ap.parse_args()

CKPT_DIR = args.ckpt.rstrip("/") + "/"
COMPILE_DIR = args.compile_dir or f"/home/ubuntu/flux1_alien_neff_w{args.world}_{args.resolution}"
OUT_DIR = args.out or f"/home/ubuntu/flux1_alien_out_{args.resolution}"
PROMPT = "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"
HEIGHT = WIDTH = args.resolution

os.makedirs(OUT_DIR, exist_ok=True)
dtype = torch.bfloat16

clip_cfg = CLIPInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=1, world_size=args.world, torch_dtype=dtype),
    load_config=load_pretrained_config(os.path.join(CKPT_DIR, "text_encoder")),
)
t5_cfg = T5InferenceConfig(
    neuron_config=NeuronConfig(tp_degree=args.world, world_size=args.world, torch_dtype=dtype),
    load_config=load_pretrained_config(os.path.join(CKPT_DIR, "text_encoder_2")),
)
backbone_cfg = FluxBackboneInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=args.backbone_tp, world_size=args.world, torch_dtype=dtype),
    load_config=load_diffusers_config(os.path.join(CKPT_DIR, "transformer")),
    height=HEIGHT, width=WIDTH,
)
decoder_cfg = VAEDecoderInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=1, world_size=args.world, torch_dtype=dtype),
    load_config=load_diffusers_config(os.path.join(CKPT_DIR, "vae")),
    height=HEIGHT, width=WIDTH,
    transformer_in_channels=backbone_cfg.in_channels,
)
setattr(backbone_cfg, "vae_scale_factor", decoder_cfg.vae_scale_factor)

print(f"[init] res={args.resolution} world={args.world} backbone_tp={args.backbone_tp}", flush=True)
flux_app = NeuronFluxApplication(
    model_path=CKPT_DIR,
    text_encoder_config=clip_cfg,
    text_encoder2_config=t5_cfg,
    backbone_config=backbone_cfg,
    decoder_config=decoder_cfg,
    height=HEIGHT, width=WIDTH,
)

print(f"[compile] target dir: {COMPILE_DIR}", flush=True)
t0 = time.time()
flux_app.compile(COMPILE_DIR)
compile_s = time.time() - t0
print(f"[compile] done in {compile_s:.1f}s", flush=True)

t0 = time.time()
flux_app.load(COMPILE_DIR)
load_s = time.time() - t0
print(f"[load] {load_s:.1f}s", flush=True)

print(f"[warmup] {args.warmup} rounds @ {args.steps} steps", flush=True)
for i in range(args.warmup):
    t = time.time()
    _ = flux_app(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 guidance_scale=args.guidance, num_inference_steps=args.steps)
    print(f"  warmup {i+1}/{args.warmup}: {time.time()-t:.2f}s", flush=True)

runs = []
for seed in args.seeds:
    t = time.time()
    out = flux_app(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                   guidance_scale=args.guidance, num_inference_steps=args.steps,
                   generator=torch.Generator("cpu").manual_seed(seed))
    e2e = time.time() - t
    img = out.images[0]
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    std_v = float(arr.std())
    png = os.path.join(OUT_DIR, f"seed{seed}_alien.png")
    img.save(png)
    runs.append({"seed": seed, "time_s": round(e2e, 2), "std": round(std_v, 2), "png": png})
    print(f"[seed {seed}] {e2e:.2f}s std={std_v:.2f}", flush=True)

summary = {
    "device": f"Neuron trn2.3xlarge (WORLD={args.world}, backbone_tp={args.backbone_tp}, LNC=2)",
    "model": "FLUX.1-dev BF16",
    "prompt": PROMPT, "steps": args.steps, "guidance": args.guidance,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "compile_s": round(compile_s, 1), "load_s": round(load_s, 1),
    "mean_s": round(sum(r["time_s"] for r in runs) / len(runs), 2) if runs else None,
    "n_seeds": len(runs),
    "runs": runs,
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s over {len(runs)} seeds", flush=True)
