#!/usr/bin/env python3
"""FLUX.1-dev Neuron benchmark on trn2.3xlarge (1 Trainium2, LNC=2 → 4 logical cores).

Adjusted for world_size=4 (vs world_size=8 on trn2.48xlarge).
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

CKPT_DIR = "/home/ubuntu/models/FLUX.1-dev/"
COMPILE_DIR = "/home/ubuntu/flux1_alien_neff_w4"
OUT_DIR = "/home/ubuntu/flux1_alien_out"
PROMPT = "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"
STEPS = 28
SEEDS = list(range(42, 52))
GUIDANCE = 3.5
HEIGHT, WIDTH = 1024, 1024
WORLD = 4
BACKBONE_TP = 4
WARMUP_ROUNDS = 3

os.makedirs(OUT_DIR, exist_ok=True)
dtype = torch.bfloat16

clip_cfg  = CLIPInferenceConfig(neuron_config=NeuronConfig(tp_degree=1, world_size=WORLD, torch_dtype=dtype),
                                load_config=load_pretrained_config(os.path.join(CKPT_DIR, "text_encoder")))
t5_cfg    = T5InferenceConfig(neuron_config=NeuronConfig(tp_degree=WORLD, world_size=WORLD, torch_dtype=dtype),
                              load_config=load_pretrained_config(os.path.join(CKPT_DIR, "text_encoder_2")))
backbone_cfg = FluxBackboneInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=BACKBONE_TP, world_size=WORLD, torch_dtype=dtype),
    load_config=load_diffusers_config(os.path.join(CKPT_DIR, "transformer")),
    height=HEIGHT, width=WIDTH,
)
decoder_cfg = VAEDecoderInferenceConfig(
    neuron_config=NeuronConfig(tp_degree=1, world_size=WORLD, torch_dtype=dtype),
    load_config=load_diffusers_config(os.path.join(CKPT_DIR, "vae")),
    height=HEIGHT, width=WIDTH,
    transformer_in_channels=backbone_cfg.in_channels,
)
setattr(backbone_cfg, "vae_scale_factor", decoder_cfg.vae_scale_factor)

print(f"[init] world={WORLD} backbone_tp={BACKBONE_TP}")
flux_app = NeuronFluxApplication(
    model_path=CKPT_DIR,
    text_encoder_config=clip_cfg,
    text_encoder2_config=t5_cfg,
    backbone_config=backbone_cfg,
    decoder_config=decoder_cfg,
    height=HEIGHT, width=WIDTH,
)

print(f"[compile] target dir: {COMPILE_DIR}")
t0 = time.time()
flux_app.compile(COMPILE_DIR)
compile_s = time.time() - t0
print(f"[compile] done in {compile_s:.1f}s")

t0 = time.time()
flux_app.load(COMPILE_DIR)
load_s = time.time() - t0
print(f"[load] {load_s:.1f}s")

print(f"[warmup] {WARMUP_ROUNDS} rounds @ {STEPS} steps")
for i in range(WARMUP_ROUNDS):
    t = time.time()
    _ = flux_app(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 guidance_scale=GUIDANCE, num_inference_steps=STEPS)
    print(f"  warmup {i+1}/{WARMUP_ROUNDS}: {time.time()-t:.2f}s")

runs = []
for seed in SEEDS:
    t = time.time()
    out = flux_app(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                   guidance_scale=GUIDANCE, num_inference_steps=STEPS,
                   generator=torch.Generator("cpu").manual_seed(seed))
    e2e = time.time() - t
    img = out.images[0]
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    std_v = float(arr.std())
    png = os.path.join(OUT_DIR, f"seed{seed}_alien.png")
    img.save(png)
    runs.append({"seed": seed, "time_s": round(e2e, 2), "std": round(std_v, 2), "png": png})
    print(f"[seed {seed}] {e2e:.2f}s std={std_v:.2f}")

summary = {
    "device": "Neuron trn2.3xlarge (WORLD=4, backbone_tp=4, LNC=2)",
    "model": "FLUX.1-dev BF16",
    "prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "compile_s": round(compile_s, 1), "load_s": round(load_s, 1),
    "mean_s": round(sum(r["time_s"] for r in runs)/len(runs), 2),
    "runs": runs,
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s over {len(runs)} seeds")
