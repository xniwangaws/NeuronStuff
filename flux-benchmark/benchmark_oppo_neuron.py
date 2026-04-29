#!/usr/bin/env python3
"""FLUX.1-dev benchmark on Neuron (Trn2) per OPPO spec.

Phases reported per customer email (Wang Xiaoning 2026-04-24):
  compile_s         - NxDI compile (one-time, cacheable; excluded from "load")
  load_s            - load compiled artifacts + weights onto NeuronCores
  first_inference_s - run[0], cold start
  steady_mean_s     - mean(run[1..5])  == (total - first) / (N-1)
"""
import csv
import json
import os
import time

import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.application import NeuronFluxApplication
from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import CLIPInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import FluxBackboneInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import T5InferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import VAEDecoderInferenceConfig
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

CKPT_DIR = "/home/ubuntu/models/FLUX.1-dev/"
BASE_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir/"
OUTPUT_DIR = "/home/ubuntu/outputs/trn2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = (
    "A close-up image of a green alien with fluorescent skin in the middle "
    "of a dark purple forest"
)
STEPS = 28
SEED = 0
HEIGHT, WIDTH = 1024, 1024
GUIDANCE_SCALE = 3.5
N_RUNS = 6  # run[0] = first/cold; mean(run[1..5]) = steady

world_size = 8
backbone_tp_degree = 4
dtype = torch.bfloat16

text_encoder_path = os.path.join(CKPT_DIR, "text_encoder")
text_encoder_2_path = os.path.join(CKPT_DIR, "text_encoder_2")
backbone_path = os.path.join(CKPT_DIR, "transformer")
vae_decoder_path = os.path.join(CKPT_DIR, "vae")

clip_neuron_config = NeuronConfig(tp_degree=1, world_size=world_size, torch_dtype=dtype)
clip_config = CLIPInferenceConfig(
    neuron_config=clip_neuron_config,
    load_config=load_pretrained_config(text_encoder_path),
)
t5_neuron_config = NeuronConfig(tp_degree=world_size, world_size=world_size, torch_dtype=dtype)
t5_config = T5InferenceConfig(
    neuron_config=t5_neuron_config,
    load_config=load_pretrained_config(text_encoder_2_path),
)
backbone_neuron_config = NeuronConfig(
    tp_degree=backbone_tp_degree, world_size=world_size, torch_dtype=dtype
)
backbone_config = FluxBackboneInferenceConfig(
    neuron_config=backbone_neuron_config,
    load_config=load_diffusers_config(backbone_path),
    height=HEIGHT,
    width=WIDTH,
)
decoder_neuron_config = NeuronConfig(tp_degree=1, world_size=world_size, torch_dtype=dtype)
decoder_config = VAEDecoderInferenceConfig(
    neuron_config=decoder_neuron_config,
    load_config=load_diffusers_config(vae_decoder_path),
    height=HEIGHT,
    width=WIDTH,
    transformer_in_channels=backbone_config.in_channels,
)
setattr(backbone_config, "vae_scale_factor", decoder_config.vae_scale_factor)

print("Initializing NeuronFluxApplication...")
flux_app = NeuronFluxApplication(
    model_path=CKPT_DIR,
    text_encoder_config=clip_config,
    text_encoder2_config=t5_config,
    backbone_config=backbone_config,
    decoder_config=decoder_config,
    height=HEIGHT,
    width=WIDTH,
)

print("Phase A: compile")
t0 = time.time()
flux_app.compile(BASE_COMPILE_WORK_DIR)
compile_s = time.time() - t0
print(f"  compile_s = {compile_s:.2f}")

print("Phase B: load")
t0 = time.time()
flux_app.load(BASE_COMPILE_WORK_DIR)
load_s = time.time() - t0
print(f"  load_s = {load_s:.2f}")

runs = []
for i in range(N_RUNS):
    t0 = time.time()
    output = flux_app(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=STEPS,
    )
    dt = time.time() - t0
    runs.append(dt)
    print(f"  run[{i}] = {dt:.2f}s")
    if i == 0:
        output.images[0].save(os.path.join(OUTPUT_DIR, f"flux1_step{STEPS}_seed{SEED}_first.png"))
    elif i == 1:
        output.images[0].save(os.path.join(OUTPUT_DIR, f"flux1_step{STEPS}_seed{SEED}.png"))

first_inference_s = runs[0]
steady_runs = runs[1:]
steady_mean_s = sum(steady_runs) / len(steady_runs)

summary = {
    "instance": "trn2.48xlarge",
    "accelerator": "2x Trainium2 (tp=4, cp=2)",
    "dtype": "bfloat16",
    "prompt": PROMPT,
    "steps": STEPS,
    "seed": SEED,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "guidance_scale": GUIDANCE_SCALE,
    "n_runs": N_RUNS,
    "compile_s": round(compile_s, 2),
    "load_s": round(load_s, 2),
    "first_inference_s": round(first_inference_s, 2),
    "steady_mean_s": round(steady_mean_s, 2),
    "steady_runs_s": [round(x, 2) for x in steady_runs],
}

with open(os.path.join(OUTPUT_DIR, "run.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "run.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "value_s"])
    w.writerow(["compile", f"{compile_s:.2f}"])
    w.writerow(["load", f"{load_s:.2f}"])
    w.writerow(["first_inference", f"{first_inference_s:.2f}"])
    w.writerow(["steady_mean", f"{steady_mean_s:.2f}"])
    for i, v in enumerate(runs):
        w.writerow([f"run_{i}", f"{v:.2f}"])

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
