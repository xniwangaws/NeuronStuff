#!/usr/bin/env python3
"""Unified FLUX.1-dev benchmark on Neuron - steps 15, 25, 50."""
import os
import time
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
BASE_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir/"
world_size = 8
backbone_tp_degree = 4
dtype = torch.bfloat16
height, width = 1024, 1024
guidance_scale = 3.5
PROMPT = "A cat holding a sign that says hello world"
WARMUP_ROUNDS = 5
STEPS_LIST = [15, 25, 50]

# Component paths
text_encoder_path = os.path.join(CKPT_DIR, "text_encoder")
text_encoder_2_path = os.path.join(CKPT_DIR, "text_encoder_2")
backbone_path = os.path.join(CKPT_DIR, "transformer")
vae_decoder_path = os.path.join(CKPT_DIR, "vae")

# Configs
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

backbone_neuron_config = NeuronConfig(tp_degree=backbone_tp_degree, world_size=world_size, torch_dtype=dtype)
backbone_config = FluxBackboneInferenceConfig(
    neuron_config=backbone_neuron_config,
    load_config=load_diffusers_config(backbone_path),
    height=height, width=width,
)

decoder_neuron_config = NeuronConfig(tp_degree=1, world_size=world_size, torch_dtype=dtype)
decoder_config = VAEDecoderInferenceConfig(
    neuron_config=decoder_neuron_config,
    load_config=load_diffusers_config(vae_decoder_path),
    height=height, width=width,
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
    height=height, width=width,
)

print("Compiling...")
compile_start = time.time()
flux_app.compile(BASE_COMPILE_WORK_DIR)
compile_time = time.time() - compile_start
print(f"Compile time: {compile_time:.2f}s")

print("Loading...")
flux_app.load(BASE_COMPILE_WORK_DIR)

results = {}
for steps in STEPS_LIST:
    print(f"\nsteps={steps}: warming up ({WARMUP_ROUNDS} rounds)...")
    for i in range(WARMUP_ROUNDS):
        _ = flux_app(
            prompt=PROMPT,
            height=height, width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
        print(f"  Warmup {i+1}/{WARMUP_ROUNDS}")

    start = time.time()
    output = flux_app(
        prompt=PROMPT,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    )
    elapsed = time.time() - start
    results[steps] = elapsed
    output.images[0].save(f"/home/ubuntu/flux_unified_steps{steps}.png")
    print(f"  steps={steps}: {elapsed:.2f}s")

print(f"\n{'='*50}")
print(f"FLUX.1-dev Unified Neuron Benchmark")
print(f"{'='*50}")
print(f"world_size={world_size}, backbone_tp={backbone_tp_degree}")
print(f"dtype=bfloat16, warmup={WARMUP_ROUNDS}")
for steps in STEPS_LIST:
    print(f"  steps={steps:>2}: {results[steps]:.2f}s")
print(f"{'='*50}")
