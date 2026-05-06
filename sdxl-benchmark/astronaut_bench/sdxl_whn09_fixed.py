"""SDXL whn09 fork benchmark on trn2.3xlarge — adapted for our 10-seed astronaut bench.

Based on whn09 fork (NKI flash-attention kernel via attention_isa_kernel + KernelizedAttnProcessor2_0)
with pipeline structure from the AWS inf2 notebook (UNetWrap/NeuronUNet/DataParallel).
"""
import os
import json
import time
import math
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa
from torch_neuronx.xla_impl.ops import nki_jit  # noqa

# ==================== whn09 kernelized attention ====================
_flash_fwd_call = nki_jit()(attention_isa_kernel)

def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))
    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)
    _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


class KernelizedAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            diffusers.utils.deprecate("scale", "1.0.0",
                                      "scale arg is deprecated and ignored.")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attention_mask is not None or query.shape[3] > query.shape[2] or query.shape[3] > 128 or value.shape[2] == 77:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = attention_wrapper_without_swap(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


def custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
        diffusers.models.attention_processor.AttnProcessor2_0.__call__ = KernelizedAttnProcessor2_0.__call__

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(text_embeds=out_tuple[0], last_hidden_state=out_tuple[1], hidden_states=out_tuple[2])


class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        return self.text_encoder(text_input_ids, output_hidden_states=True, return_dict=False)


# ==================== config ====================
COMPILER_WORKDIR_ROOT = '/home/ubuntu/sdxl_whn09_compile'
MODEL_ID = "/home/ubuntu/models/sdxl-base"
PROMPT = "An astronaut riding a green horse"
HEIGHT, WIDTH = 1024, 1024
STEPS = 50
GUIDANCE = 7.5
N_SEEDS = 10
RESULTS_DIR = "/home/ubuntu/sdxl_whn09_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
text_encoder_2_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')


def compile_all():
    print("=" * 60, flush=True); print("COMPILE STAGE", flush=True); print("=" * 60, flush=True)

    # --- Text encoders ---
    if not (os.path.exists(text_encoder_filename) and os.path.exists(text_encoder_2_filename)):
        print("[compile] text encoders...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        traceable_text_encoder = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
        traceable_text_encoder_2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))
        del pipe

        # dummy input_ids with correct length (77 for SDXL)
        text_input_ids_1 = torch.tensor([[49406] + [49407] * 76])
        text_input_ids_2 = torch.tensor([[49406] + [0] * 76])

        neuron_text_encoder = torch_neuronx.trace(
            traceable_text_encoder, text_input_ids_1,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
            compiler_args=["--lnc=2"],
        )
        torch.jit.save(neuron_text_encoder, text_encoder_filename)

        neuron_text_encoder_2 = torch_neuronx.trace(
            traceable_text_encoder_2, text_input_ids_2,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder_2'),
            compiler_args=["--lnc=2"],
        )
        torch.jit.save(neuron_text_encoder_2, text_encoder_2_filename)
        del traceable_text_encoder, traceable_text_encoder_2, neuron_text_encoder, neuron_text_encoder_2

    # --- VAE decoder ---
    if not os.path.exists(decoder_filename):
        print("[compile] vae decoder...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        decoder = copy.deepcopy(pipe.vae.decoder)
        del pipe
        decoder_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
        decoder_neuron = torch_neuronx.trace(
            decoder, decoder_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
            compiler_args=["--model-type=unet-inference", "--lnc=2"],
        )
        torch.jit.save(decoder_neuron, decoder_filename)
        del decoder, decoder_neuron

    # --- UNet ---
    if not os.path.exists(unet_filename):
        print("[compile] UNet (largest, expect ~20-30 min)...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        Attention.get_attention_scores = get_attention_scores_neuron
        pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
        unet = copy.deepcopy(pipe.unet.unetwrap)
        del pipe
        sample_1b = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
        timestep_1b = torch.tensor(999).float().expand((1,))
        encoder_hidden_states_1b = torch.randn([1, 77, 2048], dtype=torch.bfloat16)
        text_embeds = torch.randn([1, 1280], dtype=torch.bfloat16)
        time_ids = torch.randn([1, 6], dtype=torch.bfloat16)
        example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, text_embeds, time_ids)
        unet_neuron = torch_neuronx.trace(
            unet, example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
            compiler_args=["--model-type=unet-inference", "--lnc=2"],
        )
        torch.jit.save(unet_neuron, unet_filename)
        del unet, unet_neuron

    # --- VAE post_quant_conv ---
    if not os.path.exists(post_quant_conv_filename):
        print("[compile] vae post_quant_conv...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
        del pipe
        post_quant_conv_in = torch.randn([1, 4, 128, 128], dtype=torch.bfloat16)
        post_quant_conv_neuron = torch_neuronx.trace(
            post_quant_conv, post_quant_conv_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
            compiler_args=["--lnc=2"],
        )
        torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)
        del post_quant_conv, post_quant_conv_neuron

    print("[compile] all 5 NEFFs present:", flush=True)
    for p in [text_encoder_filename, text_encoder_2_filename, decoder_filename, unet_filename, post_quant_conv_filename]:
        print(f"    {os.path.exists(p)}  {p}", flush=True)


def bench():
    print("=" * 60, flush=True); print("BENCH STAGE", flush=True); print("=" * 60, flush=True)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0, 1]  # 2-core DataParallel
    pipe.unet.unetwrap = torch_neuronx.DataParallel(
        torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    pipe.text_encoder = TextEncoderOutputWrapper(torch.jit.load(text_encoder_filename), pipe.text_encoder)
    pipe.text_encoder_2 = TextEncoderOutputWrapper(torch.jit.load(text_encoder_2_filename), pipe.text_encoder_2)

    # warmup (not timed)
    print("[bench] warmup...", flush=True)
    t0 = time.time()
    _ = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
             guidance_scale=GUIDANCE,
             generator=torch.Generator("cpu").manual_seed(9999)).images[0]
    print(f"[bench] warmup took {time.time()-t0:.2f}s", flush=True)

    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        img = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        elapsed = time.time() - t0
        per_seed.append(elapsed)
        out_png = os.path.join(RESULTS_DIR, f"seed{seed}.png")
        img.save(out_png)
        print(f"[bench] seed={seed}  time={elapsed:.3f}s  -> {out_png}", flush=True)

    times = np.array(per_seed)
    mean_s = float(times.mean())
    std_s = float(times.std(ddof=0))
    min_s = float(times.min())
    max_s = float(times.max())
    results = {
        "config": {
            "model_id": MODEL_ID,
            "prompt": PROMPT,
            "height": HEIGHT, "width": WIDTH,
            "steps": STEPS, "guidance_scale": GUIDANCE,
            "n_seeds": N_SEEDS, "seeds": SEEDS, "device_ids": device_ids,
            "fork": "whn09",
            "instance": "trn2.3xlarge",
            "sdk": "2.29",
        },
        "per_seed_s": per_seed,
        "mean_s": mean_s, "std_s": std_s, "min_s": min_s, "max_s": max_s,
    }
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 60, flush=True)
    print(f"[bench] mean={mean_s:.3f}s  std={std_s:.3f}s  min={min_s:.3f}s  max={max_s:.3f}s", flush=True)
    print(f"[bench] results saved -> {results_path}", flush=True)


if __name__ == "__main__":
    compile_all()
    bench()
