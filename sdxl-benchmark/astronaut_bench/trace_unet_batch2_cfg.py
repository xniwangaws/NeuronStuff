"""Track A: batch=2 BF16 UNet compile for SDXL 1024 with CFG.

Based on trace_unet_only.py but all sample shapes doubled (batch=2) so CFG
(guidance=7.5) works without pipeline having to re-trace. NEFF goes to a fresh
dir so we don't overwrite the batch=1 baseline.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import copy, math, time, resource
from typing import Optional

import torch, torch.nn as nn, torch.nn.functional as F
import torch_neuronx
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit

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
    return attn_output.reshape((bs, n_head, q_len, d_head))


class KernelizedAttnProcessor2_0:
    def __init__(self):
        assert hasattr(F, "scaled_dot_product_attention")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            diffusers.utils.deprecate("scale", "1.0.0", "ignored")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
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
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
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
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        return self.unet(sample, timestep, encoder_hidden_states,
                         added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                         return_dict=False)


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.config.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
        diffusers.models.attention_processor.AttnProcessor2_0.__call__ = KernelizedAttnProcessor2_0.__call__

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond=None,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    COMPILER_WORKDIR_ROOT = os.path.expanduser("~/sdxl_compile_b2_1024")
    model_id = "/home/ubuntu/models/sdxl-base"
    os.makedirs(COMPILER_WORKDIR_ROOT, exist_ok=True)

    timings = {}

    # ---- UNet (batch=2 for CFG) ----
    print(f"[rss] start: {rss_gb():.2f} GB", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    Attention.get_attention_scores = get_attention_scores_neuron
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    # batch=2 inputs (pipeline CFG duplicates uncond+cond -> batch=2)
    sample_b2 = torch.randn([2, 4, 128, 128])
    timestep_0d = torch.tensor(999).float()  # 0-dim; wrapper expands to (B,)
    encoder_hidden_states_b2 = torch.randn([2, 77, 2048])
    added_cond_kwargs_b2 = {
        "text_embeds": torch.randn([2, 1280]),
        "time_ids": torch.randn([2, 6]),
    }
    example_inputs = (
        sample_b2,
        timestep_0d,
        encoder_hidden_states_b2,
        added_cond_kwargs_b2["text_embeds"],
        added_cond_kwargs_b2["time_ids"],
    )

    print(f"[compile] unet batch=2 starting ... rss={rss_gb():.2f}GB", flush=True)
    t0 = time.time()
    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, "unet"),
        compiler_args=["--model-type=unet-inference", "--auto-cast", "matmult", "--auto-cast-type", "bf16"],
    )
    timings["unet"] = time.time() - t0
    torch.jit.save(unet_neuron, os.path.join(COMPILER_WORKDIR_ROOT, "unet/model.pt"))
    print(f"[compile] unet done in {timings['unet']:.1f}s  rss={rss_gb():.2f}GB", flush=True)
    del unet, unet_neuron

    print("=" * 50, flush=True)
    for k, v in timings.items():
        print(f"  {k:25s} {v:8.1f}s", flush=True)
    print("[done] NEFFs in", COMPILER_WORKDIR_ROOT, flush=True)


if __name__ == "__main__":
    main()
