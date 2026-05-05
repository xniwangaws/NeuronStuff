"""
FLUX.2-dev NxDI scaffold — adapted from AWS PR #146 (FLUX.2-klein).

Architecture source: github.com/aws-neuron/neuronx-distributed-inference PR #146
(jimburtoft/contrib/flux2-klein). Author: AWS Neuron team.

Differences vs Klein (base config adjusted per /home/ubuntu/flux2_weights/transformer/config.json):
  - num_attention_heads:   24  -> 48     (Klein -> dev)
  - attention_head_dim:    128 -> 128
  - inner_dim = H*D:       3840? -> 6144
  - num_layers (double):    8  -> 8
  - num_single_layers:     24  -> 48
  - joint_attention_dim: Qwen3-8B -> 15360 (Mistral-3-24B stacked hidden)
  - mlp_ratio:             3.0 -> 3.0
  - rope_theta:            2000 -> 2000
  - axes_dims_rope:    (32,32,32,32) unchanged
  - guidance_embeds:       False (Klein) -> True (dev has guidance_embedder)

Forward signature is 5-arg (hidden_states, encoder_hidden_states, timestep, guidance,
image_rotary_emb) to match the existing /home/ubuntu/neuron_flux2_pipeline.py and
/home/ubuntu/compile_dit_tp8.py harnesses. Klein's 4-arg version simply drops
`guidance` — we keep it for dev.

Structural TP pattern (the whole point of using PR #146):
  - Single-stream block: 5 separate ColumnParallel (Q, K, V, mlp_gate, mlp_value),
    2 separate RowParallel (proj_out_attn, proj_out_mlp) with explicit all-reduce
    of their sum. This AVOIDS splitting SwiGLU gate|value across a TP partition
    boundary (the issue causing our prior v1 std=26 cloud).
  - Double-stream FF: linear_in split into linear_in_gate + linear_in_value.
  - convert_hf_to_neuron_state_dict splits the HF fused weights to match.
"""

import logging
import math
import os
from typing import List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed_inference.models.diffusers.embeddings import (
    FluxPosEmbed,
    apply_rotary_emb,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from nkilib.core.attention.attention_cte import attention_cte
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target
from neuronx_distributed_inference.models.application_base import (
    NeuronApplicationBase,
)
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.model_wrapper import (
    BaseModelInstance,
    ModelWrapper,
)

_HARDWARE = hardware(get_platform_target())
logger = logging.getLogger(__name__)


# ============================================================
# NKI Flash Attention Wrapper (verbatim from PR #146)
# ============================================================

def attention_wrapper_sharded(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.reshape((bs * n_head, q_len, d_head))
    k = key.reshape((bs * n_head, k_len, d_head))
    v = value.reshape((bs * n_head, v_len, d_head))

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded = vc_size == 2
    scale = 1 / math.sqrt(d_head)

    if use_sharded:
        attn_output = attention_cte[2](
            q, k, v, scale,
            causal_mask=False, tp_q=True, tp_k=True, tp_out=False,
        )
    else:
        attn_output = attention_cte(
            q, k, v, scale,
            causal_mask=False, tp_q=True, tp_k=True, tp_out=False,
        )
    return attn_output.reshape((bs, n_head, q_len, d_head))


# ============================================================
# Modulation (verbatim from PR #146)
# ============================================================

class NeuronFlux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 1, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.act = nn.SiLU()
        self.linear = ColumnParallelLinear(
            dim, 3 * mod_param_sets * dim,
            bias=False, gather_output=True, reduce_dtype=reduce_dtype,
        )

    def forward(self, temb):
        return self.linear(self.act(temb))

    @staticmethod
    def split(mod, n):
        chunks = mod.chunk(3 * n, dim=-1)
        groups = []
        for i in range(n):
            groups.append((chunks[3 * i], chunks[3 * i + 1], chunks[3 * i + 2]))
        return groups


# ============================================================
# Timestep + Guidance Embedding (dev has guidance; adapted from Klein)
# ============================================================

class _Embedder(nn.Module):
    """Container matching HF key naming `.linear_1 / .linear_2`."""
    def __init__(self, in_dim, out_dim, reduce_dtype):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            in_dim, out_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype,
        )
        self.linear_2 = RowParallelLinear(
            out_dim, out_dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype,
        )


class NeuronFlux2TimestepEmbedding(nn.Module):
    """Timestep + guidance embedding for FLUX.2-dev.

    HF key structure:
        time_guidance_embed.timestep_embedder.linear_1.weight  [inner_dim, time_proj_dim]
        time_guidance_embed.timestep_embedder.linear_2.weight  [inner_dim, inner_dim]
        time_guidance_embed.guidance_embedder.linear_1.weight  [inner_dim, guidance_proj_dim]
        time_guidance_embed.guidance_embedder.linear_2.weight  [inner_dim, inner_dim]

    Sum of timestep embedding + guidance embedding.
    """
    def __init__(self, embedding_dim: int, time_proj_dim: int = 256,
                 guidance_proj_dim: int = 256, guidance_embeds: bool = True,
                 reduce_dtype=torch.bfloat16):
        super().__init__()
        self.time_proj_dim = time_proj_dim
        self.guidance_proj_dim = guidance_proj_dim
        self.guidance_embeds = guidance_embeds

        self.timestep_embedder = _Embedder(time_proj_dim, embedding_dim, reduce_dtype)
        if guidance_embeds:
            self.guidance_embedder = _Embedder(guidance_proj_dim, embedding_dim, reduce_dtype)

    @staticmethod
    def _sinusoidal(values: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=values.device) * -emb
        )
        emb = values.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb.to(values.dtype)

    def forward(self, timestep, guidance=None):
        t_emb = self._sinusoidal(timestep, self.time_proj_dim)
        temb = F.silu(self.timestep_embedder.linear_1(t_emb))
        temb = self.timestep_embedder.linear_2(temb)
        if self.guidance_embeds and guidance is not None:
            g_emb = self._sinusoidal(guidance, self.guidance_proj_dim)
            gemb = F.silu(self.guidance_embedder.linear_1(g_emb))
            gemb = self.guidance_embedder.linear_2(gemb)
            temb = temb + gemb
        return temb


# ============================================================
# FeedForward (SwiGLU split) — verbatim from PR #146
# ============================================================

class NeuronFlux2FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, reduce_dtype=torch.bfloat16):
        super().__init__()
        inner_dim = int(dim * mlp_ratio)
        self.linear_in_gate = ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype,
        )
        self.linear_in_value = ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype,
        )
        self.linear_out = RowParallelLinear(
            inner_dim, dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype,
        )

    def forward(self, x):
        gate = F.silu(self.linear_in_gate(x))
        value = self.linear_in_value(x)
        return self.linear_out(gate * value)


# ============================================================
# Attention (double-stream) — verbatim from PR #146
# ============================================================

class NeuronFlux2Attention(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 added_kv_proj_dim=None, bias=False, eps=1e-6,
                 reduce_dtype=torch.bfloat16):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        tp_degree = get_tensor_model_parallel_size()
        padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
        self.padded_inner_dim = padded_heads * attention_head_dim
        self.heads_per_rank = padded_heads // tp_degree

        self.to_q = ColumnParallelLinear(dim, self.padded_inner_dim, bias=bias,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.to_k = ColumnParallelLinear(dim, self.padded_inner_dim, bias=bias,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.to_v = ColumnParallelLinear(dim, self.padded_inner_dim, bias=bias,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.norm_q = CustomRMSNorm(attention_head_dim, eps=eps)
        self.norm_k = CustomRMSNorm(attention_head_dim, eps=eps)

        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_q_proj = ColumnParallelLinear(added_kv_proj_dim, self.padded_inner_dim, bias=bias,
                                                   gather_output=False, reduce_dtype=reduce_dtype)
            self.add_k_proj = ColumnParallelLinear(added_kv_proj_dim, self.padded_inner_dim, bias=bias,
                                                   gather_output=False, reduce_dtype=reduce_dtype)
            self.add_v_proj = ColumnParallelLinear(added_kv_proj_dim, self.padded_inner_dim, bias=bias,
                                                   gather_output=False, reduce_dtype=reduce_dtype)
            self.norm_added_q = CustomRMSNorm(attention_head_dim, eps=eps)
            self.norm_added_k = CustomRMSNorm(attention_head_dim, eps=eps)
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.padded_inner_dim, dim, bias=False,
                                  input_is_parallel=True, reduce_dtype=reduce_dtype),
            ])
            self.to_add_out = RowParallelLinear(self.padded_inner_dim, dim, bias=False,
                                                input_is_parallel=True, reduce_dtype=reduce_dtype)

    def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None):
        batch_size = hidden_states.shape[0]
        head_dim = self.head_dim

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            add_query = self.add_q_proj(encoder_hidden_states)
            add_key = self.add_k_proj(encoder_hidden_states)
            add_value = self.add_v_proj(encoder_hidden_states)
            add_query = add_query.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
            add_key = add_key.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
            add_value = add_value.view(batch_size, -1, self.heads_per_rank, head_dim).transpose(1, 2)
            add_query = self.norm_added_q(add_query)
            add_key = self.norm_added_k(add_key)

            query = torch.cat([add_query, query], dim=2)
            key = torch.cat([add_key, key], dim=2)
            value = torch.cat([add_value, value], dim=2)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            if _HARDWARE == hardware.TRN1:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False,
                )
            else:
                hidden_states = attention_wrapper_sharded(query, key, value)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads_per_rank * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            txt_len = encoder_hidden_states.shape[1]
            encoder_attn_out = hidden_states[:, :txt_len]
            hidden_attn_out = hidden_states[:, txt_len:]
            hidden_attn_out = self.to_out[0](hidden_attn_out)
            encoder_attn_out = self.to_add_out(encoder_attn_out)
            return hidden_attn_out, encoder_attn_out
        else:
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            if _HARDWARE == hardware.TRN1:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False,
                )
            else:
                hidden_states = attention_wrapper_sharded(query, key, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads_per_rank * head_dim
            )
            return hidden_states.to(query.dtype), None


# ============================================================
# Double-Stream Block — verbatim from PR #146
# ============================================================

class NeuronFlux2TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 mlp_ratio=3.0, eps=1e-6, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = NeuronFlux2Attention(
            dim=dim, num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim, added_kv_proj_dim=dim,
            bias=False, eps=eps, reduce_dtype=reduce_dtype,
        )
        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = NeuronFlux2FeedForward(dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype)
        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = NeuronFlux2FeedForward(dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype)

    def forward(self, hidden_states, encoder_hidden_states,
                temb_mod_img, temb_mod_txt, image_rotary_emb=None):
        img_mods = NeuronFlux2Modulation.split(temb_mod_img, 2)
        txt_mods = NeuronFlux2Modulation.split(temb_mod_txt, 2)

        img_shift_attn, img_scale_attn, img_gate_attn = img_mods[0]
        img_shift_ff, img_scale_ff, img_gate_ff = img_mods[1]
        txt_shift_attn, txt_scale_attn, txt_gate_attn = txt_mods[0]
        txt_shift_ff, txt_scale_ff, txt_gate_ff = txt_mods[1]

        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + img_scale_attn[:, None]) + img_shift_attn[:, None]
        norm_encoder = self.norm1_context(encoder_hidden_states)
        norm_encoder = norm_encoder * (1 + txt_scale_attn[:, None]) + txt_shift_attn[:, None]

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden,
            encoder_hidden_states=norm_encoder,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate_attn[:, None] * attn_output
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + img_scale_ff[:, None]) + img_shift_ff[:, None]
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + img_gate_ff[:, None] * ff_output

        encoder_hidden_states = encoder_hidden_states + txt_gate_attn[:, None] * context_attn_output
        norm_encoder = self.norm2_context(encoder_hidden_states)
        norm_encoder = norm_encoder * (1 + txt_scale_ff[:, None]) + txt_shift_ff[:, None]
        context_ff_output = self.ff_context(norm_encoder)
        encoder_hidden_states = encoder_hidden_states + txt_gate_ff[:, None] * context_ff_output

        return encoder_hidden_states, hidden_states


# ============================================================
# Single-Stream Block — verbatim from PR #146
# ============================================================

class NeuronFlux2SingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 mlp_ratio=3.0, eps=1e-6, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.norm = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        tp_degree = get_tensor_model_parallel_size()
        padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
        self.padded_inner_dim = padded_heads * attention_head_dim
        self.heads_per_rank = padded_heads // tp_degree
        self.head_dim = attention_head_dim

        self.to_q = ColumnParallelLinear(dim, self.padded_inner_dim, bias=False,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.to_k = ColumnParallelLinear(dim, self.padded_inner_dim, bias=False,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.to_v = ColumnParallelLinear(dim, self.padded_inner_dim, bias=False,
                                         gather_output=False, reduce_dtype=reduce_dtype)
        self.norm_q = CustomRMSNorm(attention_head_dim, eps=eps)
        self.norm_k = CustomRMSNorm(attention_head_dim, eps=eps)

        self.proj_mlp_gate = ColumnParallelLinear(dim, self.mlp_hidden_dim, bias=False,
                                                  gather_output=False, reduce_dtype=reduce_dtype)
        self.proj_mlp_value = ColumnParallelLinear(dim, self.mlp_hidden_dim, bias=False,
                                                   gather_output=False, reduce_dtype=reduce_dtype)

        self.proj_out_attn = RowParallelLinear(
            self.padded_inner_dim, dim, bias=False, input_is_parallel=True,
            reduce_dtype=reduce_dtype, reduce_output=False,
        )
        self.proj_out_mlp = RowParallelLinear(
            self.mlp_hidden_dim, dim, bias=False, input_is_parallel=True,
            reduce_dtype=reduce_dtype, reduce_output=False,
        )

    def forward(self, hidden_states, temb_mod, image_rotary_emb=None):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        mods = NeuronFlux2Modulation.split(temb_mod, 1)
        shift, scale, gate = mods[0]

        norm_hidden = self.norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale[:, None]) + shift[:, None]

        query = self.to_q(norm_hidden)
        key = self.to_k(norm_hidden)
        value = self.to_v(norm_hidden)

        query = query.view(batch_size, -1, self.heads_per_rank, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads_per_rank, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads_per_rank, self.head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        mlp_gate = F.silu(self.proj_mlp_gate(norm_hidden))
        mlp_value = self.proj_mlp_value(norm_hidden)
        mlp_hidden = mlp_gate * mlp_value

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if _HARDWARE == hardware.TRN1:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False,
            )
        else:
            attn_output = attention_wrapper_sharded(query, key, value)

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.heads_per_rank * self.head_dim
        )
        attn_output = attn_output.to(query.dtype)

        out_attn = self.proj_out_attn(attn_output)
        out_mlp = self.proj_out_mlp(mlp_hidden)
        proj_out = reduce_from_tensor_model_parallel_region(
            out_attn + out_mlp,
            process_group=self.proj_out_attn.tensor_parallel_group,
        )

        hidden_states = gate[:, None] * proj_out
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================
# Output AdaLN — verbatim from PR #146
# ============================================================

class NeuronFlux2AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim, conditioning_dim, eps=1e-6, reduce_dtype=torch.bfloat16):
        super().__init__()
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)
        self.linear = ColumnParallelLinear(
            conditioning_dim, embedding_dim * 2, bias=False,
            gather_output=True, reduce_dtype=reduce_dtype,
        )
        self.act = nn.SiLU()

    def forward(self, x, conditioning):
        emb = self.linear(self.act(conditioning))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


# ============================================================
# Top-Level Transformer — adapted for dev (5-arg forward, guidance)
# ============================================================

class NeuronFlux2Transformer(nn.Module):
    """FLUX.2-dev transformer backbone.

    Forward signature (5-arg) matches the existing /home/ubuntu/neuron_flux2_pipeline.py
    and /home/ubuntu/compile_dit_tp8.py contract:
        forward(hidden_states, encoder_hidden_states, timestep, guidance, image_rotary_emb)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.inner_dim = config.num_attention_heads * config.attention_head_dim  # 48*128=6144
        reduce_dtype = config.neuron_config.torch_dtype

        self.x_embedder = ColumnParallelLinear(
            config.in_channels, self.inner_dim, bias=False,
            gather_output=True, reduce_dtype=reduce_dtype,
        )
        self.context_embedder = ColumnParallelLinear(
            config.joint_attention_dim, self.inner_dim, bias=False,
            gather_output=True, reduce_dtype=reduce_dtype,
        )

        self.time_guidance_embed = NeuronFlux2TimestepEmbedding(
            embedding_dim=self.inner_dim,
            time_proj_dim=getattr(config, "timestep_guidance_channels", 256),
            guidance_proj_dim=getattr(config, "timestep_guidance_channels", 256),
            guidance_embeds=getattr(config, "guidance_embeds", True),
            reduce_dtype=reduce_dtype,
        )

        self.double_stream_modulation_img = NeuronFlux2Modulation(
            self.inner_dim, mod_param_sets=2, reduce_dtype=reduce_dtype,
        )
        self.double_stream_modulation_txt = NeuronFlux2Modulation(
            self.inner_dim, mod_param_sets=2, reduce_dtype=reduce_dtype,
        )
        self.single_stream_modulation = NeuronFlux2Modulation(
            self.inner_dim, mod_param_sets=1, reduce_dtype=reduce_dtype,
        )

        self.transformer_blocks = nn.ModuleList([
            NeuronFlux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=getattr(config, "mlp_ratio", 3.0),
                reduce_dtype=reduce_dtype,
            )
            for _ in range(config.num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            NeuronFlux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=getattr(config, "mlp_ratio", 3.0),
                reduce_dtype=reduce_dtype,
            )
            for _ in range(config.num_single_layers)
        ])

        self.norm_out = NeuronFlux2AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, eps=1e-6, reduce_dtype=reduce_dtype,
        )
        # HF config often has out_channels=null meaning "same as in_channels"
        out_channels = config.out_channels if config.out_channels is not None else config.in_channels
        self.proj_out = ColumnParallelLinear(
            self.inner_dim, config.patch_size * config.patch_size * out_channels,
            bias=False, gather_output=True, reduce_dtype=reduce_dtype,
        )

    def forward(self, hidden_states, encoder_hidden_states, timestep, guidance, image_rotary_emb):
        dtype = self.config.neuron_config.torch_dtype

        # Match existing pipeline contract: pipeline divides timestep by 1000 before calling
        # and NEFF scales back by 1000. (Consistent with Klein's *1000 and our old
        # neuron_flux2_dit.py.) Guidance NOT scaled.
        timestep = timestep.to(dtype) * 1000

        g = None
        if getattr(self.config, "guidance_embeds", True) and guidance is not None and guidance.numel() > 0:
            g = guidance.to(dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance=g)

        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        hidden_states, encoder_hidden_states = ModuleMarkerStartWrapper()(
            hidden_states, encoder_hidden_states
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_mod_img,
                temb_mod_txt=double_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )
        hidden_states, encoder_hidden_states = ModuleMarkerEndWrapper()(
            hidden_states, encoder_hidden_states
        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for idx, block in enumerate(self.single_transformer_blocks):
            if idx % 2 == 0:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod=single_mod,
                image_rotary_emb=image_rotary_emb,
            )
            if idx % 2 == 1:
                hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = ModuleMarkerEndWrapper()(output)

        return output


# ============================================================
# Config
# ============================================================

class NeuronFlux2Config(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_required_attributes(self):
        return [
            "attention_head_dim", "in_channels", "joint_attention_dim",
            "num_attention_heads", "num_layers", "num_single_layers",
            "patch_size", "out_channels",
        ]


# ============================================================
# HF -> Neuron state_dict converter (copied verbatim from PR #146,
# generalized to work for both Klein and dev via config.num_*_layers)
# ============================================================

def convert_hf_to_neuron_state_dict(state_dict, config):
    """
    Convert HuggingFace Flux2Transformer2DModel state dict to NxDI scaffold.

    PR #146's splits:
      Single blocks:
        attn.to_qkv_mlp_proj.weight [3*I + 2*M, D]
            -> to_q [I,D] + to_k [I,D] + to_v [I,D]
            -> proj_mlp_gate [M,D] + proj_mlp_value [M,D]
        attn.to_out.weight [D, I+M]
            -> proj_out_attn [D,I] + proj_out_mlp [D,M]
        attn.norm_q/k -> norm_q/k (strip `attn.` prefix)

      Double blocks:
        ff.linear_in.weight [2*M, D]   -> linear_in_gate [M,D] + linear_in_value [M,D]
        ff_context.linear_in.weight    -> same split
    """
    inner_dim = config.num_attention_heads * config.attention_head_dim
    mlp_hidden_dim = int(inner_dim * getattr(config, "mlp_ratio", 3.0))

    new_sd = {}
    for key, value in state_dict.items():
        if "single_transformer_blocks." in key:
            block_idx = key.split(".")[1]
            prefix = f"single_transformer_blocks.{block_idx}"

            if ".attn.to_qkv_mlp_proj.weight" in key:
                w = value
                q_w = w[:inner_dim, :]
                k_w = w[inner_dim:2 * inner_dim, :]
                v_w = w[2 * inner_dim:3 * inner_dim, :]
                mlp_gate_w = w[3 * inner_dim:3 * inner_dim + mlp_hidden_dim, :]
                mlp_value_w = w[3 * inner_dim + mlp_hidden_dim:, :]
                new_sd[f"{prefix}.to_q.weight"] = q_w.clone().contiguous()
                new_sd[f"{prefix}.to_k.weight"] = k_w.clone().contiguous()
                new_sd[f"{prefix}.to_v.weight"] = v_w.clone().contiguous()
                new_sd[f"{prefix}.proj_mlp_gate.weight"] = mlp_gate_w.clone().contiguous()
                new_sd[f"{prefix}.proj_mlp_value.weight"] = mlp_value_w.clone().contiguous()
                continue

            if ".attn.to_out.weight" in key:
                w = value
                attn_w = w[:, :inner_dim]
                mlp_w = w[:, inner_dim:]
                new_sd[f"{prefix}.proj_out_attn.weight"] = attn_w.clone().contiguous()
                new_sd[f"{prefix}.proj_out_mlp.weight"] = mlp_w.clone().contiguous()
                continue

            if ".attn.norm_q.weight" in key:
                new_sd[f"{prefix}.norm_q.weight"] = value.contiguous()
                continue
            if ".attn.norm_k.weight" in key:
                new_sd[f"{prefix}.norm_k.weight"] = value.contiguous()
                continue

        if "transformer_blocks." in key and "single_transformer_blocks." not in key:
            if ".ff.linear_in.weight" in key:
                block_prefix = key.rsplit(".ff.linear_in.weight", 1)[0]
                w = value
                gate_w = w[:mlp_hidden_dim, :]
                value_w = w[mlp_hidden_dim:, :]
                new_sd[f"{block_prefix}.ff.linear_in_gate.weight"] = gate_w.clone().contiguous()
                new_sd[f"{block_prefix}.ff.linear_in_value.weight"] = value_w.clone().contiguous()
                continue

            if ".ff_context.linear_in.weight" in key:
                block_prefix = key.rsplit(".ff_context.linear_in.weight", 1)[0]
                w = value
                gate_w = w[:mlp_hidden_dim, :]
                value_w = w[mlp_hidden_dim:, :]
                new_sd[f"{block_prefix}.ff_context.linear_in_gate.weight"] = gate_w.clone().contiguous()
                new_sd[f"{block_prefix}.ff_context.linear_in_value.weight"] = value_w.clone().contiguous()
                continue

        new_sd[key] = value.contiguous()

    return new_sd
