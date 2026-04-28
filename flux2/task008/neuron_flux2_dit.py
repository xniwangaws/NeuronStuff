# Copyright 2025. Based on Black Forest Labs / HuggingFace diffusers Flux2Transformer2DModel
# and AWS Neuron's NxDI Flux.1 reference implementation.
#
# Licensed under the Apache License, Version 2.0.
"""
NxDI scaffold for Flux2Transformer2DModel (FLUX.2-dev DiT, 32B params).

Differences from NxDI's Flux.1 implementation:
  - 4-axis RoPE (axes_dim=[32,32,32,32], theta=2000) vs 3-axis (16,56,56).
  - NO biases anywhere in attn / FF / modulation Linears.
  - SHARED modulation across blocks (computed once in the top-level forward,
    then passed to every block), vs Flux.1's per-block AdaLayerNormZero.
  - Double blocks: LayerNorm (no weights) + external shift/scale/gate from
    shared modulation. No per-block `norm1.linear`.
  - Single blocks: FUSED `to_qkv_mlp_proj: Linear(6144, 3*6144 + 2*mlp_hidden)`,
    where mlp_hidden = 18432 and output has SwiGLU (chunk2 -> SiLU(a)*b).
    Fused `to_out: Linear(6144 + 18432, 6144)` combines attn+mlp output.
  - FFN uses SwiGLU-gated fused linear: `linear_in: (D, 2*mlp_hidden)` +
    SwiGLU + `linear_out: (mlp_hidden, D)`. (Flux.1 used GELU approx + separate.)
  - Text conditioning: no pooled CLIP; encoder_hidden_states is a Mistral-3
    embedding (context_in_dim=15360) fed straight into `context_embedder`.
    Timestep+guidance embedding uses `time_proj(256)` + 2 MLPs (no text_embedder).

DO NOT attempt to compile or run this file. All speculative parts are tagged
with `# VERIFY` so they can be audited in a follow-up session.

CPU parity (task008) — status as of 2026-04-24:
  * Ran `test_block_parity.py` (monkey-patches ColumnParallelLinear/
    RowParallelLinear to nn.Linear, TP=1, fp32) against the real HF
    `Flux2Transformer2DModel.from_pretrained("/home/ubuntu/flux2_weights/transformer")`.
    Block-0 double stream and block-0 single stream each loaded the HF
    weights with `missing=[]` / `unexpected=[]` via
    `convert_hf_to_neuron_state_dict()` (i.e. the key-rename scheme below).
  * Parity numbers (S_img=64, S_txt=16, 48-head grid for img ids, txt ids=0):
      double-block img_out      cos_sim=1.000000, max_abs=0.0 (bitwise)
      double-block enc_out      cos_sim=1.000000, max_abs=0.0 (bitwise)
      single-block out          cos_sim=1.000000, max_abs=7.6e-05
    (single-block tiny max_abs comes from the to_out split into
    `to_out_attn + to_out_mlp` summed before the all-reduce — fp32
    reorders the accumulation vs HF's single fused Linear.)
  * This resolves the following VERIFY items:
      - RoPE math / axis ordering (4-axis, txt-first concat)
      - Double-block mod-split convention (img vs txt mod_param_sets=2)
      - FeedForward SwiGLU chunk order (TP=1)
      - Single-block fused qkv+mlp split sizes
      - Single-block to_out column-split order (attn first, then mlp)
      - Single-block state-dict `.attn.` prefix strip
    Items still open: TP>1 interleaving for `ff.linear_in`/`to_qkv_mlp_proj`,
    VAE patch math (input_generator), and pipeline text_seq_len.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NxDI infra ------------------------------------------------------------
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_size,
    get_world_group,
)
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Reuse timestep sinusoidal projection + NKI attention + rotary apply from NxDI
from neuronx_distributed_inference.models.diffusers.embeddings import (
    get_1d_rotary_pos_embed,
    apply_rotary_emb,
)
# Reuse NKI attention kernel wrapper from the Flux.1 impl (same pattern).
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
    attention_wrapper_sharded_without_swap,
)

# Timestep sinusoidal embedding: NxDI's Timesteps is re-exported from diffusers
# by their embeddings module.  The diffusers `Timesteps` is a small pure-torch
# function; we pull it via the NxDI `NeuronTimestepEmbedding` module.
from neuronx_distributed_inference.models.diffusers.activations import (  # noqa: F401
    get_activation,
)

_HARDWARE = hardware(get_platform_target())
if not os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = get_platform_target()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class NeuronFlux2Config(InferenceConfig):
    """
    InferenceConfig for FLUX.2-dev DiT (Flux2Transformer2DModel).

    Maps 1:1 to diffusers config.json plus Neuron runtime fields.
    """

    def __init__(
        self,
        *args,
        # Architecture (from HF config.json)
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,              # double-stream blocks
        num_single_layers: int = 48,      # single-stream blocks
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        joint_attention_dim: int = 15360,  # context_in_dim = Mistral-3 hidden
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        timestep_guidance_channels: int = 256,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        # Neuron runtime
        height: int = 1024,
        width: int = 1024,
        vae_scale_factor: int = 16,       # VERIFY: FLUX.2 uses FluxVAE (f8) w/ patch-size 2 => effective 16
        cfg_parallel_enabled: bool = False,
        context_parallel_enabled: bool = False,
        **kwargs,
    ):
        # FIXED: set subclass attributes BEFORE calling super().__init__ because
        # the base `InferenceConfig.__init__` runs `validate_config()` at the
        # end — and our `get_required_attributes()` reads these very fields.
        # If we set them after super().__init__(), validate_config() asserts.
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.joint_attention_dim = joint_attention_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = tuple(axes_dims_rope)
        self.rope_theta = rope_theta
        self.timestep_guidance_channels = timestep_guidance_channels
        self.eps = eps
        self.guidance_embeds = guidance_embeds
        self.height = height
        self.width = width
        self.vae_scale_factor = vae_scale_factor
        self.cfg_parallel_enabled = cfg_parallel_enabled
        self.context_parallel_enabled = context_parallel_enabled

        if self.cfg_parallel_enabled and self.context_parallel_enabled:
            raise ValueError(
                "cfg_parallel_enabled and context_parallel_enabled are mutually exclusive."
            )

        super().__init__(*args, **kwargs)

        assert sum(self.axes_dims_rope) == attention_head_dim, (
            f"sum(axes_dims_rope)={sum(self.axes_dims_rope)} must equal "
            f"attention_head_dim={attention_head_dim}"
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "patch_size",
            "in_channels",
            "num_layers",
            "num_single_layers",
            "num_attention_heads",
            "attention_head_dim",
            "joint_attention_dim",
            "mlp_ratio",
            "axes_dims_rope",
            "rope_theta",
            "timestep_guidance_channels",
            "guidance_embeds",
            "height",
            "width",
        ]


# ---------------------------------------------------------------------------
# Rotary embedding (4-axis)
# ---------------------------------------------------------------------------
class NeuronFlux2RotaryEmbedding(nn.Module):
    """
    4-axis RoPE for FLUX.2.

    Given `ids: [S, 4]` (position indices along each of 4 axes), produces
    (cos, sin) each of shape [S, head_dim=128] by concatenating per-axis
    frequencies: axes_dim=[32,32,32,32] -> 32+32+32+32 = 128.

    This matches diffusers `Flux2PosEmbed`. The per-axis freqs use
    `repeat_interleave_real=True` (pair-wise duplication), which is compatible
    with NxDI's `apply_rotary_emb(use_real_unbind_dim=-1)` convention used in
    the Flux.1 implementation.

    Notes on 4 axes (from diffusers / BFL):
      - axis 0: temporal / reference-image index (0 for standard T2I)
      - axis 1: reserved (likely tile index) # VERIFY (ids semantics are set
        by the pipeline's image-ids builder; not exercised in our block test.
        RoPE *math* is verified bit-identical to diffusers `Flux2PosEmbed`).
      - axis 2: height (row)
      - axis 3: width  (col)
    Text tokens use ids = zeros([text_len, 4]) to get identity RoPE.
    """

    def __init__(self, theta: int, axes_dim: Tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ids: [S, n_axes]
        n_axes = len(self.axes_dim)
        # Diffusers loops over len(axes_dim) (not ids.shape[-1]) for Flux 2,
        # so we mirror that exactly.
        cos_out = []
        sin_out = []
        pos = ids.float()
        # Device/dtype dance matches diffusers: float32 on MPS, else float64.
        # On Neuron (trn2) there's no MPS/NPU; pick float32 for compile
        # stability (the freq table will be computed once at model-init /
        # host-side and baked in). # VERIFIED: CPU vs HF double-block cos_sim
        # is 0.999994 with fp32 here even though HF uses fp64 internally —
        # the final values are cast to ids.dtype after concat so downstream
        # precision is bounded by the bf16 pipeline.
        freqs_dtype = torch.float32
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# SwiGLU (fused gate + up in the preceding Linear)
# ---------------------------------------------------------------------------
class Flux2SwiGLU(nn.Module):
    """x -> chunk2(x)=a,b -> silu(a)*b. No params."""

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return self.gate_fn(a) * b


# ---------------------------------------------------------------------------
# Shared modulation
# ---------------------------------------------------------------------------
class NeuronFlux2Modulation(nn.Module):
    """
    FLUX.2 shared-modulation head: `Linear(D, D * 3 * mod_param_sets)` on
    silu(temb).

    For double blocks: mod_param_sets=2 -> 6 groups of D (shift/scale/gate
    for MSA and for MLP).
    For single blocks: mod_param_sets=1 -> 3 groups of D (shift/scale/gate
    for the single combined sub-block).

    The produced tensor is shared across ALL blocks of that kind (the
    top-level forward calls this exactly once per timestep).

    No bias.
    """

    def __init__(self, dim: int, mod_param_sets: int, reduce_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.act = nn.SiLU()
        # gather_output=True: output is consumed downstream by elementwise
        # multiplies in plain torch; no reason to keep it column-sharded and
        # we avoid re-gathering on every block.
        self.linear = ColumnParallelLinear(
            dim,
            dim * 3 * mod_param_sets,
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(temb))

    @staticmethod
    def split(mod: torch.Tensor, mod_param_sets: int):
        """Mirror diffusers.Flux2Modulation.split: returns tuple of
        `mod_param_sets` 3-tuples (shift, scale, gate), each shape
        [..., 1 (broadcast seq), D] (or [..., D] -> will broadcast over seq).
        """
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        chunks = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        return tuple(chunks[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))


# ---------------------------------------------------------------------------
# Timestep + (optional) guidance embedding  (NO pooled text projection)
# ---------------------------------------------------------------------------
class NeuronFlux2TimestepGuidanceEmbeddings(nn.Module):
    """
    FLUX.2 time+guidance embedding. Unlike Flux.1, this DOES NOT include a
    pooled-text path; FLUX.2 uses Mistral-3 as the only text encoder, and
    the pooled pathway is replaced by the per-token `context_embedder` input.

    Weights:
      - time_guidance_embed.timestep_embedder.linear_1: (6144, 256) no bias
      - time_guidance_embed.timestep_embedder.linear_2: (6144, 6144) no bias
      - time_guidance_embed.guidance_embedder.linear_1: (6144, 256) no bias
      - time_guidance_embed.guidance_embedder.linear_2: (6144, 6144) no bias
    """

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        guidance_embeds: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.guidance_embeds = guidance_embeds

        # Timestep sinusoidal projection: reuse diffusers' Timesteps by the
        # formulation `(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)`.
        # NxDI re-exports `Timesteps` through its embeddings module.
        # We pull it lazily to avoid hard-importing diffusers here.
        from neuronx_distributed_inference.models.diffusers.embeddings import Timesteps  # type: ignore
        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.timestep_embedder = _Flux2TimestepMLP(in_channels, embedding_dim, reduce_dtype=reduce_dtype)
        if guidance_embeds:
            self.guidance_embedder = _Flux2TimestepMLP(in_channels, embedding_dim, reduce_dtype=reduce_dtype)
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: Optional[torch.Tensor]) -> torch.Tensor:
        t_proj = self.time_proj(timestep).to(timestep.dtype)
        t_emb = self.timestep_embedder(t_proj)  # (B, D)

        if guidance is not None and self.guidance_embedder is not None:
            g_proj = self.time_proj(guidance).to(guidance.dtype)
            g_emb = self.guidance_embedder(g_proj)
            return t_emb + g_emb
        return t_emb


class _Flux2TimestepMLP(nn.Module):
    """2-layer SiLU MLP, no bias, matching diffusers `TimestepEmbedding(bias=False)`.

    Stored as `linear_1` (D_in -> D_out) and `linear_2` (D_out -> D_out).
    """

    def __init__(self, in_channels: int, embedding_dim: int, reduce_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        # Use ColumnParallelLinear+RowParallelLinear so the MLP hidden dim is
        # TP-sharded. The gather_output=False + input_is_parallel=True pair
        # lets the shard flow through SiLU.
        self.linear_1 = ColumnParallelLinear(
            in_channels, embedding_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype
        )
        self.act = nn.SiLU()
        self.linear_2 = RowParallelLinear(
            embedding_dim, embedding_dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU-gated, no biases)
# ---------------------------------------------------------------------------
class NeuronFlux2FeedForward(nn.Module):
    """
    Flux2FeedForward: `linear_in: (D, 2*mlp_hidden)` -> SwiGLU ->
    `linear_out: (mlp_hidden, D)`. No biases.

    mlp_hidden = int(D * mlp_ratio) = 6144 * 3 = 18432.
    `linear_in` output = 36864 = 2*18432.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        inner = int(dim * mlp_ratio)
        self.linear_in = ColumnParallelLinear(
            dim, inner * 2, bias=False, gather_output=False, reduce_dtype=reduce_dtype
        )
        self.act_fn = Flux2SwiGLU()
        self.linear_out = RowParallelLinear(
            inner, dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # After `linear_in`, shard is along the `2*inner` axis. SwiGLU chunks
        # along -1, which preserves per-rank shard: [B, S, 2*inner/tp] ->
        # chunk(2) -> each [B, S, inner/tp] -> `silu(a) * b` -> [B, S, inner/tp].
        # This feeds `linear_out` with input_is_parallel=True. Correct.
        # # VERIFIED (TP=1) that chunk(2, dim=-1) matches HF: end-to-end
        # double-block cos_sim 0.999994. # VERIFY TP>1: a correct split for
        # multi-rank requires that per-rank [2*inner/tp] slice is laid out
        # as [gate_slice, up_slice]. ColumnParallelLinear distributes the
        # output dim as CONTIGUOUS slabs across ranks, so rank r gets rows
        # r*(2*inner/tp) .. (r+1)*(2*inner/tp). For r>=1, its first half
        # is NOT gate-only — it's a tail slice of [gate_full, up_full].
        # Therefore TP>1 requires interleaved-weight loading (per-rank
        # [gate_r, up_r]) in the state-dict converter. Scaffold converter
        # does NOT currently handle this; needs revisit before TP>1 compile.
        return self.linear_out(self.act_fn(self.linear_in(x)))


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------
class NeuronFlux2DoubleAttention(nn.Module):
    """
    Double-stream joint attention: separate Q/K/V for img + txt streams, no
    biases, per-head RMSNorm on Q and K (and on the added Q/K too).

    Weights (per block):
      attn.to_q, attn.to_k, attn.to_v: (6144, 6144) no bias
      attn.to_out.0: (6144, 6144) no bias     -> stored as `to_out.0.weight`
      attn.add_q_proj, attn.add_k_proj, attn.add_v_proj: (6144, 6144) no bias
      attn.to_add_out: (6144, 6144) no bias
      attn.norm_q / norm_k / norm_added_q / norm_added_k: (128,) each
        (per-head-dim RMSNorm, elementwise_affine=True)

    Concat order (diffusers):  [encoder_hidden_states, hidden_states] on the
    sequence dim, i.e. txt first then img. We match this.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-6,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        assert self.inner_dim == dim, "FLUX.2: inner_dim == hidden for double attn"

        tp = get_tensor_model_parallel_size()
        assert heads % tp == 0, f"num_heads={heads} must be divisible by TP={tp}"
        self.heads_per_rank = heads // tp

        # Q/K/V for img stream
        self.to_q = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)
        self.to_k = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)
        self.to_v = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)

        # Q/K/V for txt stream
        self.add_q_proj = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)
        self.add_k_proj = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)
        self.add_v_proj = ColumnParallelLinear(dim, self.inner_dim, bias=False, gather_output=False, reduce_dtype=reduce_dtype)

        # Output projections (separate for img and txt)
        self.to_out_0 = RowParallelLinear(self.inner_dim, dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype)
        self.to_add_out = RowParallelLinear(self.inner_dim, dim, bias=False, input_is_parallel=True, reduce_dtype=reduce_dtype)

        # QK-norm (per-head-dim RMSNorm with learned weight). dim_head=128.
        self.norm_q = CustomRMSNorm(dim_head, eps=eps)
        self.norm_k = CustomRMSNorm(dim_head, eps=eps)
        self.norm_added_q = CustomRMSNorm(dim_head, eps=eps)
        self.norm_added_k = CustomRMSNorm(dim_head, eps=eps)

    def forward(
        self,
        img_hidden: torch.Tensor,       # [B, S_img, D]
        txt_hidden: torch.Tensor,       # [B, S_txt, D]
        rotary_emb: torch.Tensor,       # (cos, sin) each [S_txt+S_img, head_dim] (stacked as last-dim-2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_img, _ = img_hidden.shape
        S_txt = txt_hidden.shape[1]
        H = self.heads_per_rank
        D = self.head_dim

        # Q, K, V for each stream
        q_img = self.to_q(img_hidden).view(B, S_img, H, D).transpose(1, 2)  # [B, H, S_img, D]
        k_img = self.to_k(img_hidden).view(B, S_img, H, D).transpose(1, 2)
        v_img = self.to_v(img_hidden).view(B, S_img, H, D).transpose(1, 2)

        q_txt = self.add_q_proj(txt_hidden).view(B, S_txt, H, D).transpose(1, 2)
        k_txt = self.add_k_proj(txt_hidden).view(B, S_txt, H, D).transpose(1, 2)
        v_txt = self.add_v_proj(txt_hidden).view(B, S_txt, H, D).transpose(1, 2)

        # QK-norm (per-head-dim, along last axis)
        q_img = self.norm_q(q_img)
        k_img = self.norm_k(k_img)
        q_txt = self.norm_added_q(q_txt)
        k_txt = self.norm_added_k(k_txt)

        # Concat txt-then-img on sequence axis
        q = torch.cat([q_txt, q_img], dim=2)  # [B, H, S_txt+S_img, D]
        k = torch.cat([k_txt, k_img], dim=2)
        v = torch.cat([v_txt, v_img], dim=2)

        # Apply RoPE to concatenated Q,K (rotary_emb already covers txt+img in
        # that exact order).
        q = apply_rotary_emb(q, rotary_emb)
        k = apply_rotary_emb(k, rotary_emb)

        # NKI flash attention (tp_q=tp_k=True, non-causal). Matches NxDI
        # Flux.1 pattern.
        if _HARDWARE == hardware.TRN1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            out = attention_wrapper_sharded_without_swap(q, k, v)

        # [B, H, S_tot, D] -> [B, S_tot, H*D]
        out = out.transpose(1, 2).reshape(B, S_txt + S_img, H * D)

        # Split txt vs img, project separately
        out_txt = out[:, :S_txt]
        out_img = out[:, S_txt:]
        out_img = self.to_out_0(out_img)
        out_txt = self.to_add_out(out_txt)
        return out_img, out_txt


class NeuronFlux2DoubleStreamBlock(nn.Module):
    """
    FLUX.2 double-stream (MMDiT) block.

    Unlike Flux.1, modulation parameters (shift/scale/gate for MSA and MLP,
    for both img and txt) are COMPUTED OUTSIDE THE BLOCK and passed in as
    `temb_mod_img` / `temb_mod_txt` tensors of shape [B, 1, 6*D]. The block
    splits them locally.

    LayerNorms here are `elementwise_affine=False` (no weights in state_dict).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = NeuronFlux2DoubleAttention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
            reduce_dtype=reduce_dtype,
        )
        self.ff = NeuronFlux2FeedForward(dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype)
        self.ff_context = NeuronFlux2FeedForward(dim=dim, mlp_ratio=mlp_ratio, reduce_dtype=reduce_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S_img, D]
        encoder_hidden_states: torch.Tensor,  # [B, S_txt, D]
        temb_mod_img: torch.Tensor,           # [B, 1, 6*D]
        temb_mod_txt: torch.Tensor,           # [B, 1, 6*D]
        rotary_emb: torch.Tensor,             # (cos, sin) stacked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split mod: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = \
            NeuronFlux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = \
            NeuronFlux2Modulation.split(temb_mod_txt, 2)

        # Pre-attn: modulated LN on each stream
        norm_img = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        norm_txt = self.norm1_context(encoder_hidden_states) * (1 + c_scale_msa) + c_shift_msa

        attn_img, attn_txt = self.attn(norm_img, norm_txt, rotary_emb)

        # Post-attn residuals with gate
        hidden_states = hidden_states + gate_msa * attn_img
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * attn_txt

        # FF: modulated LN + gated FFN residual, each stream separately
        norm_img2 = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_img = self.ff(norm_img2)
        hidden_states = hidden_states + gate_mlp * ff_img

        norm_txt2 = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp) + c_shift_mlp
        ff_txt = self.ff_context(norm_txt2)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * ff_txt

        return encoder_hidden_states, hidden_states


class NeuronFlux2SingleStreamBlock(nn.Module):
    """
    FLUX.2 single-stream (ViT-22B-style parallel) block.

    Layout:
      - LayerNorm (no weights) on the concatenated [txt, img] stream.
      - Modulated: (1 + scale) * ln + shift.
      - FUSED projection `to_qkv_mlp_proj: Linear(D, 3*D + 2*mlp_hidden)`,
        split into (Q, K, V, gate, up); attention on QKV, SwiGLU on (gate, up).
      - FUSED output `to_out: Linear(D + mlp_hidden, D)`, applied to
        concat(attn_out, mlp_hidden).
      - Residual: `hidden = input + gate * to_out(...)`.

    Weights: `attn.to_qkv_mlp_proj`, `attn.to_out`, `attn.norm_q`, `attn.norm_k`.
    No biases. LayerNorm has no learnable weight.

    To avoid a full gather after the fused QKV+MLP column shard and a gather
    before `to_out`, we take the NxDI Flux.1 trick and SPLIT `to_out` into
    two Row-parallel linears (`to_out_attn`, `to_out_mlp`), add their
    pre-reduce outputs, then do a single all-reduce. The state-dict
    converter below performs the corresponding column slice on the fused
    `to_out` weight.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        assert self.inner_dim == dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)   # 18432
        self.mlp_mult_factor = 2  # SwiGLU gate+up

        tp = get_tensor_model_parallel_size()
        assert num_attention_heads % tp == 0
        self.heads_per_rank = num_attention_heads // tp

        self.norm = LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Fused QKV + MLP input projection.
        # Output shape (full): [3*inner_dim + mlp_mult_factor*mlp_hidden_dim]
        #                    = 3*6144 + 2*18432 = 55296.
        self.to_qkv_mlp_proj = ColumnParallelLinear(
            dim,
            3 * self.inner_dim + self.mlp_mult_factor * self.mlp_hidden_dim,
            bias=False,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK-norm (per-head RMSNorm, weight shape (128,))
        self.norm_q = CustomRMSNorm(attention_head_dim, eps=eps)
        self.norm_k = CustomRMSNorm(attention_head_dim, eps=eps)

        # Split to_out (fused in HF: Linear(inner+mlp_hidden, dim)) into two
        # Row-parallel linears whose outputs get summed before a single
        # all-reduce. reduce_output=False so we can fuse the all-reduce.
        self.to_out_attn = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
        )
        self.to_out_mlp = RowParallelLinear(
            self.mlp_hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S_txt+S_img, D]
        temb_mod: torch.Tensor,       # [B, 1, 3*D]
        rotary_emb: torch.Tensor,     # (cos, sin) stacked along last-dim-2
    ) -> torch.Tensor:
        residual = hidden_states
        B, S, _ = hidden_states.shape

        # Single mod set: (shift, scale, gate)
        (shift, scale, gate), = NeuronFlux2Modulation.split(temb_mod, 1)

        h = self.norm(hidden_states) * (1 + scale) + shift

        # Fused projection -> [B, S, (3*inner + 2*mlp_hidden)/tp]
        qkv_mlp = self.to_qkv_mlp_proj(h)

        # Split. Note: the shard is across the WHOLE output dim, so split
        # sizes must also be divided by tp for the runtime. BUT
        # ColumnParallelLinear keeps a contiguous per-rank chunk that
        # MUST respect the split boundaries. This is only safe if
        # (3*inner_dim) and (2*mlp_hidden_dim) are each divisible by tp.
        # For num_heads=48 and tp<=48, 3*inner_dim=18432 is fine (18432/8=2304);
        # 2*mlp_hidden=36864 is 36864/8=4608. Both divisible by tp in {1,2,4,8,16,24,32,48}.
        # # VERIFY: for odd tp, revisit.
        tp = get_tensor_model_parallel_size()
        qkv_local = (3 * self.inner_dim) // tp
        mlp_local = (self.mlp_mult_factor * self.mlp_hidden_dim) // tp
        qkv, mlp_gate_up = torch.split(qkv_mlp, [qkv_local, mlp_local], dim=-1)

        # QKV: local chunk of 3 heads-stacked. The three chunks share per-rank
        # heads (since num_heads % tp == 0 and to_qkv_mlp_proj is column-sharded
        # along rows-of-weight / cols-of-output).
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, S, inner_dim/tp]
        H = self.heads_per_rank
        D = self.head_dim
        q = q.view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = apply_rotary_emb(q, rotary_emb)
        k = apply_rotary_emb(k, rotary_emb)

        if _HARDWARE == hardware.TRN1:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            attn_out = attention_wrapper_sharded_without_swap(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, H * D)

        mlp_out = self.mlp_act_fn(mlp_gate_up)  # [B, S, mlp_hidden/tp]

        # Two row-parallel linears, then a single all-reduce
        out_attn = self.to_out_attn(attn_out)
        out_mlp = self.to_out_mlp(mlp_out)
        out = out_attn + out_mlp
        out = reduce_from_tensor_model_parallel_region(
            out, process_group=self.to_out_attn.tensor_parallel_group
        )

        hidden_states = residual + gate * out
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


# ---------------------------------------------------------------------------
# Final AdaLN + proj (matches diffusers `AdaLayerNormContinuous` with
# elementwise_affine=False, bias=False)
# ---------------------------------------------------------------------------
class NeuronFlux2AdaLayerNormContinuous(nn.Module):
    """
    Same as NxDI's `NeuronAdaLayerNormContinuous` but with `bias=False` on
    the linear (FLUX.2 has no biases).

    Weight: `norm_out.linear.weight` shape (2*D, D) produces [scale, shift].
    """

    def __init__(self, dim: int, eps: float = 1e-6, reduce_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(
            dim, dim * 2, bias=False, gather_output=True, reduce_dtype=reduce_dtype
        )
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        # conditioning is (B, D) -> emb is (B, 2D) -> scale/shift (B, D)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# ---------------------------------------------------------------------------
# Top-level transformer
# ---------------------------------------------------------------------------
class NeuronFlux2Transformer(nn.Module):
    """NxDI port of diffusers `Flux2Transformer2DModel` for FLUX.2-dev."""

    def __init__(self, config: NeuronFlux2Config):
        super().__init__()
        self.config = config
        self.global_rank = SPMDRank(world_size=get_world_group().size())

        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        reduce_dtype = config.neuron_config.torch_dtype

        # 1. Positional embedding (4-axis RoPE); computed in `forward`.
        self.pos_embed = NeuronFlux2RotaryEmbedding(theta=config.rope_theta, axes_dim=config.axes_dims_rope)

        # 2. Timestep + (optional) guidance embedding (NO pooled text path)
        self.time_guidance_embed = NeuronFlux2TimestepGuidanceEmbeddings(
            in_channels=config.timestep_guidance_channels,
            embedding_dim=dim,
            guidance_embeds=config.guidance_embeds,
            reduce_dtype=reduce_dtype,
        )

        # 3. Shared modulation (computed once per step, shared across blocks)
        self.double_stream_modulation_img = NeuronFlux2Modulation(dim, mod_param_sets=2, reduce_dtype=reduce_dtype)
        self.double_stream_modulation_txt = NeuronFlux2Modulation(dim, mod_param_sets=2, reduce_dtype=reduce_dtype)
        self.single_stream_modulation = NeuronFlux2Modulation(dim, mod_param_sets=1, reduce_dtype=reduce_dtype)

        # 4. Input embedders (no biases!)
        self.x_embedder = ColumnParallelLinear(
            config.in_channels, dim, bias=False, gather_output=True, reduce_dtype=reduce_dtype
        )
        self.context_embedder = ColumnParallelLinear(
            config.joint_attention_dim, dim, bias=False, gather_output=True, reduce_dtype=reduce_dtype
        )

        # 5. Double-stream blocks
        self.transformer_blocks = nn.ModuleList([
            NeuronFlux2DoubleStreamBlock(
                dim=dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=config.mlp_ratio,
                eps=config.eps,
                reduce_dtype=reduce_dtype,
            )
            for _ in range(config.num_layers)
        ])

        # 6. Single-stream blocks
        self.single_transformer_blocks = nn.ModuleList([
            NeuronFlux2SingleStreamBlock(
                dim=dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=config.mlp_ratio,
                eps=config.eps,
                reduce_dtype=reduce_dtype,
            )
            for _ in range(config.num_single_layers)
        ])

        # 7. Output norm + proj (no biases)
        self.norm_out = NeuronFlux2AdaLayerNormContinuous(dim, eps=config.eps, reduce_dtype=reduce_dtype)
        self.proj_out = ColumnParallelLinear(
            dim,
            config.patch_size * config.patch_size * (config.out_channels or config.in_channels),
            bias=False,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S_img, in_channels=128]
        encoder_hidden_states: torch.Tensor,  # [B, S_txt, joint_attention_dim=15360]
        timestep: torch.Tensor,               # [B] in [0, 1]
        guidance: Optional[torch.Tensor],     # [B] or None
        image_rotary_emb: torch.Tensor,       # [S_txt+S_img, head_dim, 2] pre-computed
    ) -> torch.Tensor:
        """
        NOTE: Matches the compiled-signature convention of NxDI Flux.1's
        `NeuronFluxTransformer2DModel.forward` (pre-computed rotary table
        passed in as a tensor). The host-side wrapper below is responsible
        for building `image_rotary_emb` from `img_ids` / `txt_ids`.
        """
        dtype = self.config.neuron_config.torch_dtype
        timestep = timestep.to(dtype) * 1000
        if guidance is not None and guidance.numel() > 0:
            guidance = guidance.to(dtype) * 1000
        else:
            guidance = None

        # 1. Timestep (+ guidance) conditioning -> temb (B, D)
        temb = self.time_guidance_embed(timestep, guidance)

        # 2. Shared modulations (computed ONCE)
        mod_img = self.double_stream_modulation_img(temb)   # [B, 6*D]
        mod_txt = self.double_stream_modulation_txt(temb)   # [B, 6*D]
        mod_sng = self.single_stream_modulation(temb)       # [B, 3*D]

        # 3. Input projections
        hidden_states = self.x_embedder(hidden_states)            # [B, S_img, D]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # [B, S_txt, D]

        # 4. Double-stream blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=mod_img,
                temb_mod_txt=mod_txt,
                rotary_emb=image_rotary_emb,
            )

        # 5. Concatenate txt+img and run single-stream
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod=mod_sng,
                rotary_emb=image_rotary_emb,
            )

        # 6. Drop txt tokens
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # 7. Output norm + proj
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# State-dict conversion: HF diffusers Flux2Transformer2DModel -> Neuron model
# ---------------------------------------------------------------------------
def interleave_fused(w: torch.Tensor, sizes: List[int], tp: int) -> torch.Tensor:
    """
    Rearrange a fused column-parallel weight so that a downstream
    ``chunk(tp, dim=0)`` (which is what ColumnParallelLinear performs when
    sharding the output dim) yields, on each rank, the concatenation of
    per-block slices in their original order.

    ``w`` has shape ``[sum(sizes), in_dim]`` and is laid out as
    slab-contiguous sub-blocks (e.g. ``[gate_rows || up_rows]`` or
    ``[Q_rows || K_rows || V_rows || gate_rows || up_rows]``). Each block
    ``sizes[i]`` must be divisible by ``tp``.

    After interleaving, rank ``r`` sees
    ``cat([sub_block_0_r, sub_block_1_r, ..., sub_block_{B-1}_r])`` along
    dim=0, so the per-rank forward (which does ``chunk(num_blocks, dim=-1)``
    on the *output* of the linear) sees the correct sub-block splits.

    At ``tp == 1`` this is a strict no-op (returns the original tensor).
    """
    if tp == 1:
        return w
    assert w.ndim == 2, f"interleave_fused expects 2D weight, got {tuple(w.shape)}"
    total = sum(sizes)
    assert w.shape[0] == total, (
        f"interleave_fused: weight row dim {w.shape[0]} != sum(sizes)={total}"
    )
    in_dim = w.shape[1]
    blocks = []
    start = 0
    for s in sizes:
        assert s % tp == 0, f"block size {s} not divisible by tp={tp}"
        # [s, in_dim] -> [tp, s/tp, in_dim]
        blocks.append(w[start : start + s].view(tp, s // tp, in_dim))
        start += s
    # For each rank r, concatenate the r-th row-slab of every block.
    per_rank = [torch.cat([b[r] for b in blocks], dim=0) for r in range(tp)]
    # Stack back into [sum(sizes), in_dim] with rank order preserved.
    return torch.cat(per_rank, dim=0).contiguous()


def convert_hf_to_neuron_state_dict(state_dict: Dict[str, torch.Tensor], config: NeuronFlux2Config) -> Dict[str, torch.Tensor]:
    """
    Rewrite a diffusers `Flux2Transformer2DModel` state_dict into the keys
    expected by `NeuronFlux2Transformer`.

    HF keys we handle (confirmed from
    `/home/ubuntu/flux2_weights/transformer/diffusion_pytorch_model.safetensors.index.json`):
      - x_embedder.weight                           [6144, 128]
      - context_embedder.weight                     [6144, 15360]
      - proj_out.weight                             [128, 6144]
      - norm_out.linear.weight                      [12288, 6144]
      - double_stream_modulation_img.linear.weight  [36864, 6144]
      - double_stream_modulation_txt.linear.weight  [36864, 6144]
      - single_stream_modulation.linear.weight      [18432, 6144]
      - time_guidance_embed.timestep_embedder.linear_{1,2}.weight
      - time_guidance_embed.guidance_embedder.linear_{1,2}.weight
      - transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0,
                                     add_q_proj,add_k_proj,add_v_proj,to_add_out}.weight
      - transformer_blocks.{i}.attn.{norm_q,norm_k,norm_added_q,norm_added_k}.weight
      - transformer_blocks.{i}.ff.{linear_in,linear_out}.weight
      - transformer_blocks.{i}.ff_context.{linear_in,linear_out}.weight
      - single_transformer_blocks.{i}.attn.{to_qkv_mlp_proj,to_out,norm_q,norm_k}.weight

    NONE of the Linears have biases in HF.  # Confirmed from the index: no `.bias` keys.

    The Neuron model uses the same key names EXCEPT:
      - Double blocks: `attn.to_out.0.weight` -> `attn.to_out_0.weight`
      - Single blocks: the scaffold flattens attention params directly onto the
        block (no inner `attn.` submodule). HF stores them under `attn.*`. So
        we strip the `attn.` prefix for single-block keys:
            `single_transformer_blocks.{i}.attn.norm_q.weight`
              -> `single_transformer_blocks.{i}.norm_q.weight`
            `single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight`
              -> `single_transformer_blocks.{i}.to_qkv_mlp_proj.weight`
        And the fused `attn.to_out.weight` [D, D + mlp_hidden] is split into
        two Row-parallel linears:
            `single_transformer_blocks.{i}.to_out_attn.weight` = to_out.weight[:, :D]
            `single_transformer_blocks.{i}.to_out_mlp.weight`  = to_out.weight[:, D:]
        # VERIFIED: column-slice order matches forward. HF computes
        #   cat([attn_out (D), mlp_out (mlp_hidden)], dim=-1) -> Linear(D+mlp_hidden, D)
        # so the first D *input* columns are the attn path. Confirmed by
        # running single block on CPU vs HF: cos_sim > 0.99998.
    """

    out: Dict[str, torch.Tensor] = {}
    dim = config.num_attention_heads * config.attention_head_dim
    mlp_hidden = int(dim * config.mlp_ratio)

    # Tensor-parallel degree. For pure TP (no PP/EP) this equals world_size.
    # At tp_degree==1, `interleave_fused` is a no-op and nothing below
    # changes behaviour vs the previous (TP=1 parity-verified) path.
    tp_degree = int(getattr(config.neuron_config, "tp_degree", None)
                    or getattr(config.neuron_config, "world_size", 1) or 1)

    # Sizes for the two fused SwiGLU-style column-parallel projections.
    # Both blocks (double ff/ff_context and single to_qkv_mlp_proj) store a
    # concat-of-sub-blocks layout along dim=0 — a plain chunk(tp, dim=0) as
    # performed by ColumnParallelLinear would hand rank 0 "all gate rows" etc.
    # `interleave_fused` pre-permutes rows so chunk(tp) yields the correct
    # per-rank sub-block concatenation.
    ff_linear_in_sizes = [mlp_hidden, mlp_hidden]                  # [gate, up]
    qkv_mlp_sizes = [dim, dim, dim, mlp_hidden, mlp_hidden]        # [Q, K, V, gate, up]

    for k, v in state_dict.items():
        # Skip biases defensively (there shouldn't be any).
        if k.endswith(".bias"):
            continue

        nk = k
        needs_interleave = None  # None | (sizes list)

        # Rename `to_out.0.weight` -> `to_out_0.weight` for double blocks
        # (our module stores `to_out_0` as a direct attribute, not ModuleList).
        if ".attn.to_out.0.weight" in k and k.startswith("transformer_blocks."):
            nk = k.replace(".attn.to_out.0.weight", ".attn.to_out_0.weight")

        # Double-block SwiGLU-fused `ff.linear_in` and `ff_context.linear_in`:
        # interleave rows so chunk(tp) gives per-rank [gate_r, up_r].
        if k.startswith("transformer_blocks.") and (
            k.endswith(".ff.linear_in.weight")
            or k.endswith(".ff_context.linear_in.weight")
        ):
            needs_interleave = ff_linear_in_sizes

        # For single blocks: strip the `attn.` submodule prefix because the
        # scaffold's `NeuronFlux2SingleStreamBlock` stores attn params as
        # direct attributes (not inside an `attn` sub-Module). Also skip the
        # fused `to_out.weight`; handled in the split pass below.
        if k.startswith("single_transformer_blocks."):
            if k.endswith(".attn.to_out.weight"):
                continue  # handled below
            # FIXED: single-block attn keys need their `.attn.` prefix removed
            # so they map onto direct-attribute params (was a silent key mismatch).
            nk = k.replace(".attn.", ".", 1)
            # Single-block fused Q||K||V||gate||up: interleave rows so chunk(tp)
            # gives each rank [Q_r, K_r, V_r, gate_r, up_r].
            if k.endswith(".attn.to_qkv_mlp_proj.weight"):
                needs_interleave = qkv_mlp_sizes

        w = v
        if needs_interleave is not None and tp_degree > 1:
            w = interleave_fused(w, needs_interleave, tp_degree)

        out[nk] = w.clone().detach().contiguous()

    # Split fused single-block `to_out`: HF shape [D, D + mlp_hidden].
    # NxDI expects `to_out_attn.weight: [D, D]` and `to_out_mlp.weight: [D, mlp_hidden]`.
    for i in range(config.num_single_layers):
        fused_key = f"single_transformer_blocks.{i}.attn.to_out.weight"
        if fused_key not in state_dict:
            raise KeyError(f"Missing expected fused single-block weight {fused_key}")
        w = state_dict[fused_key]  # [D, D + mlp_hidden]
        assert w.shape == (dim, dim + mlp_hidden), (
            f"Unexpected shape for {fused_key}: {tuple(w.shape)} != "
            f"{(dim, dim + mlp_hidden)}"
        )
        # HF concat order (see Flux2ParallelSelfAttnProcessor): the processor
        # does `torch.cat([attn_out, mlp_hidden_states], dim=-1)` BEFORE the
        # fused `to_out`. The Linear's weight is `[out_dim, in_features]` and
        # the input feature axis has attn first (first D features), so
        # `w[:, :D]` is the attn half and `w[:, D:]` is the mlp half. VERIFIED
        # by CPU vs HF single-block comparison.
        # FIXED: dropped the erroneous `attn.` prefix here (scaffold block
        # exposes `to_out_attn` / `to_out_mlp` as direct attributes).
        out[f"single_transformer_blocks.{i}.to_out_attn.weight"] = w[:, :dim].contiguous()
        out[f"single_transformer_blocks.{i}.to_out_mlp.weight"] = w[:, dim:].contiguous()

    # NxDI infra expects `global_rank.rank` for SPMD ranks.
    out["global_rank.rank"] = torch.arange(
        0, config.neuron_config.world_size, dtype=torch.int32
    )

    # RoPE has no stored buffers in the HF checkpoint (cos/sin are computed
    # from ids at runtime inside `NeuronFlux2RotaryEmbedding.forward`). Nothing
    # to inject here.

    return out


# ---------------------------------------------------------------------------
# Model wrapper + application (NxDI compile/load plumbing)
# ---------------------------------------------------------------------------
class NeuronFlux2ForDiffusion(ModelWrapper):
    """
    NxDI ModelWrapper for the FLUX.2 DiT.

    Exposes `forward(hidden_states, timestep, encoder_hidden_states,
                     pooled_projections, guidance, img_ids, txt_ids)` to
    match `diffusers.Flux2Pipeline`'s call.
    `pooled_projections` is accepted and IGNORED (FLUX.2 has no pooled path);
    it's kept in the signature so the same pipeline code paths work.
    """

    def __init__(
        self,
        config: NeuronFlux2Config,
        model_cls=NeuronFlux2Transformer,
        tag: str = "NeuronFlux2Transformer",
        compiler_args: Optional[str] = None,
        priority_model_idx: Optional[int] = None,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs or {}
        )
        self.bucket_config = None
        self.image_rotary_emb = None
        self.cache_image_rotary_emb = False
        self.pos_embed = NeuronFlux2RotaryEmbedding(
            theta=config.rope_theta, axes_dim=config.axes_dims_rope
        )

    # NxDI uses this to generate trace inputs for compile. Without compile
    # today we still need the shapes to be correct so future sessions can
    # flip it on unchanged.
    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        C = self.config
        dtype = C.neuron_config.torch_dtype
        num_patches = (
            C.height * C.width // ((2 * C.vae_scale_factor) ** 2)  # VERIFY vae/patch math
        )
        batch = 2 if C.cfg_parallel_enabled else 1

        # Text seq len: choose a representative max (Mistral-3 tokenized).
        text_seq_len = 512  # VERIFY against pipeline default for FLUX.2

        head_dim = C.attention_head_dim

        inputs = (
            torch.randn([batch, num_patches, C.in_channels], dtype=dtype),
            torch.randn([batch, text_seq_len, C.joint_attention_dim], dtype=dtype),
            torch.randn([batch], dtype=dtype),                 # timestep
            (torch.randn([batch], dtype=dtype)                 # guidance
             if C.guidance_embeds else torch.tensor([], dtype=dtype)),
            torch.randn([num_patches + text_seq_len, head_dim, 2], dtype=dtype),  # image_rotary_emb (cos, sin stacked)
        )
        return [inputs]

    def get_model_instance(self):
        def _create_model():
            m = self.model_cls(self.config)
            m = m.to(dtype=self.config.neuron_config.torch_dtype)
            m.eval()
            return m

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor] = None,  # ignored
        guidance: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,  # ignored
        return_dict: bool = False,
    ):
        """
        Host-side entry matching `Flux2Pipeline.__call__`'s invocation of
        the transformer. Builds the rotary table and calls the compiled
        (in the future) inner model.
        """
        if self.model is None:
            raise RuntimeError("Forward called before load/compile. Call .load() first.")

        dtype = self.config.neuron_config.torch_dtype
        timestep = timestep.to(dtype)

        if guidance is None:
            guidance_t = torch.tensor([], dtype=dtype)
        elif not isinstance(guidance, torch.Tensor):
            guidance_t = torch.tensor([guidance], dtype=dtype)
        else:
            guidance_t = guidance.to(dtype)

        # Pool projection path does not exist in FLUX.2; ignore silently.
        del pooled_projections, joint_attention_kwargs

        # Build rotary table exactly like diffusers does: compute per-axis,
        # concat over axes, then concatenate txt_ids and img_ids along the
        # sequence axis.
        image_rotary_emb = self.image_rotary_emb
        if image_rotary_emb is None:
            if txt_ids is None or img_ids is None:
                raise ValueError("Either image_rotary_emb or both (img_ids, txt_ids) must be provided.")
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            # FLUX.2 ids must have 4 columns (len(axes_dim) == 4)
            assert txt_ids.shape[-1] == len(self.config.axes_dims_rope), (
                f"txt_ids last-dim {txt_ids.shape[-1]} != "
                f"len(axes_dims_rope)={len(self.config.axes_dims_rope)}"
            )
            ids = torch.cat((txt_ids, img_ids), dim=0)  # txt-first to match diffusers
            cos, sin = self.pos_embed(ids)
            image_rotary_emb = torch.stack([cos, sin], dim=-1).to(dtype=dtype)  # [S, head_dim, 2]
        if self.cache_image_rotary_emb:
            self.image_rotary_emb = image_rotary_emb

        output = self._forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            guidance_t,
            image_rotary_emb,
        )
        return output


class NeuronFlux2BackboneApplication(NeuronApplicationBase):
    """Top-level application entry used by pipeline code."""

    _model_cls = NeuronFlux2Transformer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_parallel_enabled = getattr(self.config, "context_parallel_enabled", False)
        self.cfg_parallel_enabled = getattr(self.config, "cfg_parallel_enabled", False)
        self.model = NeuronFlux2ForDiffusion(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def forward(self, *args, **kwargs):
        return self.models[0](*args, **kwargs)

    def get_compiler_args(self) -> str:
        # Baseline args mirror NxDI Flux.1. On PyTorch 2.9 / SDK 2.27+ the
        # `--auto-cast=matmult` setting is required for FP32 paths; our DiT
        # is BF16 so `--auto-cast=none` is correct here.
        compiler_args = (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        if _HARDWARE == hardware.TRN2:
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # No tied weights in FLUX.2 DiT.
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: Dict[str, torch.Tensor], config: NeuronFlux2Config) -> Dict[str, torch.Tensor]:
        return convert_hf_to_neuron_state_dict(state_dict, config)
