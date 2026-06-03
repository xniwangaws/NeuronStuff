# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==========================================================================
# NeuronX Distributed Inference port of Google's Gemma-4-26B-A4B-it.
# ==========================================================================
#
# This is a *dry-run* port: it generates the Python implementation that should
# compile against AWS Trainium when paired with NxDI 2.27+. No code in this
# file has been run on hardware. See ``agent_artifacts/traces/architecture_analysis.md``
# for the architectural reasoning behind the choices below, and ``README.md``
# in this directory for what was reused vs ported fresh.
#
# Reference HF source:
#     transformers_src/src/transformers/models/gemma4/modeling_gemma4.py
#
# Naming convention (per skill instructions): a class with no 1:1 mapping to
# HuggingFace gets a ``_u`` suffix.

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# NxDI / NxD imports — kept at the top level (this file is only imported
# inside the venv that has them).
# ---------------------------------------------------------------------------

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# MoE block: NxDI ships two implementations (moe.py + moe_v2.py). Per the
# system prompt we use moe_v2.
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

from .configuration_gemma4_neuron import (
    make_gemma4_inference_config_class,
    make_gemma4_neuron_config_class,
)


# ---------------------------------------------------------------------------
# 0. Materialise the runtime config classes (factories defined in
# configuration_gemma4_neuron.py — see that file's docstring).
# ---------------------------------------------------------------------------

Gemma4InferenceConfig = make_gemma4_inference_config_class()
Gemma4NeuronConfig = make_gemma4_neuron_config_class()


# ---------------------------------------------------------------------------
# 1. Normalisation helpers
# ---------------------------------------------------------------------------
#
# Gemma4 uses ``Gemma4RMSNorm`` everywhere. The genericmoe v16 knowledge-base
# entry however found that on Neuron hardware the *outer* norms (input,
# post-attention, pre-/post-MLP, final) must be ``nn.LayerNorm`` to avoid
# gibberish output, while inner norms (q_norm, k_norm, router norm) are fine
# as RMSNorm.
#
# We expose two helpers so both stay obvious in the code below.


def _outer_norm(hidden_size: int, eps: float) -> nn.Module:
    """LayerNorm for outer normalisations (decoder, final).

    KB-driven: see ``genericmoe_v16_final_success_summary.md`` — outer
    LayerNorm prevents activation drift on Neuron.
    """

    return nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)


def _inner_norm(hidden_size: int, eps: float, with_scale: bool = True) -> nn.Module:
    """RMSNorm for inner normalisations (Q/K/V/router).

    Uses NxDI's ``CustomRMSNorm`` for the affine variant (matches gemma4's
    ``with_scale=True``). For ``with_scale=False`` (gemma4's v_norm and the
    router input norm) we fall back to a plain non-affine RMSNorm.
    """

    if with_scale:
        return CustomRMSNorm(hidden_size, eps=eps)

    return _RMSNormNoScale_u(hidden_size, eps=eps)


class _RMSNormNoScale_u(nn.Module):
    """Centring-only RMSNorm (no learned scale).

    Mirrors gemma4's ``Gemma4RMSNorm(..., with_scale=False)``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.float()
        ms = hidden_states_f32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = hidden_states_f32 * torch.pow(ms, -0.5)
        return normed.to(in_dtype)


# ---------------------------------------------------------------------------
# 2. Dense MLP (Gemma4TextMLP)
# ---------------------------------------------------------------------------
#
# Plain SwiGLU with `gelu_pytorch_tanh` activation. ``up_proj`` and ``gate_proj``
# are kept separate so the existing HF checkpoint maps directly. NxDI's
# parallel layers handle the TP sharding.


class NeuronGemma4MLP(nn.Module):
    """Dense feed-forward block (mirrors ``Gemma4TextMLP``)."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Gemma4 has an optional "double-wide MLP" on KV-shared layers; for
        # 26B-A4B ``num_kv_shared_layers=0`` so this stays at intermediate_size.
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        )
        is_kv_shared_layer = (
            getattr(config, "num_kv_shared_layers", 0) > 0
            and layer_idx >= first_kv_shared_layer_idx
        )
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)

        dtype = config.neuron_config.torch_dtype

        # Column-parallel: hidden -> intermediate (gate, up split inside)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
        )

        if config.hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            # Fallback path; gemma4 always specifies gelu_pytorch_tanh
            self.act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ---------------------------------------------------------------------------
# 3. Router (Gemma4TextRouter)
# ---------------------------------------------------------------------------
#
# Has two gemma4-specific learned tensors that NxDI's standard ``RouterTopK``
# does not carry: ``scale`` (per-hidden-dim) and ``per_expert_scale``. We
# therefore implement the router fresh with the ``_u`` suffix. Output mirrors
# what ``initialize_moe_module`` expects.


class NeuronGemma4Router_u(nn.Module):
    """Top-K router with gemma4's scaling tweaks. No 1:1 NxDI equivalent."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts

        # Internal RMSNorm (no scale) — gemma4 source line 1342
        self.norm = _inner_norm(self.hidden_size, self.eps, with_scale=False)

        # Projection — replicated across TP ranks so every rank computes the
        # same routing decision. (NxDI MoE expert dispatch needs identical
        # routing on every rank, otherwise tokens get routed differently per
        # rank and the all-to-all goes wrong.)
        self.proj = nn.Linear(self.hidden_size, config.num_experts, bias=False, dtype=torch.float32)

        # Learned scale parameters (gemma4-specific)
        self.scale = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float32))
        self.per_expert_scale = nn.Parameter(
            torch.ones(config.num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FP32 routing (KB: critical for MoE numerical stability)
        h = hidden_states.float()
        h = self.norm(h)
        h = h * self.scale * self.scalar_root_size

        scores = self.proj(h)  # [N, E]
        probs = F.softmax(scores, dim=-1)

        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)

        # Per-token re-normalisation so weights sum to 1 (gemma4 source line 1362)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Apply learned per-expert scale (gemma4 source line 1365)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return probs, top_k_weights, top_k_index


# ---------------------------------------------------------------------------
# 4. MoE block — wraps NxDI's ``initialize_moe_module``.
# ---------------------------------------------------------------------------
#
# We delegate the heavy lifting (expert dispatch, sharded gate_up_proj /
# down_proj, all-to-all routing, etc.) to NxDI's ``MoE`` v2 module. The only
# gemma4-specific bit is the router above.


class NeuronGemma4MoEBlock_u(nn.Module):
    """Composes Gemma4 router + NxDI expert dispatch."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.router = NeuronGemma4Router_u(config)

        # ``initialize_moe_module`` returns an `MoE` module with sharded
        # ``ExpertMLPs``. Routing is delegated back to ``self.router`` via
        # the ``router_topk_fn`` hook (some NxDI versions call this hook
        # ``router_fn`` — keep both names in mind when iterating).
        self.expert_mlps = initialize_moe_module(
            config=config,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size or config.intermediate_size,
            hidden_act=config.hidden_activation,
            normalize_top_k_affinities=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args: ``hidden_states`` of shape [B, S, H]. Returns same shape."""

        bsz, seqlen, hsz = hidden_states.shape
        flat = hidden_states.reshape(-1, hsz)

        _probs, top_k_weights, top_k_index = self.router(flat)

        # NxDI's MoE module signature varies by version; the common one
        # accepts (hidden, top_k_index, top_k_weights). We forward in the
        # canonical shape.
        out = self.expert_mlps(flat, top_k_index, top_k_weights)

        return out.reshape(bsz, seqlen, hsz)


# ---------------------------------------------------------------------------
# 5. Attention (Gemma4TextAttention) — fused class for sliding + full layers.
# ---------------------------------------------------------------------------
#
# Gemma4 has two attention shapes. We allocate **separate** Q/K/V/o linear
# layers per layer (different head_dim, different num_kv_heads), but reuse a
# single RotaryEmbedding per layer_type that lives at the model level (passed
# down through ``position_embeddings``).
#
# Why we still inherit from `NeuronAttentionBase`: it gives us KV cache
# management, flash-attention kernel selection, and the GQA sharding logic
# for free. We override the parts of ``__init__`` that need per-layer head_dim.


class NeuronGemma4Attention_u(NeuronAttentionBase):
    """Hybrid sliding/full attention. ``_u`` because the K=V case has no 1:1.

    The layer's ``layer_type`` ("sliding_attention" or "full_attention") is
    read from ``config.layer_types[layer_idx]`` to pick:
      * head_dim          (256 vs 512)
      * num_key_value_heads (4 vs 2)
      * sliding_window    (1024 vs None)
      * partial RoPE      (full layers rotate first 25 % only)
      * K=V               (full layers reuse k_proj as v_proj)
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        layer_types = getattr(config, "layer_types", None) or []
        self.layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_alternative_attention = (
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )

        # Per-layer-type head sizing
        if self.is_sliding:
            head_dim = config.head_dim
            num_kv_heads = config.num_key_value_heads
            sliding_window = config.sliding_window
        else:
            head_dim = getattr(config, "global_head_dim", None) or config.head_dim
            num_kv_heads = getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads
            sliding_window = None

        rope_params = config.rope_parameters[self.layer_type]
        rope_theta = rope_params.get("rope_theta", 10_000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

        # NxDI's RotaryEmbedding takes the rotated dimension directly. For
        # partial RoPE, that's ``head_dim * partial_rotary_factor``.
        rotated_dim = int(head_dim * partial_rotary_factor)
        # Ensure even (rotate-half requires this)
        rotated_dim = (rotated_dim // 2) * 2

        rotary_emb = RotaryEmbedding(
            dim=rotated_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            sliding_window=sliding_window,
        )

        # Q/K RMSNorm (Qwen3-style; gemma4 source line 1210, 1214)
        self.q_norm = _inner_norm(head_dim, config.rms_norm_eps, with_scale=True)
        self.k_norm = _inner_norm(head_dim, config.rms_norm_eps, with_scale=True)
        # V RMSNorm without scale (centring only)
        self.v_norm = _inner_norm(head_dim, config.rms_norm_eps, with_scale=False)

        # Sentinel so the state-dict converter knows whether to fold v_proj
        self._k_eq_v = self.use_alternative_attention

        # The base class does not know about per-layer rotated_dim for
        # partial RoPE; stash it for our forward override.
        self._rotated_dim = rotated_dim
        self._head_dim = head_dim

    # NOTE: For NxDI, the `forward` is mostly inherited from
    # ``NeuronAttentionBase``. We only override the *Q/K post-projection*
    # path so that q_norm / k_norm and partial-RoPE land in the right place.
    #
    # NxDI's base attention exposes hooks (in 2.27+: ``_apply_qk_norm`` and
    # ``_apply_rotary_pos_emb``) that we can override; if a future SDK
    # drops them we will need a full-forward override. See
    # ``OVERRIDING_FORWARD_GUIDANCE.md``.

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head RMSNorm to Q and K *before* RoPE.

        Hooks into NxDI's NeuronAttentionBase post-projection step.
        """

        return self.q_norm(q), self.k_norm(k)

    def _apply_v_norm(self, v: torch.Tensor) -> torch.Tensor:
        """Apply v_norm (gemma4 source line 1265)."""

        return self.v_norm(v)


# ---------------------------------------------------------------------------
# 6. Per-Layer-Embeddings (PLE) helper
# ---------------------------------------------------------------------------
#
# Gemma4 maintains a *second* embedding table that contributes a residual
# signal at every decoder layer. There is no NxDI module that does this; we
# implement it fresh.


class Gemma4PLE_u(nn.Module):
    """Computes per-layer input residuals from input_ids + inputs_embeds.

    Returns a tensor of shape [B, S, num_hidden_layers, hidden_size_per_layer_input].
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.vocab_size_per_layer_input = config.vocab_size_per_layer_input

        # The packed embedding ``[V_ple, L * D_ple]``. Replicated across TP for
        # v0; in a follow-up we may shard along the packed dim.
        self.embed_tokens_per_layer = nn.Embedding(
            self.vocab_size_per_layer_input,
            self.num_hidden_layers * self.hidden_size_per_layer_input,
            padding_idx=config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
        )
        self._embed_scale = self.hidden_size_per_layer_input**0.5

        # Context-aware projection from the main residual stream
        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            self.num_hidden_layers * self.hidden_size_per_layer_input,
            bias=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self._projection_scale = self.hidden_size**-0.5
        self.per_layer_projection_norm = _inner_norm(
            self.hidden_size_per_layer_input, config.rms_norm_eps, with_scale=True
        )
        self.per_layer_input_scale = 2.0**-0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Token-identity component
        ple_tokens = self.embed_tokens_per_layer(input_ids) * self._embed_scale
        ple_tokens = ple_tokens.reshape(
            *input_ids.shape, self.num_hidden_layers, self.hidden_size_per_layer_input
        )

        # Context component
        proj = self.per_layer_model_projection(inputs_embeds) * self._projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1], self.num_hidden_layers, self.hidden_size_per_layer_input
        )
        proj = self.per_layer_projection_norm(proj)

        return (proj + ple_tokens) * self.per_layer_input_scale


# ---------------------------------------------------------------------------
# 7. Decoder layer (Gemma4TextDecoderLayer)
# ---------------------------------------------------------------------------


class NeuronGemma4DecoderLayer(nn.Module):
    """One Gemma4 decoder layer (dense MLP + optional MoE branch + optional PLE)."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = NeuronGemma4Attention_u(config=config, layer_idx=layer_idx)
        self.mlp = NeuronGemma4MLP(config, layer_idx)

        # Outer norms: LayerNorm (KB-driven; see _outer_norm docstring).
        self.input_layernorm = _outer_norm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _outer_norm(self.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _outer_norm(self.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _outer_norm(self.hidden_size, config.rms_norm_eps)

        self.register_buffer("layer_scalar", torch.ones(1))

        self.enable_moe_block = bool(getattr(config, "enable_moe_block", False))
        if self.enable_moe_block:
            self.moe_block = NeuronGemma4MoEBlock_u(config)
            # Extra norms specific to the MoE branch (gemma4 source 1395-1397)
            self.post_feedforward_layernorm_1 = _outer_norm(self.hidden_size, config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = _outer_norm(self.hidden_size, config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = _outer_norm(self.hidden_size, config.rms_norm_eps)

        self.use_per_layer = bool(getattr(config, "hidden_size_per_layer_input", 0))
        if self.use_per_layer:
            if config.hidden_activation == "gelu_pytorch_tanh":
                self._ple_act = lambda x: F.gelu(x, approximate="tanh")
            else:
                self._ple_act = F.gelu
            self.per_layer_input_gate = nn.Linear(
                self.hidden_size,
                config.hidden_size_per_layer_input,
                bias=False,
                dtype=config.neuron_config.torch_dtype,
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                dtype=config.neuron_config.torch_dtype,
            )
            self.post_per_layer_input_norm = _outer_norm(self.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        per_layer_input: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states

        # ---- attention block (post-norm style: norm-attn-norm-residual) ----
        h = self.input_layernorm(hidden_states)
        h, _ = self.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            **kwargs,
        )
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        # ---- feed-forward block (dense MLP + optional MoE in parallel) ----
        residual = hidden_states
        h_pre = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(h_pre)

        if self.enable_moe_block:
            mlp_branch = self.post_feedforward_layernorm_1(mlp_out)

            # MoE branch reads the *pre-MLP residual* (gemma4 source 1433)
            moe_in = self.pre_feedforward_layernorm_2(residual)
            moe_out = self.moe_block(moe_in)
            moe_branch = self.post_feedforward_layernorm_2(moe_out)

            ff = mlp_branch + moe_branch
        else:
            ff = mlp_out

        ff = self.post_feedforward_layernorm(ff)
        hidden_states = residual + ff

        # ---- per-layer-embedding residual ----
        if self.use_per_layer and per_layer_input is not None:
            residual = hidden_states
            g = self.per_layer_input_gate(hidden_states)
            g = self._ple_act(g)
            g = g * per_layer_input
            g = self.per_layer_projection(g)
            g = self.post_per_layer_input_norm(g)
            hidden_states = residual + g

        # Final per-layer scalar (learned)
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# 8. Top-level model
# ---------------------------------------------------------------------------


class NeuronGemma4Model(NeuronBaseModel):
    """The Gemma4 text decoder backbone."""

    def setup_attr_for_model(self, config: Any) -> None:
        # Standard NxDI attributes
        self.on_device_sampling = getattr(config.neuron_config, "on_device_sampling_config", None) is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        self.num_hidden_layers = config.num_hidden_layers

    def init_inference_optimization(self, config: Any) -> None:
        # No-op for v0; placeholder for sampling head, etc.
        return

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.config = config

        dtype = config.neuron_config.torch_dtype

        # Embedding (scaled by sqrt(hidden_size) — see Gemma4TextScaledWordEmbedding)
        self.embed_tokens = ParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=dtype,
        )
        self.register_buffer(
            "embed_scale",
            torch.tensor(config.hidden_size**0.5, dtype=dtype),
            persistent=False,
        )

        # Per-Layer-Embeddings (PLE) — optional
        self.use_per_layer = bool(getattr(config, "hidden_size_per_layer_input", 0))
        if self.use_per_layer:
            self.ple = Gemma4PLE_u(config)

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        # Final norm (LayerNorm per the KB)
        self.norm = _outer_norm(config.hidden_size, config.rms_norm_eps)

        # LM head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=not self.on_device_sampling,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 1. Token embedding (scale by sqrt(hidden_size))
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # 2. Per-layer-input precomputation (PLE)
        per_layer_inputs = None
        if self.use_per_layer:
            per_layer_inputs = self.ple(input_ids=input_ids, inputs_embeds=inputs_embeds)

        hidden_states = inputs_embeds

        # 3. Decoder stack
        for i, layer in enumerate(self.layers):
            per_layer_input_i = (
                per_layer_inputs[..., i, :] if per_layer_inputs is not None else None
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                per_layer_input=per_layer_input_i,
                **kwargs,
            )

        # 4. Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# 9. Causal-LM wrapper (entrypoint for compile / inference tools)
# ---------------------------------------------------------------------------


class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """Top-level model class — the one passed to ``compile_neuron_model``."""

    _model_cls = NeuronGemma4Model

    @classmethod
    def get_config_cls(cls):  # type: ignore[override]
        return Gemma4InferenceConfig

    @staticmethod
    def load_hf_model(model_path: str):  # pragma: no cover - hardware path
        # Standard pattern: defer to AutoModelForCausalLM. Used by NxDI's
        # weight-conversion utilities. Listed here so the inference runner
        # can find it.
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: dict, config: Any) -> dict:
        """Mirror lm_head.weight from embed_tokens.weight (gemma4 ties them)."""

        if getattr(config, "tie_word_embeddings", False):
            embed_key = None
            for candidate in ("model.embed_tokens.weight", "embed_tokens.weight"):
                if candidate in state_dict:
                    embed_key = candidate
                    break
            if embed_key is not None and "lm_head.weight" not in state_dict:
                state_dict["lm_head.weight"] = state_dict[embed_key]
        return state_dict

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Any) -> dict:
        """Adapt the HuggingFace gemma4 checkpoint to NxDI key naming.

        The transformations are:
          1. Strip ``model.`` prefix where NxDI expects bare keys.
          2. For full-attention layers with ``attention_k_eq_v=True``, copy
             ``k_proj.weight`` into ``v_proj.weight`` (NxDI's base attention
             always allocates a v_proj; the K=V trick is implemented at the
             weight-loading level for v0).
          3. For MoE layers, the gemma4 source already stores the experts as
             packed tensors ``gate_up_proj`` (E, 2I, H) and ``down_proj``
             (E, H, I). NxDI's MoE module reads those names directly when
             initialised from ``initialize_moe_module`` — nothing to do.
          4. Drop ``v_norm.weight`` keys for layers where ``with_scale=False``
             (the param doesn't exist in HF either; defensive).
          5. Apply tied-weight expansion (delegated to
             ``update_state_dict_for_tied_weights``).
        """

        new_sd = {}
        for k, v in state_dict.items():
            new_k = k[len("model.") :] if k.startswith("model.") else k
            new_sd[new_k] = v

        layer_types = getattr(config, "layer_types", None) or []
        attention_k_eq_v = getattr(config, "attention_k_eq_v", False)

        if attention_k_eq_v:
            # For each full-attention layer that has no v_proj in HF, copy k_proj.
            # gemma4 source line 1220-1224: ``v_proj = ... if not use_alternative_attention else None``
            for i, lt in enumerate(layer_types):
                if lt != "full_attention":
                    continue
                k_key = f"layers.{i}.self_attn.k_proj.weight"
                v_key = f"layers.{i}.self_attn.v_proj.weight"
                if k_key in new_sd and v_key not in new_sd:
                    new_sd[v_key] = new_sd[k_key].clone()

        # Tied lm_head
        new_sd = NeuronGemma4ForCausalLM.update_state_dict_for_tied_weights(new_sd, config)
        return new_sd


__all__ = [
    "Gemma4InferenceConfig",
    "Gemma4NeuronConfig",
    "NeuronGemma4Attention_u",
    "NeuronGemma4DecoderLayer",
    "NeuronGemma4ForCausalLM",
    "NeuronGemma4MLP",
    "NeuronGemma4Model",
    "NeuronGemma4MoEBlock_u",
    "NeuronGemma4Router_u",
    "Gemma4PLE_u",
]
