# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file wraps Google's Gemma4 text-only configuration so that it can be
# consumed by the NeuronX Distributed Inference (NxDI) framework. It is a
# *thin* adapter: every value still comes from the underlying HuggingFace
# `Gemma4TextConfig` (see ``transformers_src/.../configuration_gemma4.py``),
# we just add the attributes that NxDI's `InferenceConfig` / `MoENeuronConfig`
# expect, and we hard-pin the values that the porting knowledge base flagged
# as load-bearing (tie_word_embeddings, output_attentions, ...).
#
# This module is import-safe even without the HF transformers source: it
# contains a small *compatibility-shim* `Gemma4TextConfig` so the file can be
# read and validated on a machine that does not have the upstream HF gemma4
# branch installed. When transformers ships gemma4 officially the shim is
# unused and the real `transformers.models.gemma4.configuration_gemma4` is
# imported instead.

from __future__ import annotations

from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Locate the HuggingFace Gemma4TextConfig.
# ---------------------------------------------------------------------------
# Try the upstream HF location first; fall back to a small shim with all the
# fields the porting code actually reads. The shim deliberately mirrors the
# subset documented in agent_artifacts/traces/architecture_analysis.md.

try:
    from transformers.models.gemma4.configuration_gemma4 import (  # type: ignore[import-not-found]
        Gemma4TextConfig as _HFGemma4TextConfig,
    )
except Exception:  # pragma: no cover - shim path
    _HFGemma4TextConfig = None  # noqa: N816


class _Gemma4TextConfigShim:
    """Local fallback that mirrors the fields of `Gemma4TextConfig` we read.

    Only used when the upstream HF gemma4 module is not yet importable. Every
    field below is set from the HF `config.json` shipped with the model so the
    NxDI compile path has the values it needs.
    """

    model_type = "gemma4_text"

    # Sane defaults (match HF Gemma4TextConfig); ``from_pretrained`` overrides
    # them with the actual per-checkpoint values.
    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = 0
    eos_token_id: Any = 1
    bos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = True
    rope_parameters: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 1024
    layer_types: Optional[list] = None
    final_logit_softcapping: Optional[float] = None
    use_bidirectional_attention: Optional[str] = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: Optional[int] = 2
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._post_init()

    def _post_init(self) -> None:
        if self.layer_types is None:
            sliding_window_pattern = 6  # 5 sliding : 1 full
            self.layer_types = [
                "sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
            # last layer must be full_attention (matches HF post-init)
            self.layer_types[-1] = "full_attention"

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }


# Pick whichever is available
Gemma4TextConfig = _HFGemma4TextConfig if _HFGemma4TextConfig is not None else _Gemma4TextConfigShim


# ---------------------------------------------------------------------------
# NxDI imports — these are deferred to function-call time because this file
# also needs to be readable by static tooling that does not have NxDI.
# ---------------------------------------------------------------------------


def _import_inference_config():
    from neuronx_distributed_inference.models.config import InferenceConfig  # noqa: PLC0415

    return InferenceConfig


def _import_moe_neuron_config():
    from neuronx_distributed_inference.models.config import MoENeuronConfig  # noqa: PLC0415

    return MoENeuronConfig


# ---------------------------------------------------------------------------
# Public API: Gemma4InferenceConfig + Gemma4NeuronConfig
# ---------------------------------------------------------------------------


def make_gemma4_inference_config_class():
    """Factory: returns an `Gemma4InferenceConfig` class subclassed from NxDI.

    Wrapped in a factory so that a machine without NxDI installed (this dev
    laptop, for example) can still parse this file. The class is only
    materialised when the compile/inference scripts call it.
    """

    InferenceConfig = _import_inference_config()

    class Gemma4InferenceConfig(InferenceConfig):
        """NxDI inference config for Gemma-4 text decoder (incl. MoE variants).

        Extends NxDI's ``InferenceConfig`` so that the framework can find every
        attribute it reads at compile and runtime. All fields are sourced from
        the underlying HF ``Gemma4TextConfig``; we never hand-edit values
        beyond the few ``hard-pin`` cases (tie_word_embeddings) explicitly
        required by the porting knowledge base.
        """

        def get_required_attributes(self) -> list:  # type: ignore[override]
            # Listed for the framework's introspection — mirrors what we set
            # from the HF config below.
            return [
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "num_hidden_layers",
                "vocab_size",
                "max_position_embeddings",
                "rms_norm_eps",
                "head_dim",
                "sliding_window",
                "layer_types",
                "rope_parameters",
                "tie_word_embeddings",
                "hidden_activation",
                "intermediate_size",
            ]

        def add_derived_config(self) -> None:  # type: ignore[override]
            """Populate framework-required attributes that HF doesn't expose.

            Keep additive only — never overwrite a value the HF config set.
            """

            super_add = getattr(super(), "add_derived_config", None)
            if callable(super_add):
                super_add()

            # head_dim used everywhere in NxDI base attention
            if not hasattr(self, "head_dim") or self.head_dim is None:
                self.head_dim = self.hidden_size // self.num_attention_heads

            # NxDI cores-per-group default
            if not hasattr(self, "num_cores_per_group"):
                self.num_cores_per_group = 1

            # Standard HF inference attributes that NxDI checks for
            for attr, default in (
                ("output_attentions", False),
                ("output_hidden_states", False),
                ("use_return_dict", True),
                ("return_dict", True),
                ("use_cache", True),
            ):
                if not hasattr(self, attr):
                    setattr(self, attr, default)

            # CRITICAL: gemma4 ties lm_head <-> embed_tokens. The HF config
            # already sets this True, but pin it defensively (matches the
            # genericmodel knowledge-base lesson #1).
            self.tie_word_embeddings = True

            # MoE-side attributes that the NxDI MoE module reads
            if getattr(self, "enable_moe_block", False):
                if not hasattr(self, "num_local_experts"):
                    self.num_local_experts = self.num_experts
                if not hasattr(self, "num_experts_per_tok"):
                    self.num_experts_per_tok = self.top_k_experts
                if not hasattr(self, "norm_topk_prob"):
                    # Gemma4 router *does* normalise the top-k weights to sum
                    # to 1 (see Gemma4TextRouter.forward).
                    self.norm_topk_prob = True

            # Per-layer-embedding (PLE) toggle — read by the model wrapper
            self.use_per_layer_embeddings = bool(getattr(self, "hidden_size_per_layer_input", 0))

        @classmethod
        def from_hf_config(cls, hf_config: Any, neuron_config: Any) -> "Gemma4InferenceConfig":
            """Build an inference config directly from a HuggingFace config.

            Mirrors the shape used by NxDI's `Qwen3MoeInferenceConfig.from_pretrained`.
            """

            kwargs = {
                k: v
                for k, v in hf_config.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            # Hard-pin tied embeddings (see comment in add_derived_config)
            kwargs["tie_word_embeddings"] = True
            return cls(neuron_config=neuron_config, **kwargs)

    return Gemma4InferenceConfig


def make_gemma4_neuron_config_class():
    """Factory for ``Gemma4NeuronConfig`` (subclass of ``MoENeuronConfig``).

    The reason we subclass at runtime is the same as for the inference config:
    keep the file importable on machines without NxDI.
    """

    MoENeuronConfig = _import_moe_neuron_config()

    class Gemma4NeuronConfig(MoENeuronConfig):
        """Sets ``attn_cls`` to the gemma4 attention class.

        Per the genericmodel knowledge-base finding (Issue 3): if you do not
        subclass NeuronConfig and explicitly set ``attn_cls`` then NxDI will
        silently fall back to ``NeuronLlamaAttention``, which is wrong.
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            # Late import to avoid a circular dependency at module-load time.
            from .modeling_gemma4_neuron import (  # noqa: PLC0415
                NeuronGemma4Attention_u,
            )

            self.attn_cls = NeuronGemma4Attention_u

            # bf16 throughout. ``use_fp16=True`` in the compiler also flips
            # this to bfloat16 (NxDI's naming quirk noted in the knowledge
            # base), but we set torch_dtype explicitly anyway.
            if not hasattr(self, "torch_dtype") or self.torch_dtype is None:
                self.torch_dtype = torch.bfloat16

            # MoE-specific: keep router compute in FP32 for numerical stability
            # (genericmoe knowledge-base "Critical Implementation Decision #4").
            router_cfg = getattr(self, "router_config", None)
            if router_cfg is not None:
                if hasattr(router_cfg, "dtype"):
                    router_cfg.dtype = torch.float32
                if hasattr(router_cfg, "act_fn"):
                    router_cfg.act_fn = "softmax"

    return Gemma4NeuronConfig


__all__ = [
    "Gemma4TextConfig",
    "make_gemma4_inference_config_class",
    "make_gemma4_neuron_config_class",
]
