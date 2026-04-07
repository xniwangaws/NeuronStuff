"""
Qwen2.5-VL vision encoder for NeuronX Distributed Inference.

Key differences from Qwen2-VL vision (qwen2_vl/modeling_qwen2_vl_vision.py):
1. RMSNorm instead of LayerNorm in blocks and merger
2. SwiGLU MLP (gate_proj/up_proj/down_proj) instead of GELU MLP (fc1/fc2)
3. merger.ln_q uses RMSNorm (no bias)
"""

import os
import logging

import torch
from torch import nn
from safetensors.torch import save_file
from typing import List, Optional, Tuple

from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding, PatchEmbed
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.padding import pad_tensor, pad_with_first_batchline
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
from neuronx_distributed_inference.models.qwen2_vl.utils.vision_utils import (
    calculate_max_grid_size, get_image_dimensions
)
from neuronx_distributed_inference.models.qwen2_vl.utils.input_processor import prepare_generation_inputs_hf

# Reuse unchanged components from qwen2_vl
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import (
    Qwen2VLVisionRotaryEmbedding,
    NeuronQwen2VLAttention,
    Qwen2VLVisionModelWrapper,
    NeuronQwen2VLForImageEncoding,
)

logger = logging.getLogger(__name__)


class Qwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2.5-VL vision encoder (weight only, no bias)."""
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionSwiGLUMlp(nn.Module):
    """SwiGLU MLP for Qwen2.5-VL vision blocks (replaces GELU fc1/fc2)."""
    def __init__(self, dim, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=True, gather_output=False, dtype=dtype
        )
        self.up_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=True, gather_output=False, dtype=dtype
        )
        self.down_proj = RowParallelLinear(
            hidden_dim, dim, bias=True, input_is_parallel=True, dtype=dtype, reduce_dtype=dtype
        )

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class PatchMerger2_5(nn.Module):
    """PatchMerger for Qwen2.5-VL: uses RMSNorm instead of LayerNorm for ln_q."""
    def __init__(self, dim, context_dim, spatial_merge_size=2, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6, dtype=dtype)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size, self.hidden_size, gather_output=False, dtype=dtype
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size, dim, input_is_parallel=True, dtype=dtype, reduce_dtype=dtype
            )
        )

    def forward(self, x):
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5_VLVisionBlock(nn.Module):
    """Vision block for Qwen2.5-VL: RMSNorm + SwiGLU MLP."""
    def __init__(self, vision_config):
        super().__init__()
        dtype = vision_config.neuron_config.torch_dtype
        self.norm1 = Qwen2RMSNorm(vision_config.embed_dim, eps=1e-6, dtype=dtype)
        self.norm2 = Qwen2RMSNorm(vision_config.embed_dim, eps=1e-6, dtype=dtype)
        mlp_hidden_dim = int(vision_config.embed_dim * vision_config.mlp_ratio)
        self.attn = NeuronQwen2VLAttention(vision_config)
        self.mlp = VisionSwiGLUMlp(
            dim=vision_config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            dtype=dtype,
        )

    def forward(self, hidden_states, position_embeddings=None):
        attn_output = self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )[0]
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NeuronQwen2_5_VLVisionModel(nn.Module):
    """Qwen2.5-VL vision model with RMSNorm blocks and SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config

        self.spatial_merge_size = self.vision_config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            in_channels=self.vision_config.in_channels,
            embed_dim=self.vision_config.embed_dim,
        ).to(self.vision_config.neuron_config.torch_dtype)

        head_dim = self.vision_config.embed_dim // self.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(self.vision_config) for _ in range(self.vision_config.depth)]
        )
        self.merger = PatchMerger2_5(
            dim=getattr(self.vision_config, "out_hidden_size", self.vision_config.hidden_size),
            context_dim=self.vision_config.embed_dim,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )

        image_width, image_height = get_image_dimensions(self.vision_config.neuron_config)
        self.max_grid_size = calculate_max_grid_size(
            image_width, image_height, patch_size=self.vision_config.patch_size
        )
        logger.info("Calculated max_grid_size=%d for image dimensions %dx%d",
                     self.max_grid_size, image_width, image_height)

        self.precomputed_rotary_pos_emb = self.rotary_pos_emb(self.max_grid_size)
        self.register_buffer('rotary_pos_emb_cache', self.precomputed_rotary_pos_emb, persistent=False)

    def rot_pos_ids(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def pad_to_text_seq_len(self, hidden_states):
        padded_length = self.config.neuron_config.seq_len
        hidden_states = hidden_states.to(self.config.text_config.neuron_config.torch_dtype)
        hidden_size = hidden_states.shape[-1]
        hidden_states, _ = pad_tensor(hidden_states, (padded_length, hidden_size), pad_value=0)
        hidden_states = hidden_states.view(-1, hidden_size).unsqueeze(0)
        return hidden_states

    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)

        assert grid_thw[:, 1:].max() < self.max_grid_size, \
            "Grid size exceeds max_grid_size. Increase default_image_width/height in vision_neuron_config."
        pos_ids = self.rot_pos_ids(grid_thw)
        rotary_pos_emb = self.rotary_pos_emb_cache[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        cos_emb = cos_emb.reshape(grid_thw.shape[0], -1, cos_emb.shape[-1])
        sin_emb = sin_emb.reshape(grid_thw.shape[0], -1, sin_emb.shape[-1])
        position_embeddings = (cos_emb, sin_emb)

        hidden_states = hidden_states.reshape(grid_thw.shape[0], -1, hidden_states.shape[-1])
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)
        hidden_states_merger = self.merger(hidden_states)
        return self.pad_to_text_seq_len(hidden_states_merger)


class NeuronQwen2_5_VLForImageEncoding(NeuronQwen2VLForImageEncoding):
    """
    Qwen2.5-VL image encoding: uses the 2.5 vision model and state dict conversion.
    """

    _model_cls = NeuronQwen2_5_VLVisionModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        try:
            from transformers import Qwen2_5_VLConfig
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLVisionTransformerPretrainedModel,
            )
        except ImportError:
            from transformers import Qwen2VLConfig as Qwen2_5_VLConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VisionTransformerPretrainedModel as Qwen2_5_VLVisionTransformerPretrainedModel,
            )

        class hf_vision_model(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                self.hf_config = Qwen2_5_VLConfig.from_pretrained(model_path, **kwargs)
                hf_vision_config = Qwen2_5_VLConfig(**vars(self.hf_config.vision_config))
                self.visual = Qwen2_5_VLVisionTransformerPretrainedModel._from_config(hf_vision_config)

            def forward(self, pixel_values, grid_thw):
                return self.visual(pixel_values, grid_thw)

            def save_pretrained(self, save_model_path):
                self.hf_config.save_pretrained(save_model_path)
                save_file(self.state_dict(), os.path.join(save_model_path, "model.safetensors"))

        return hf_vision_model(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, inference_config):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                key = key.replace("visual.", "")
                # Attention key remapping (same as Qwen2-VL)
                if ".attn.qkv." in key:
                    key = key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
                elif ".attn.proj." in key:
                    key = key.replace(".attn.proj.", ".attn.o_proj.")
                # MLP keys (gate_proj/up_proj/down_proj) stay as-is
                # norm keys (weight only, no bias) stay as-is
            new_state_dict[key] = (
                value.clone()
                .detach()
                .contiguous()
                .to(inference_config.vision_config.neuron_config.torch_dtype)
            )

        del state_dict
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from qwen2_5_vl_7b.modeling_qwen2_5_vl import Qwen2_5_VLInferenceConfig
        return Qwen2_5_VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        if len(prompts) > 1:
            raise NotImplementedError("Qwen2.5-VL currently only supports batch size 1")
        if isinstance(prompts, list):
            prompts = prompts[0]
        if images and isinstance(images, list) and isinstance(images[0], list):
            images = images[0]
        inputs = prepare_generation_inputs_hf(prompts, images, processor, role, config)
        vision_inputs = None
        if hasattr(inputs, "pixel_values") and hasattr(inputs, "image_grid_thw"):
            vision_inputs = {
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw,
            }
        return inputs.input_ids, inputs.attention_mask, vision_inputs
