"""NeuronS3DiffUNet — full top-level UNet matching diffusers UNet2DConditionModel.

SD-Turbo config as used by S3Diff:
  in_channels=4, out_channels=4, sample_size=64
  block_out_channels=(320, 640, 1280, 1280)
  down_blocks: [CrossAttnDown, CrossAttnDown, CrossAttnDown, DownBlock2D]
  up_blocks:   [UpBlock2D, CrossAttnUp, CrossAttnUp, CrossAttnUp]
  mid_block:   UNetMidBlock2DCrossAttn (2 resnets + 1 attention)
  cross_attention_dim=1024, attention_head_dim=(5,10,20,20), num_layers_per_block=2,
    num_layers_up=3, transformer_layers_per_block=1

LoRA ranks: 32 everywhere. LoRA scaling: 0.25.

Note: de_mod buffers are threaded as a dict of tensors keyed by the full sub-module
path (e.g. "down_blocks.0.attentions.0.proj_in"). This matches how S3Diff's
outer forward assigns `module.de_mod = ...` — we just keep the semantics but
pass via argument for trace-compatibility later.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from de_mod_lora import DeModLoRAConv2d
from s3diff_unet_blocks import (
    NeuronS3DiffDownBlock2D,
    NeuronS3DiffCrossAttnDownBlock2D,
    NeuronS3DiffUNetMidBlock2DCrossAttn,
    NeuronS3DiffUpBlock2D,
    NeuronS3DiffCrossAttnUpBlock2D,
)


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int = 320,
                                    max_period: int = 10000,
                                    downscale_freq_shift: float = 0.0,
                                    scale: float = 1.0) -> torch.Tensor:
    """Matches diffusers' Timesteps module at freq_shift=0, scale=1, flip_sin_to_cos=True."""
    assert dim % 2 == 0
    half_dim = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :] * scale
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    # flip_sin_to_cos (diffusers default for UNet2DConditionModel): swap halves
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    return emb


class NeuronS3DiffTimeEmbedding(nn.Module):
    """diffusers TimestepEmbedding replacement — linear_1 -> SiLU -> linear_2."""

    def __init__(self, in_dim: int = 320, time_embed_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample):
        x = self.linear_1(sample)
        x = nn.functional.silu(x)
        x = self.linear_2(x)
        return x


class NeuronS3DiffUNet(nn.Module):
    """Top-level UNet replacement for diffusers.UNet2DConditionModel (SD-Turbo config).

    Forward signature is simplified:
      sample:               (B, 4, H, W) latent tile
      timestep:             (B,) or scalar
      encoder_hidden_states: (B, 77, 1024) CLIP text embed
      de_mod_map:           Dict[str, Tensor(B, r, r)] indexed by LoRA site path

    Returns: (B, 4, H, W) velocity / epsilon prediction (same shape as sample).
    """

    SAMPLE_SIZE = 64  # latent side = 64 for sd-turbo 512 output (we use 128 for 1K)
    IN_CHANNELS = 4
    OUT_CHANNELS = 4
    BLOCK_OUT_CHANNELS = (320, 640, 1280, 1280)
    LAYERS_PER_BLOCK = 2
    LAYERS_UP_PER_BLOCK = 3  # up_blocks each has 3 resnets
    CROSS_ATTN_DIM = 1024
    ATTN_HEAD_DIM = 64
    # per-resolution heads: 320/64=5, 640/64=10, 1280/64=20, 1280/64=20
    NUM_HEADS_PER_BLOCK = (5, 10, 20, 20)
    TIME_EMBED_DIM = 1280

    def __init__(self, lora_rank: int = 32, lora_scaling: float = 0.25,
                 norm_num_groups: int = 32):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_scaling = lora_scaling

        # --- conv_in (NO LoRA per S3Diff config) ---
        self.conv_in = nn.Conv2d(
            self.IN_CHANNELS, self.BLOCK_OUT_CHANNELS[0],
            kernel_size=3, padding=1, bias=True,
        )

        # --- time embedding (no LoRA) ---
        # time_proj is deterministic — computed in forward; not a submodule.
        self.time_embedding = NeuronS3DiffTimeEmbedding(
            in_dim=self.BLOCK_OUT_CHANNELS[0],
            time_embed_dim=self.TIME_EMBED_DIM,
        )

        # --- down_blocks ---
        self.down_blocks = nn.ModuleList()
        output_channel = self.BLOCK_OUT_CHANNELS[0]
        for i, bout in enumerate(self.BLOCK_OUT_CHANNELS):
            input_channel = output_channel
            output_channel = bout
            is_final = (i == len(self.BLOCK_OUT_CHANNELS) - 1)
            if i < 3:  # first 3 are CrossAttn, 4th is plain DownBlock2D
                self.down_blocks.append(NeuronS3DiffCrossAttnDownBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    temb_channels=self.TIME_EMBED_DIM, num_layers=self.LAYERS_PER_BLOCK,
                    num_attention_heads=self.NUM_HEADS_PER_BLOCK[i],
                    attention_head_dim=self.ATTN_HEAD_DIM,
                    cross_attention_dim=self.CROSS_ATTN_DIM,
                    lora_rank=lora_rank, lora_scaling=lora_scaling,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final,  # 3 downsamples for first 3 blocks
                ))
            else:  # plain
                self.down_blocks.append(NeuronS3DiffDownBlock2D(
                    in_channels=input_channel, out_channels=output_channel,
                    temb_channels=self.TIME_EMBED_DIM, num_layers=self.LAYERS_PER_BLOCK,
                    lora_rank=lora_rank, lora_scaling=lora_scaling,
                    resnet_groups=norm_num_groups,
                    add_downsample=False,
                ))

        # --- mid_block ---
        self.mid_block = NeuronS3DiffUNetMidBlock2DCrossAttn(
            in_channels=self.BLOCK_OUT_CHANNELS[-1],
            temb_channels=self.TIME_EMBED_DIM,
            num_layers=1,
            num_attention_heads=self.NUM_HEADS_PER_BLOCK[-1],
            attention_head_dim=self.ATTN_HEAD_DIM,
            cross_attention_dim=self.CROSS_ATTN_DIM,
            lora_rank=lora_rank, lora_scaling=lora_scaling,
            resnet_groups=norm_num_groups,
        )

        # --- up_blocks ---
        self.up_blocks = nn.ModuleList()
        reversed_bout = list(reversed(self.BLOCK_OUT_CHANNELS))
        reversed_heads = list(reversed(self.NUM_HEADS_PER_BLOCK))
        output_channel = reversed_bout[0]
        for i, bout in enumerate(reversed_bout):
            prev_output = output_channel
            output_channel = bout
            in_channels_this = self.BLOCK_OUT_CHANNELS[max(0, len(self.BLOCK_OUT_CHANNELS) - 2 - i)]
            is_final = (i == len(reversed_bout) - 1)
            if i == 0:  # plain UpBlock2D (matches diffusers: up_blocks[0])
                self.up_blocks.append(NeuronS3DiffUpBlock2D(
                    in_channels=in_channels_this,
                    prev_output_channels=prev_output,
                    out_channels=output_channel,
                    temb_channels=self.TIME_EMBED_DIM,
                    num_layers=self.LAYERS_UP_PER_BLOCK,
                    lora_rank=lora_rank, lora_scaling=lora_scaling,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final,
                ))
            else:
                self.up_blocks.append(NeuronS3DiffCrossAttnUpBlock2D(
                    in_channels=in_channels_this,
                    prev_output_channels=prev_output,
                    out_channels=output_channel,
                    temb_channels=self.TIME_EMBED_DIM,
                    num_layers=self.LAYERS_UP_PER_BLOCK,
                    num_attention_heads=reversed_heads[i],
                    attention_head_dim=self.ATTN_HEAD_DIM,
                    cross_attention_dim=self.CROSS_ATTN_DIM,
                    lora_rank=lora_rank, lora_scaling=lora_scaling,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final,
                ))

        # --- conv out (LoRA) ---
        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.BLOCK_OUT_CHANNELS[0],
            eps=1e-5, affine=True,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = DeModLoRAConv2d(
            self.BLOCK_OUT_CHANNELS[0], self.OUT_CHANNELS,
            kernel_size=3, padding=1, lora_rank=lora_rank,
            bias=True, scaling=lora_scaling,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        de_mod_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # timestep -> sinusoidal embedding -> MLP
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(sample.shape[0])
        t_emb = _sinusoidal_timestep_embedding(timesteps, self.BLOCK_OUT_CHANNELS[0])
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # conv_in (no LoRA)
        sample = self.conv_in(sample)

        # down_blocks
        down_block_res_samples = (sample,)
        for i, down in enumerate(self.down_blocks):
            path_prefix = f"down_blocks.{i}"
            if isinstance(down, NeuronS3DiffCrossAttnDownBlock2D):
                sample, res_samples = down(
                    sample, emb, encoder_hidden_states,
                    de_mod_map=de_mod_map, path_prefix=path_prefix,
                )
            else:
                sample, res_samples = down(
                    sample, emb, de_mod_map=de_mod_map, path_prefix=path_prefix,
                )
            down_block_res_samples = down_block_res_samples + res_samples

        # mid_block
        sample = self.mid_block(
            sample, emb, encoder_hidden_states,
            de_mod_map=de_mod_map, path_prefix="mid_block",
        )

        # up_blocks
        for i, up in enumerate(self.up_blocks):
            path_prefix = f"up_blocks.{i}"
            res_samples = down_block_res_samples[-len(up.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(up.resnets)]
            if isinstance(up, NeuronS3DiffCrossAttnUpBlock2D):
                sample = up(
                    sample, res_samples, emb, encoder_hidden_states,
                    upsample_size=None,
                    de_mod_map=de_mod_map, path_prefix=path_prefix,
                )
            else:
                sample = up(
                    sample, res_samples, emb,
                    upsample_size=None,
                    de_mod_map=de_mod_map, path_prefix=path_prefix,
                )

        # conv_norm_out -> silu -> conv_out
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, de_mod_map["conv_out"])
        return sample
