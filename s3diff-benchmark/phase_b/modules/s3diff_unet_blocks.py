"""NeuronS3Diff UNet block wrappers — matching diffusers block interfaces.

Five block types that compose the S3Diff UNet:
  - NeuronS3DiffDownBlock2D           (resnets only)
  - NeuronS3DiffCrossAttnDownBlock2D  (resnets + Transformer2D + downsample)
  - NeuronS3DiffUNetMidBlock2DCrossAttn (2 resnets + 1 Transformer2D)
  - NeuronS3DiffUpBlock2D             (resnets + upsample)
  - NeuronS3DiffCrossAttnUpBlock2D    (resnets + Transformer2D + upsample)

Each forward takes hidden_states, temb, encoder_hidden_states, and threaded
de_mod buffers (typically a dict keyed by sub-module path).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from de_mod_lora import DeModLoRAConv2d
from s3diff_resnet import NeuronS3DiffResnetBlock2D
from s3diff_transformer_2d import NeuronS3DiffTransformer2DModel


class NeuronS3DiffDownsample2D(nn.Module):
    """Diffusers Downsample2D replacement — just a strided DeModLoRAConv2d."""
    def __init__(self, channels, lora_rank=32, lora_scaling=0.25):
        super().__init__()
        # diffusers uses kernel=3, stride=2, padding=1
        self.conv = DeModLoRAConv2d(
            channels, channels, kernel_size=3, stride=2, padding=1,
            lora_rank=lora_rank, bias=True, scaling=lora_scaling,
        )

    def forward(self, x, de_mod_conv):
        return self.conv(x, de_mod_conv)


class NeuronS3DiffUpsample2D(nn.Module):
    """Diffusers Upsample2D replacement — nearest 2x + DeModLoRAConv2d."""
    def __init__(self, channels, lora_rank=32, lora_scaling=0.25):
        super().__init__()
        self.conv = DeModLoRAConv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            lora_rank=lora_rank, bias=True, scaling=lora_scaling,
        )

    def forward(self, x, de_mod_conv):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x, de_mod_conv)


class NeuronS3DiffDownBlock2D(nn.Module):
    """Plain resnet down block (no attention). S3Diff uses at down_blocks[3]."""

    def __init__(
        self, in_channels, out_channels, temb_channels, num_layers=2,
        lora_rank=32, lora_scaling=0.25, resnet_groups=32, add_downsample=False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(NeuronS3DiffResnetBlock2D(
                in_channels=ic, out_channels=out_channels, temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            ))
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                NeuronS3DiffDownsample2D(out_channels, lora_rank, lora_scaling)
            ])

    def forward(self, hidden_states, temb, de_mod_map, path_prefix):
        """Forward with de_mod_map: dict[str -> Tensor(B, r, r)] keyed by full module path.

        Expected keys under path_prefix:
          f"{prefix}.resnets.{i}.conv1", conv2, conv_shortcut
          f"{prefix}.downsamplers.0.conv" (if downsample)
        """
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            rp = f"{path_prefix}.resnets.{i}"
            hidden_states = resnet(
                hidden_states, temb,
                de_mod_conv1=de_mod_map[f"{rp}.conv1"],
                de_mod_conv2=de_mod_map[f"{rp}.conv2"],
                de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](
                hidden_states,
                de_mod_map[f"{path_prefix}.downsamplers.0.conv"],
            )
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


def _collect_transformer_demod(de_mod_map, attn_prefix):
    """Build the 12 de_mod tensors a NeuronS3DiffTransformer2DModel expects."""
    k = lambda suffix: f"{attn_prefix}.{suffix}"
    return dict(
        de_mod_proj_in=de_mod_map[k("proj_in")],
        de_mod_proj_out=de_mod_map[k("proj_out")],
        de_mod_attn1_q=de_mod_map[k("transformer_blocks.0.attn1.to_q")],
        de_mod_attn1_k=de_mod_map[k("transformer_blocks.0.attn1.to_k")],
        de_mod_attn1_v=de_mod_map[k("transformer_blocks.0.attn1.to_v")],
        de_mod_attn1_o=de_mod_map[k("transformer_blocks.0.attn1.to_out.0")],
        de_mod_attn2_q=de_mod_map[k("transformer_blocks.0.attn2.to_q")],
        de_mod_attn2_k=de_mod_map[k("transformer_blocks.0.attn2.to_k")],
        de_mod_attn2_v=de_mod_map[k("transformer_blocks.0.attn2.to_v")],
        de_mod_attn2_o=de_mod_map[k("transformer_blocks.0.attn2.to_out.0")],
        de_mod_ff_0=de_mod_map[k("transformer_blocks.0.ff.net.0.proj")],
        de_mod_ff_2=de_mod_map[k("transformer_blocks.0.ff.net.2")],
    )


class NeuronS3DiffCrossAttnDownBlock2D(nn.Module):
    """Resnet + Transformer2D + downsample. S3Diff uses at down_blocks[0,1,2]."""

    def __init__(
        self, in_channels, out_channels, temb_channels, num_layers=2,
        num_attention_heads=5, attention_head_dim=64, cross_attention_dim=1024,
        lora_rank=32, lora_scaling=0.25, resnet_groups=32, add_downsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(NeuronS3DiffResnetBlock2D(
                in_channels=ic, out_channels=out_channels, temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            ))
            self.attentions.append(NeuronS3DiffTransformer2DModel(
                in_channels=out_channels,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                lora_rank=lora_rank, lora_scaling=lora_scaling,
                num_groups=resnet_groups,
            ))
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                NeuronS3DiffDownsample2D(out_channels, lora_rank, lora_scaling)
            ])

    def forward(self, hidden_states, temb, encoder_hidden_states,
                de_mod_map, path_prefix):
        output_states = ()
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            rp = f"{path_prefix}.resnets.{i}"
            ap = f"{path_prefix}.attentions.{i}"
            hidden_states = resnet(
                hidden_states, temb,
                de_mod_conv1=de_mod_map[f"{rp}.conv1"],
                de_mod_conv2=de_mod_map[f"{rp}.conv2"],
                de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
            )
            hidden_states = attn(
                hidden_states, encoder_hidden_states,
                **_collect_transformer_demod(de_mod_map, ap),
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](
                hidden_states,
                de_mod_map[f"{path_prefix}.downsamplers.0.conv"],
            )
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class NeuronS3DiffUNetMidBlock2DCrossAttn(nn.Module):
    """2 resnets + 1 transformer2d sandwiched. Diffusers pattern: resnet -> attn -> resnet."""

    def __init__(
        self, in_channels, temb_channels, num_layers=1,
        num_attention_heads=20, attention_head_dim=64, cross_attention_dim=1024,
        lora_rank=32, lora_scaling=0.25, resnet_groups=32,
    ):
        super().__init__()
        # Diffusers UNetMidBlock2DCrossAttn always has 1 extra resnet before attentions
        self.resnets = nn.ModuleList([
            NeuronS3DiffResnetBlock2D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            )
        ])
        self.attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(NeuronS3DiffTransformer2DModel(
                in_channels=in_channels,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                lora_rank=lora_rank, lora_scaling=lora_scaling,
                num_groups=resnet_groups,
            ))
            self.resnets.append(NeuronS3DiffResnetBlock2D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            ))

    def forward(self, hidden_states, temb, encoder_hidden_states,
                de_mod_map, path_prefix):
        # First resnet
        rp = f"{path_prefix}.resnets.0"
        hidden_states = self.resnets[0](
            hidden_states, temb,
            de_mod_conv1=de_mod_map[f"{rp}.conv1"],
            de_mod_conv2=de_mod_map[f"{rp}.conv2"],
            de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
        )
        # Alternating attn -> resnet
        for i, attn in enumerate(self.attentions):
            ap = f"{path_prefix}.attentions.{i}"
            hidden_states = attn(
                hidden_states, encoder_hidden_states,
                **_collect_transformer_demod(de_mod_map, ap),
            )
            rp = f"{path_prefix}.resnets.{i + 1}"
            hidden_states = self.resnets[i + 1](
                hidden_states, temb,
                de_mod_conv1=de_mod_map[f"{rp}.conv1"],
                de_mod_conv2=de_mod_map[f"{rp}.conv2"],
                de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
            )
        return hidden_states


class NeuronS3DiffUpBlock2D(nn.Module):
    """Plain resnet up block (no attention). S3Diff uses at up_blocks[0]."""

    def __init__(
        self, in_channels, prev_output_channels, out_channels, temb_channels,
        num_layers=3, lora_rank=32, lora_scaling=0.25, resnet_groups=32,
        add_upsample=True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            # diffusers: resnet[0] takes skip+prev_up, resnet[i>0] takes skip+out
            res_skip_ch = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_ch = prev_output_channels if i == 0 else out_channels
            self.resnets.append(NeuronS3DiffResnetBlock2D(
                in_channels=resnet_in_ch + res_skip_ch,
                out_channels=out_channels,
                temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            ))
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                NeuronS3DiffUpsample2D(out_channels, lora_rank, lora_scaling)
            ])

    def forward(self, hidden_states, res_hidden_states_tuple, temb,
                upsample_size, de_mod_map, path_prefix):
        for i, resnet in enumerate(self.resnets):
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            rp = f"{path_prefix}.resnets.{i}"
            hidden_states = resnet(
                hidden_states, temb,
                de_mod_conv1=de_mod_map[f"{rp}.conv1"],
                de_mod_conv2=de_mod_map[f"{rp}.conv2"],
                de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
            )

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](
                hidden_states,
                de_mod_map[f"{path_prefix}.upsamplers.0.conv"],
            )
        return hidden_states


class NeuronS3DiffCrossAttnUpBlock2D(nn.Module):
    """Resnet + transformer2d + (optional) upsample. Used at up_blocks[1,2,3]."""

    def __init__(
        self, in_channels, prev_output_channels, out_channels, temb_channels,
        num_layers=3, num_attention_heads=10, attention_head_dim=64,
        cross_attention_dim=1024, lora_rank=32, lora_scaling=0.25,
        resnet_groups=32, add_upsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            res_skip_ch = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_ch = prev_output_channels if i == 0 else out_channels
            self.resnets.append(NeuronS3DiffResnetBlock2D(
                in_channels=resnet_in_ch + res_skip_ch,
                out_channels=out_channels,
                temb_channels=temb_channels,
                lora_rank=lora_rank, lora_scaling=lora_scaling, groups=resnet_groups,
            ))
            self.attentions.append(NeuronS3DiffTransformer2DModel(
                in_channels=out_channels,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                lora_rank=lora_rank, lora_scaling=lora_scaling,
                num_groups=resnet_groups,
            ))
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                NeuronS3DiffUpsample2D(out_channels, lora_rank, lora_scaling)
            ])

    def forward(self, hidden_states, res_hidden_states_tuple, temb,
                encoder_hidden_states, upsample_size, de_mod_map, path_prefix):
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            rp = f"{path_prefix}.resnets.{i}"
            ap = f"{path_prefix}.attentions.{i}"
            hidden_states = resnet(
                hidden_states, temb,
                de_mod_conv1=de_mod_map[f"{rp}.conv1"],
                de_mod_conv2=de_mod_map[f"{rp}.conv2"],
                de_mod_conv_shortcut=de_mod_map.get(f"{rp}.conv_shortcut"),
            )
            hidden_states = attn(
                hidden_states, encoder_hidden_states,
                **_collect_transformer_demod(de_mod_map, ap),
            )

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](
                hidden_states,
                de_mod_map[f"{path_prefix}.upsamplers.0.conv"],
            )
        return hidden_states
