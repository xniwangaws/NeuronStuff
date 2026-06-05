"""TP=4 SDXL VAE decoder via in-place module replacement.

Same approach as UNet TP=4 (sdxl-benchmark/unet_tp4/neuron_unet.py):
- Load diffusers AutoencoderKL on CPU bf16.
- Walk vae.decoder + vae.post_quant_conv module tree.
- Replace nn.Conv2d with NxD OutputChannelParallelConv2d(gather_output=True).
- Replace nn.Linear (only present in mid_block.attentions[0]) with
  ColumnParallelLinear(gather_output=True).
- gather_output=True keeps the activation full at the layer boundary so the
  rest of the diffusers VAE forward (GroupNorm, SiLU, attention) sees the
  expected shape. Per-rank weight + per-rank instruction count both /TP.

Why this should fit at 2K:
- The 38.14 GB graph activation memory (kaena-30652 with Luo's
  --tiled-inst-limit flag) is dominated by intermediate tensors after
  upsampling at 1024x1024 / 2048x2048 spatial. Sharding the conv outputs
  across TP=4 splits those activation tiles by 4 -> ~9.5 GB per rank,
  comfortably below the 24 GB / logical core HBM budget.

Channel sizing (all friendly to TP=4):
- post_quant_conv: 4 -> 4   (NOT sharded; numel < threshold and 4 not /4 cleanly anyway? 4/4=1 ok but tiny)
- conv_in:         4 -> 512
- mid_block:       512 (resnets + 1 attention)
- up_blocks.0:     512 -> 512  (3 resnets, 1 upsampler)
- up_blocks.1:     512 -> 512  (3 resnets, 1 upsampler)
- up_blocks.2:     512 -> 256  (3 resnets, 1 upsampler)
- up_blocks.3:     256 -> 128  (3 resnets, no upsampler)
- conv_norm_out -> conv_out: 128 -> 3   (NOT sharded; out_channels=3 not div by 4)

GroupNorm: VAE uses num_groups=32. After sharding 512->128 channels per rank,
128 % 32 = 0 -> ok. After 256->64: 64 % 32 = 0 ok. After 128->32: 32 % 32 = 0 ok.
So per-rank GroupNorm runs fine on the sharded activation IF gather_output=False
were used. Since we use gather_output=True, GN sees full channels and behaves
identically to the unsharded model.

Attention in mid_block: heads=1, head_dim=512 (single-head spatial self-attention).
to_q/to_k/to_v are nn.Linear(512, 512). 512 % 4 = 0 -> shardable. With
gather_output=True the output is full (1, HW, 512) so the rest of the attention
math (reshape to heads etc.) is unchanged.
"""
import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    OutputChannelParallelConv2d,
)


# Smaller threshold than UNet (UNet had ~860M params, VAE decoder is ~50M).
# Set low enough that conv_in (4->512, kernel 3x3 = 4*512*9 = 18432 weights)
# is still below threshold while the bigger convs (512->512 3x3 = 2.36M) are sharded.
SHARD_NUMEL_THRESHOLD = 100_000


def _shardable_dim(out_dim: int, tp: int) -> bool:
    return out_dim % tp == 0


def _replace_conv2d(parent: nn.Module, name: str, conv: nn.Conv2d, tp: int, dtype: torch.dtype, stats: dict, full_name: str):
    if conv.weight.numel() < SHARD_NUMEL_THRESHOLD:
        stats["conv_skipped_small"] += 1
        return
    if conv.groups != 1:
        stats["conv_skipped_grouped"] += 1
        return
    if not _shardable_dim(conv.out_channels, tp):
        stats["conv_skipped_dim"] += 1
        return
    new = OutputChannelParallelConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
        gather_output=True,
        dtype=dtype,
    )
    setattr(parent, name, new)
    stats["conv_replaced"] += 1
    stats["replaced_keys"].append(("C", full_name, conv.in_channels, conv.out_channels))


def _replace_linear(parent: nn.Module, name: str, lin: nn.Linear, tp: int, dtype: torch.dtype, stats: dict, full_name: str):
    if lin.weight.numel() < SHARD_NUMEL_THRESHOLD:
        stats["linear_skipped_small"] += 1
        return
    if not _shardable_dim(lin.out_features, tp):
        stats["linear_skipped_dim"] += 1
        return
    new = ColumnParallelLinear(
        input_size=lin.in_features,
        output_size=lin.out_features,
        bias=(lin.bias is not None),
        gather_output=True,
        dtype=dtype,
    )
    setattr(parent, name, new)
    stats["linear_replaced"] += 1
    stats["replaced_keys"].append(("L", full_name, lin.in_features, lin.out_features))


def shard_vae_decoder_in_place(
    vae: nn.Module, tp: int = 4, dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Walk the VAE (post_quant_conv + decoder subtree) and replace eligible
    Conv2d/Linear with NxD parallel layers. Returns stats.
    """
    stats = {
        "linear_replaced": 0,
        "linear_skipped_small": 0,
        "linear_skipped_dim": 0,
        "conv_replaced": 0,
        "conv_skipped_small": 0,
        "conv_skipped_grouped": 0,
        "conv_skipped_dim": 0,
        "replaced_keys": [],
    }

    # Snapshot (parent, attr_name, child) up front to avoid named_modules
    # confusion during in-place mutation.
    targets = []
    for parent_name, parent in vae.named_modules():
        # Only touch decoder subtree + post_quant_conv (encoder is unused).
        if not (
            parent_name == ""
            or parent_name.startswith("decoder")
            or parent_name == "post_quant_conv"
        ):
            continue
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            # Skip encoder children at root level.
            if full.startswith("encoder") or full.startswith("quant_conv"):
                continue
            targets.append((parent_name, parent, child_name, child, full))

    for parent_name, parent, child_name, child, full in targets:
        if isinstance(child, nn.Conv2d):
            _replace_conv2d(parent, child_name, child, tp, dtype, stats, full)
        elif isinstance(child, nn.Linear):
            _replace_linear(parent, child_name, child, tp, dtype, stats, full)
    return stats


class VAEDecoderTraceWrapper(nn.Module):
    """Wraps the diffusers VAE so that forward(latent) runs post_quant_conv
    then the decoder. parallel_model_trace / ModelBuilder require a
    plain (Tensor) -> Tensor contract.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent):
        z = self.post_quant_conv(latent)
        return self.decoder(z)


def build_sharded_vae_decoder(
    vae_path: str, tp: int = 4, dtype: torch.dtype = torch.bfloat16
):
    """Load SDXL VAE on CPU and replace eligible layers with NxD parallel
    layers. Returns (wrapped_module, stats).
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae", variant="fp16", torch_dtype=dtype
    )
    vae.eval()
    stats = shard_vae_decoder_in_place(vae, tp=tp, dtype=dtype)
    wrapped = VAEDecoderTraceWrapper(vae)
    wrapped.eval()
    return wrapped, stats
