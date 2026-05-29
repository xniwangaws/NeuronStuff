"""TP=4 SDXL UNet via in-place module replacement.

Strategy:
- Load diffusers UNet2DConditionModel on CPU bf16.
- Walk the module tree and replace nn.Linear / nn.Conv2d with NxD parallel layers,
  using gather_output=True so each replaced layer's output shape matches the
  original (full activation). This keeps the rest of the UNet unchanged and gives
  per-rank weight + per-rank instruction count both /TP.
- For correctness with diffusers' built-in modules (which call .forward as-is),
  the parallel layers must accept the same input dtype/shape and return the same
  shape. NxD ColumnParallelLinear with gather_output=True does exactly that.
- For Conv2d in residual blocks: OutputChannelParallelConv2d(gather_output=True).
- Skip very small layers (numel < SHARD_THRESHOLD) — replicate them. Also skip
  layers whose channel count is not divisible by TP (e.g. conv_out: 4 channels).
- Time embedding linears (small, 320 -> 1280) we shard or keep — sharding
  reduces compute but adds all-gather overhead; keep replicated since they're
  tiny and only run once.

The diffusers UNet forward path is unchanged. We trace the whole UNet via
parallel_model_trace which spawns one rank-aware copy per TP rank.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    OutputChannelParallelConv2d,
)


# Threshold: layers smaller than this stay replicated (the all-gather overhead
# would exceed compute savings).
SHARD_NUMEL_THRESHOLD = 1_000_000  # ~1M params


def _shardable_dim(out_dim: int, tp: int) -> bool:
    return out_dim % tp == 0


def replace_linear_with_column_parallel(
    parent: nn.Module, name: str, lin: nn.Linear, tp: int, dtype: torch.dtype
) -> bool:
    """Replace a nn.Linear with a ColumnParallelLinear (gather_output=True).
    Returns True if replaced. Skips if out_features not divisible by tp or numel
    too small.
    """
    if lin.weight.numel() < SHARD_NUMEL_THRESHOLD:
        return False
    out_f = lin.out_features
    in_f = lin.in_features
    if not _shardable_dim(out_f, tp):
        return False
    new = ColumnParallelLinear(
        input_size=in_f,
        output_size=out_f,
        bias=(lin.bias is not None),
        gather_output=True,
        dtype=dtype,
    )
    setattr(parent, name, new)
    return True


def replace_conv2d_with_output_channel_parallel(
    parent: nn.Module, name: str, conv: nn.Conv2d, tp: int, dtype: torch.dtype
) -> bool:
    """Replace a nn.Conv2d with OutputChannelParallelConv2d (gather_output=True)."""
    if conv.weight.numel() < SHARD_NUMEL_THRESHOLD:
        return False
    out_c = conv.out_channels
    in_c = conv.in_channels
    if not _shardable_dim(out_c, tp):
        return False
    if conv.groups != 1:
        # Grouped convs not supported by OutputChannelParallelConv2d natively
        return False
    k = conv.kernel_size
    pad = conv.padding
    stride = conv.stride
    new = OutputChannelParallelConv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=k,
        stride=stride,
        padding=pad,
        bias=(conv.bias is not None),
        gather_output=True,
        dtype=dtype,
    )
    setattr(parent, name, new)
    return True


def shard_unet_in_place(unet: nn.Module, tp: int = 4, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Walk the UNet tree and replace eligible Linear/Conv2d with NxD parallel
    versions. Returns a stats dict for logging.
    """
    stats = {
        "linear_replaced": 0,
        "linear_skipped_small": 0,
        "linear_skipped_dim": 0,
        "conv_replaced": 0,
        "conv_skipped_small": 0,
        "conv_skipped_dim": 0,
        "conv_skipped_grouped": 0,
        "replaced_keys": [],
    }

    # Collect ALL (parent, attr_name, child) triples first; modifying during walk
    # confuses named_modules.
    targets = []
    for parent_name, parent in unet.named_modules():
        for child_name, child in list(parent.named_children()):
            targets.append((parent_name, parent, child_name, child))

    for parent_name, parent, child_name, child in targets:
        full = f"{parent_name}.{child_name}" if parent_name else child_name
        if isinstance(child, nn.Linear):
            if child.weight.numel() < SHARD_NUMEL_THRESHOLD:
                stats["linear_skipped_small"] += 1
                continue
            if not _shardable_dim(child.out_features, tp):
                stats["linear_skipped_dim"] += 1
                continue
            ok = replace_linear_with_column_parallel(parent, child_name, child, tp, dtype)
            if ok:
                stats["linear_replaced"] += 1
                stats["replaced_keys"].append(("L", full, child.in_features, child.out_features))
        elif isinstance(child, nn.Conv2d):
            if child.weight.numel() < SHARD_NUMEL_THRESHOLD:
                stats["conv_skipped_small"] += 1
                continue
            if child.groups != 1:
                stats["conv_skipped_grouped"] += 1
                continue
            if not _shardable_dim(child.out_channels, tp):
                stats["conv_skipped_dim"] += 1
                continue
            ok = replace_conv2d_with_output_channel_parallel(parent, child_name, child, tp, dtype)
            if ok:
                stats["conv_replaced"] += 1
                stats["replaced_keys"].append(("C", full, child.in_channels, child.out_channels))
    return stats


class UNetTraceWrapper(nn.Module):
    """Wraps a diffusers UNet so it accepts positional tensor args and returns
    a single tensor (parallel_model_trace requires this contract).
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        # diffusers UNet wants timestep as long; sample/encoder_hidden_states bf16.
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return out


def build_sharded_unet(weights_path: str, tp: int = 4, dtype: torch.dtype = torch.bfloat16):
    """Build an SDXL UNet on CPU and replace eligible layers with NxD parallel
    layers. Returns (wrapped_module, stats).
    """
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(weights_path, variant="fp16", torch_dtype=dtype)
    unet.eval()
    stats = shard_unet_in_place(unet, tp=tp, dtype=dtype)
    wrapped = UNetTraceWrapper(unet)
    wrapped.eval()
    return wrapped, stats


def build_state_dict_for_sharded(orig_unet_state_dict: dict, sharded_unet: nn.Module, dtype) -> dict:
    """Map orig diffusers UNet state_dict keys onto the sharded module.

    The sharded module's parameters live at the same key paths as the original
    (we only swap module instances; named_parameter paths are identical for
    weight/bias). NxD's shard_checkpoint walks by module type so it'll slice
    Conv/Linear weights along dim 0 for all replaced layers automatically.
    The state dict produced here is the FULL master copy keyed identically.
    """
    out = {}
    target_keys = set(dict(sharded_unet.named_parameters()).keys()) | set(
        dict(sharded_unet.named_buffers()).keys()
    )
    # Prefix is "unet." since wrapped in UNetTraceWrapper
    for k, v in orig_unet_state_dict.items():
        new_k = f"unet.{k}"
        if new_k in target_keys:
            out[new_k] = v.detach().clone().to(v.dtype)
        else:
            # Some keys (norm/embedding) still match
            out[new_k] = v.detach().clone()
    return out
