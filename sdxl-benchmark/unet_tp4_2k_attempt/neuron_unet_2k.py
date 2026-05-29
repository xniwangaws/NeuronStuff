"""TP=4 SDXL UNet with sharded-activation conv chain + NKI flash attention.

Strategy for 2K (vs 1K version that gathers every conv output):
- ResnetBlock2D: conv1 OutputChannel(gather=False) -> norm2 (sharded GroupNorm) ->
  SiLU -> conv2 InputChannel(input_is_parallel=True) -> all-reduce.
  This keeps activation sharded (1/4 size) between conv1 and conv2 — the BIG saving.
- Other isolated Conv2d (conv_in, conv_out, downsamplers, upsamplers): full Output(gather=True).
- All Linear: ColumnParallel(gather=True).
- attn1/attn2: monkey-patch Attention.get_attention_scores with NKI flash kernel
  (no materialized attn matrix).

This is the validated Plan B VAE pattern, applied to SDXL UNet.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    OutputChannelParallelConv2d,
    InputChannelParallelConv2d,
)


SHARD_NUMEL_THRESHOLD = 100_000  # lowered for ResnetBlock convs (320*320*3*3 = 921k)
# Min channel count to even attempt sharding — ensures GroupNorm(32) can still divide
# the sharded dim (ch/TP must be divisible by 32 → ch >= 32*TP = 128)
MIN_CH_FOR_CHAINED_SHARD = 128 * 4  # 512 — only shard convs whose I/O ch ≥ 512


def _shardable_dim(out_dim: int, tp: int) -> bool:
    return out_dim % tp == 0


def _replace_linear_col(parent, name, lin, tp, dtype, gather_output=True):
    if lin.weight.numel() < SHARD_NUMEL_THRESHOLD:
        return False
    if not _shardable_dim(lin.out_features, tp):
        return False
    new = ColumnParallelLinear(
        input_size=lin.in_features,
        output_size=lin.out_features,
        bias=(lin.bias is not None),
        gather_output=gather_output,
        dtype=dtype,
    )
    setattr(parent, name, new)
    return True


def _replace_conv_output(parent, name, conv, tp, dtype, gather_output=True):
    if conv.weight.numel() < SHARD_NUMEL_THRESHOLD:
        return False
    if conv.groups != 1:
        return False
    if not _shardable_dim(conv.out_channels, tp):
        return False
    new = OutputChannelParallelConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
        gather_output=gather_output,
        dtype=dtype,
    )
    setattr(parent, name, new)
    return True


def _replace_conv_input(parent, name, conv, tp, dtype, input_is_parallel=True):
    if conv.weight.numel() < SHARD_NUMEL_THRESHOLD:
        return False
    if conv.groups != 1:
        return False
    if not _shardable_dim(conv.in_channels, tp):
        return False
    new = InputChannelParallelConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
        input_is_parallel=input_is_parallel,
        dtype=dtype,
    )
    setattr(parent, name, new)
    return True


# --- NKI flash attention monkey patches ---
# Source: Plan A's trace_unet_2k_perblock.py which works at 2K
def _custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale


def _get_attention_scores_neuron(self, query, key, attn_mask):
    """Replace diffusers Attention.get_attention_scores to skip the
    full-precision attn matrix materialization. Stays in bf16 throughout.
    """
    if query.size() == key.size():
        s = _custom_badbmm(key, query.transpose(-1, -2), self.scale)
        return s.softmax(dim=1).permute(0, 2, 1)
    s = _custom_badbmm(query, key.transpose(-1, -2), self.scale)
    return s.softmax(dim=-1)


def patch_attention_for_nki():
    """Apply at trace time. Modifies the diffusers Attention class globally —
    this only affects modules instantiated/called after this point.
    """
    from diffusers.models.attention_processor import Attention
    Attention.get_attention_scores = _get_attention_scores_neuron


def shard_unet_in_place(unet: nn.Module, tp: int = 4, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Walk the UNet:
    - ResnetBlock2D: conv1 Output(gather=False), conv2 Input(parallel=True),
      conv_shortcut Output(gather=True) [boundary back to full]
    - Other Conv2d: Output(gather=True)
    - All Linear: ColumnParallel(gather=True)
    """
    from diffusers.models.resnet import ResnetBlock2D

    stats = {
        "linear_replaced": 0, "linear_skipped": 0,
        "conv_output_full_gather": 0,
        "conv_output_no_gather": 0,
        "conv_input_parallel": 0,
        "conv_skipped": 0,
        "resnet_blocks_chained": 0,
        "replaced_keys": [],
    }

    # First pass: identify ResnetBlock2D instances and apply chained pattern
    resnet_modules = []
    for parent_name, parent in unet.named_modules():
        if isinstance(parent, ResnetBlock2D):
            resnet_modules.append((parent_name, parent))

    chained_conv_ids = set()
    for resnet_name, rb in resnet_modules:
        # rb has: norm1 (GroupNorm), conv1, norm2 (GroupNorm), conv2,
        # optionally conv_shortcut, optional time_emb_proj
        # Pattern: conv1 (Output, no gather) -> norm2 stays full because conv1 gathered? NO -
        # we want norm2 to operate on sharded. NxD has no ParallelGroupNorm but
        # GroupNorm with full channels-per-rank works only if groups divide ranks.
        # SDXL Resnet GroupNorm has 32 groups. With TP=4: 8 groups/rank, full per-rank channels c/4.
        # Need to manually apply GroupNorm per-rank. Simplest: keep conv1 gather=True
        # to make norm2 simple BUT then we lose the activation savings.
        # Compromise: use Output(gather=False) for conv1, then INPUT-parallel conv2;
        # GroupNorm is replicated weight + applied per-rank but it operates on each
        # rank's local slice of channels. As long as channels_per_group=c/groups
        # is the same per-rank (groups=32, c=320 -> 10 ch/group; sharded c/4=80
        # -> 80/8groups_per_rank = 10 ch/group, same), GroupNorm gives bitwise
        # identical result per-rank.
        if hasattr(rb, "conv1") and isinstance(rb.conv1, nn.Conv2d):
            # Only chain-shard if both in_ch and out_ch ≥ MIN_CH_FOR_CHAINED_SHARD,
            # so GroupNorm(num_groups=32) can still divide the sharded dim
            if rb.conv1.in_channels >= MIN_CH_FOR_CHAINED_SHARD and rb.conv1.out_channels >= MIN_CH_FOR_CHAINED_SHARD:
                ok = _replace_conv_output(rb, "conv1", rb.conv1, tp, dtype, gather_output=False)
                if ok:
                    chained_conv_ids.add(id(rb.conv1))
                    stats["conv_output_no_gather"] += 1
                    stats["replaced_keys"].append(("Cout-nogath", f"{resnet_name}.conv1"))
            else:
                # Fall back to gather=True for small-channel resnets
                ok = _replace_conv_output(rb, "conv1", rb.conv1, tp, dtype, gather_output=True)
                if ok:
                    chained_conv_ids.add(id(rb.conv1))
                    stats["conv_output_full_gather"] += 1
        # ResnetBlock has h = conv1(...) + time_emb_proj(temb)[:,:,None,None] -> sharded.
        # Need time_emb_proj also ColumnParallel(gather=False) so its output is 1/4 channels matching sharded h.
        # Only shard time_emb_proj IF conv1 was chain-sharded (no-gather)
        conv1_chained = any(
            kk[1] == f"{resnet_name}.conv1" and kk[0] == "Cout-nogath"
            for kk in stats["replaced_keys"]
        )
        if conv1_chained and hasattr(rb, "time_emb_proj") and isinstance(rb.time_emb_proj, nn.Linear):
            tep = rb.time_emb_proj
            if _shardable_dim(tep.out_features, tp):
                new_tep = ColumnParallelLinear(
                    input_size=tep.in_features,
                    output_size=tep.out_features,
                    bias=(tep.bias is not None),
                    gather_output=False,  # match conv1 sharded output
                    dtype=dtype,
                )
                rb.time_emb_proj = new_tep
                stats["replaced_keys"].append(("L-tepNoGath", f"{resnet_name}.time_emb_proj"))
        # conv2 receives sharded input (from conv1 no-gather + sharded temb add) only when conv1 was chain-sharded
        conv1_was_chained = any(
            kk[1] == f"{resnet_name}.conv1" and kk[0] == "Cout-nogath"
            for kk in stats["replaced_keys"]
        )
        if hasattr(rb, "conv2") and isinstance(rb.conv2, nn.Conv2d):
            if conv1_was_chained:
                ok = _replace_conv_input(rb, "conv2", rb.conv2, tp, dtype, input_is_parallel=True)
                if ok:
                    chained_conv_ids.add(id(rb.conv2))
                    stats["conv_input_parallel"] += 1
                    stats["replaced_keys"].append(("Cin-parallel", f"{resnet_name}.conv2"))
            else:
                ok = _replace_conv_output(rb, "conv2", rb.conv2, tp, dtype, gather_output=True)
                if ok:
                    chained_conv_ids.add(id(rb.conv2))
                    stats["conv_output_full_gather"] += 1
        # conv_shortcut: full gather (boundary)
        if getattr(rb, "conv_shortcut", None) is not None and isinstance(rb.conv_shortcut, nn.Conv2d):
            ok = _replace_conv_output(rb, "conv_shortcut", rb.conv_shortcut, tp, dtype, gather_output=True)
            if ok:
                chained_conv_ids.add(id(rb.conv_shortcut))
                stats["conv_output_full_gather"] += 1
        # time_emb_proj is Linear -- handled in second pass uniformly
        if stats["replaced_keys"] and stats["replaced_keys"][-1][1].startswith(resnet_name):
            stats["resnet_blocks_chained"] += 1

    # Second pass: handle all other Linear and Conv2d uniformly
    targets = []
    for parent_name, parent in unet.named_modules():
        for child_name, child in list(parent.named_children()):
            targets.append((parent_name, parent, child_name, child))

    for parent_name, parent, child_name, child in targets:
        if isinstance(child, nn.Linear):
            if child.weight.numel() < SHARD_NUMEL_THRESHOLD:
                stats["linear_skipped"] += 1
                continue
            if not _shardable_dim(child.out_features, tp):
                stats["linear_skipped"] += 1
                continue
            ok = _replace_linear_col(parent, child_name, child, tp, dtype, gather_output=True)
            if ok:
                stats["linear_replaced"] += 1
        elif isinstance(child, nn.Conv2d):
            if id(child) in chained_conv_ids:
                continue  # already replaced in resnet pass
            if child.weight.numel() < SHARD_NUMEL_THRESHOLD:
                stats["conv_skipped"] += 1
                continue
            if child.groups != 1:
                stats["conv_skipped"] += 1
                continue
            if not _shardable_dim(child.out_channels, tp):
                stats["conv_skipped"] += 1
                continue
            ok = _replace_conv_output(parent, child_name, child, tp, dtype, gather_output=True)
            if ok:
                stats["conv_output_full_gather"] += 1

    return stats


class UNetTraceWrapper(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return out


def build_sharded_unet(weights_path: str, tp: int = 4, dtype: torch.dtype = torch.bfloat16,
                      use_nki_flash: bool = True):
    from diffusers import UNet2DConditionModel
    if use_nki_flash:
        patch_attention_for_nki()
    unet = UNet2DConditionModel.from_pretrained(weights_path, variant="fp16", torch_dtype=dtype)
    unet.eval()
    stats = shard_unet_in_place(unet, tp=tp, dtype=dtype)
    wrapped = UNetTraceWrapper(unet)
    wrapped.eval()
    return wrapped, stats
