"""Inventory of the full UNet structure (class names, per-block counts)."""
import sys
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")

import argparse as _ap
s3args = _ap.Namespace(
    lora_rank_unet=32, lora_rank_vae=16, latent_tiled_size=96, latent_tiled_overlap=32,
    vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224, padding_offset=32,
    pos_prompt="x", neg_prompt="y",
    sd_path="/home/ubuntu/s3diff/models/sd-turbo",
    pretrained_path="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl",
)
from s3diff_tile import S3Diff
net_sr = S3Diff(sd_path=s3args.sd_path, pretrained_path=s3args.pretrained_path,
                lora_rank_unet=32, lora_rank_vae=16, args=s3args)
net_sr.set_eval()

u = net_sr.unet
print("conv_in:", type(u.conv_in).__name__)
if hasattr(u.conv_in, "base_layer"):
    bl = u.conv_in.base_layer
    print("  base:", bl.in_channels, "->", bl.out_channels, "k=", bl.kernel_size, "p=", bl.padding)
print("time_proj:", type(u.time_proj).__name__)
print("time_embedding:", type(u.time_embedding).__name__)

for i, b in enumerate(u.down_blocks):
    has_attn = hasattr(b, "attentions")
    has_ds = getattr(b, "downsamplers", None) is not None
    print(f"down_blocks.{i}: {type(b).__name__} resnets={len(b.resnets)} "
          f"attentions={len(b.attentions) if has_attn else 0} "
          f"downsamplers={len(b.downsamplers) if has_ds else 0}")
    if has_ds:
        ds = b.downsamplers[0]
        print(f"  downsamplers.0.conv: {type(ds.conv).__name__}")
    for j, rn in enumerate(b.resnets):
        print(f"  resnets.{j}: in={rn.in_channels} out={rn.out_channels} shortcut={rn.conv_shortcut is not None}")

print("mid_block:", type(u.mid_block).__name__, "resnets=", len(u.mid_block.resnets),
      "attns=", len(u.mid_block.attentions))
for j, rn in enumerate(u.mid_block.resnets):
    print(f"  mid resnets.{j}: in={rn.in_channels} out={rn.out_channels}")
for j, at in enumerate(u.mid_block.attentions):
    print(f"  mid attns.{j}: type={type(at).__name__} in_channels={at.in_channels}")

for i, b in enumerate(u.up_blocks):
    has_attn = hasattr(b, "attentions")
    has_us = getattr(b, "upsamplers", None) is not None
    print(f"up_blocks.{i}: {type(b).__name__} resnets={len(b.resnets)} "
          f"attentions={len(b.attentions) if has_attn else 0} "
          f"upsamplers={len(b.upsamplers) if has_us else 0}")
    if has_us:
        us = b.upsamplers[0]
        print(f"  upsamplers.0.conv: {type(us.conv).__name__}")
    for j, rn in enumerate(b.resnets):
        print(f"  resnets.{j}: in={rn.in_channels} out={rn.out_channels} shortcut={rn.conv_shortcut is not None}")

print("conv_norm_out:", u.conv_norm_out)
print("conv_out:", type(u.conv_out).__name__)
if hasattr(u.conv_out, "base_layer"):
    bl = u.conv_out.base_layer
    print("  base:", bl.in_channels, "->", bl.out_channels, "k=", bl.kernel_size)
