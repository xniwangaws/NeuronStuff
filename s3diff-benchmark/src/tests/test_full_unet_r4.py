"""End-to-end UNet correctness test: peft LoRA (ref) vs DeModLoRA-attr replacements.

Loads S3Diff, deep-copies the UNet, replaces all peft LoRA modules with our
DeMod variants, sets identical de_mod tensors on both, runs a single forward,
compares outputs. Gate: cosine > 0.999 on CPU fp32.

If this passes, we know the full UNet forward reproduces correctly when the
LoRA implementation is swapped — that's R4 core correctness.
"""
from __future__ import annotations

import copy
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def build_s3diff():
    import argparse as _ap
    s3args = _ap.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32, pos_prompt="x", neg_prompt="y",
        sd_path="/home/ubuntu/s3diff/models/sd-turbo",
        pretrained_path="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl",
    )
    from s3diff_tile import S3Diff
    net_sr = S3Diff(sd_path=s3args.sd_path, pretrained_path=s3args.pretrained_path,
                    lora_rank_unet=32, lora_rank_vae=16, args=s3args)
    net_sr.set_eval()
    return net_sr


def set_de_mods_on_unet(unet, rank=32, seed=42):
    """Set a random de_mod on every peft LoraLayer and return a dict {name: tensor} for reference."""
    from peft.tuners.lora.layer import Linear as PeftLinear, Conv2d as PeftConv2d
    torch.manual_seed(seed)
    out = {}
    for name, m in unet.named_modules():
        if isinstance(m, (PeftLinear, PeftConv2d)):
            t = torch.randn(1, rank, rank)
            m.de_mod = t
            out[name] = t
    return out


def set_de_mods_on_demod_unet(unet, ref_de_mods):
    """Apply the same de_mod tensors to the replaced DeMod* modules."""
    from de_mod_lora_attr import DeModLoRALinearAttr, DeModLoRAConv2dAttr
    count = 0
    for name, m in unet.named_modules():
        if isinstance(m, (DeModLoRALinearAttr, DeModLoRAConv2dAttr)):
            t = ref_de_mods[name]
            m.de_mod = t
            count += 1
    return count


def main():
    print("[load] S3Diff CPU fp32...", flush=True)
    net_sr = build_s3diff()

    # Build 2 inputs that match a typical UNet call
    torch.manual_seed(0)
    # Latent: 1x4x16x16 for speed (normally 1x4x128x128 but testing correctness)
    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([999])
    enc = torch.randn(1, 77, 1024)

    # Assign de_mods to the live (peft) UNet — reference path
    ref_de_mods = set_de_mods_on_unet(net_sr.unet, rank=32, seed=42)
    print(f"[ref] set de_mod on {len(ref_de_mods)} peft modules", flush=True)

    print("[ref] forward on peft UNet...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_out = net_sr.unet(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc, return_dict=False)[0]
    print(f"[ref] shape={tuple(ref_out.shape)} mean={ref_out.mean():.4e} std={ref_out.std():.4e} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Now replace in the SAME unet instance (peft -> DeMod). Simpler than deepcopy.
    print("[replace] replacing peft LoRA with DeModLoRA-attr...", flush=True)
    from de_mod_lora_attr import replace_lora_modules_in_unet
    t0 = time.perf_counter()
    replaced = replace_lora_modules_in_unet(net_sr.unet, adapter_name="default")
    print(f"[replace] replaced {len(replaced)} modules ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Set de_mods on the replaced modules
    count = set_de_mods_on_demod_unet(net_sr.unet, ref_de_mods)
    print(f"[new] set de_mod on {count} DeMod modules", flush=True)
    assert count == len(ref_de_mods), f"count mismatch: {count} vs {len(ref_de_mods)}"

    print("[new] forward on DeMod UNet...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        new_out = net_sr.unet(sample=sample, timestep=timestep,
                              encoder_hidden_states=enc, return_dict=False)[0]
    print(f"[new] shape={tuple(new_out.shape)} mean={new_out.mean():.4e} std={new_out.std():.4e} ({time.perf_counter()-t0:.1f}s)", flush=True)

    c = cos(ref_out, new_out)
    md = (ref_out - new_out).abs().max().item()
    mn = (ref_out - new_out).abs().mean().item()
    print(f"\n[compare] cos={c:.6f} max|diff|={md:.2e} mean|diff|={mn:.2e}")

    GATE = 0.999
    if c > GATE:
        print(f"[PASS] cosine {c:.6f} > {GATE}")
    else:
        print(f"[FAIL] cosine {c:.6f} <= {GATE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
