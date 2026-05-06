"""Test DeModLoRA modules can load from converted Neuron-shaped state_dict.

Picks one known site from unet_sd_neuron.pt (e.g. down_blocks.0.attentions.0.proj_in)
and verifies that DeModLoRALinear loads its 3 parameters cleanly with strict=True.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")
from de_mod_lora import DeModLoRALinear, DeModLoRAConv2d


def extract_site(full_sd: dict, site_prefix: str) -> dict:
    """Extract keys under site_prefix and strip the prefix."""
    out = {}
    for k, v in full_sd.items():
        if k.startswith(site_prefix + "."):
            out[k[len(site_prefix) + 1:]] = v
    return out


def test_linear_load():
    sd = torch.load("/home/ubuntu/workspace/s3diff_nxdi/data/unet_sd_neuron.pt",
                    map_location="cpu", weights_only=True)
    site = "down_blocks.0.attentions.0.proj_in"
    sub_sd = extract_site(sd, site)
    print(f"[test_linear_load] keys at {site}: {list(sub_sd.keys())}")
    for k, v in sub_sd.items():
        print(f"  {k}  {tuple(v.shape)}")

    in_f = sub_sd["base.weight"].shape[1]
    out_f = sub_sd["base.weight"].shape[0]
    r = sub_sd["lora_A.weight"].shape[0]
    has_bias = "base.bias" in sub_sd
    print(f"  -> in={in_f} out={out_f} rank={r} bias={has_bias}")

    layer = DeModLoRALinear(in_f, out_f, r, bias=has_bias)
    layer.load_state_dict(sub_sd, strict=True)
    print("[test_linear_load] strict load OK")

    x = torch.randn(1, 128, in_f)
    de = torch.randn(1, r, r)
    y = layer(x, de)
    assert y.shape == (1, 128, out_f)
    print(f"[test_linear_load] forward shape OK: {tuple(y.shape)}")


def test_conv_load():
    sd = torch.load("/home/ubuntu/workspace/s3diff_nxdi/data/unet_sd_neuron.pt",
                    map_location="cpu", weights_only=True)
    site = "down_blocks.0.resnets.0.conv1"
    sub_sd = extract_site(sd, site)
    print(f"\n[test_conv_load] keys at {site}: {list(sub_sd.keys())}")
    for k, v in sub_sd.items():
        print(f"  {k}  {tuple(v.shape)}")

    # base.weight shape: (out_c, in_c, kH, kW)
    out_c, in_c, kH, kW = sub_sd["base.weight"].shape
    r = sub_sd["lora_A.weight"].shape[0]
    has_bias = "base.bias" in sub_sd
    assert kH == kW
    print(f"  -> in={in_c} out={out_c} k={kH} rank={r} bias={has_bias}")

    layer = DeModLoRAConv2d(in_c, out_c, kernel_size=kH, lora_rank=r, padding=kH // 2, bias=has_bias)
    layer.load_state_dict(sub_sd, strict=True)
    print("[test_conv_load] strict load OK")

    x = torch.randn(1, in_c, 32, 32)
    de = torch.randn(1, r, r)
    y = layer(x, de)
    assert y.shape == (1, out_c, 32, 32)
    print(f"[test_conv_load] forward shape OK: {tuple(y.shape)}")


if __name__ == "__main__":
    test_linear_load()
    test_conv_load()
    print("\n[LOAD ALL PASS]")
