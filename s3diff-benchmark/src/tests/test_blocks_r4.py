"""Test NeuronS3DiffTransformer2DModel + NeuronS3DiffResnetBlock2D vs diffusers reference.

Gate: cosine > 0.9999 on CPU fp32 for each.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")


def load_s3diff():
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


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def copy_lora_linear(dst, src):
    with torch.no_grad():
        dst.base.weight.copy_(src.base_layer.weight)
        if src.base_layer.bias is not None:
            dst.base.bias.copy_(src.base_layer.bias)
        dst.lora_A.weight.copy_(src.lora_A.default.weight)
        dst.lora_B.weight.copy_(src.lora_B.default.weight)


def copy_lora_conv(dst, src):
    with torch.no_grad():
        dst.base.weight.copy_(src.base_layer.weight)
        if src.base_layer.bias is not None:
            dst.base.bias.copy_(src.base_layer.bias)
        dst.lora_A.weight.copy_(src.lora_A.default.weight)
        dst.lora_B.weight.copy_(src.lora_B.default.weight)


def test_transformer2d(net_sr):
    from s3diff_transformer_2d import NeuronS3DiffTransformer2DModel

    print("\n=== Transformer2D test ===")
    ref = net_sr.unet.down_blocks[0].attentions[0]
    print(f"ref: {type(ref).__name__}, in_channels={ref.in_channels}")

    r = 32
    LORA_SCALING = 0.25
    dim = 320
    heads = 5
    dim_head = 64
    cross_dim = 1024

    new = NeuronS3DiffTransformer2DModel(
        in_channels=dim, num_attention_heads=heads, attention_head_dim=dim_head,
        cross_attention_dim=cross_dim, lora_rank=r, lora_scaling=LORA_SCALING,
    ).eval()

    # Copy weights
    new.norm.load_state_dict(ref.norm.state_dict())
    copy_lora_linear(new.proj_in, ref.proj_in)
    copy_lora_linear(new.proj_out, ref.proj_out)
    ref_block = ref.transformer_blocks[0]
    nb = new.transformer_blocks[0]
    nb.norm1.load_state_dict(ref_block.norm1.state_dict())
    nb.norm2.load_state_dict(ref_block.norm2.state_dict())
    nb.norm3.load_state_dict(ref_block.norm3.state_dict())
    for attn_n in ["attn1", "attn2"]:
        s = getattr(ref_block, attn_n)
        d = getattr(nb, attn_n)
        copy_lora_linear(d.to_q, s.to_q)
        copy_lora_linear(d.to_k, s.to_k)
        copy_lora_linear(d.to_v, s.to_v)
        copy_lora_linear(d.to_out_0, s.to_out[0])
    copy_lora_linear(nb.ff.net_0.proj, ref_block.ff.net[0].proj)
    copy_lora_linear(nb.ff.net_2, ref_block.ff.net[2])

    # de_mods
    torch.manual_seed(1)
    def de(): return torch.randn(1, r, r)
    de_proj_in = de(); de_proj_out = de()
    de_ks = ["attn1_q","attn1_k","attn1_v","attn1_o","attn2_q","attn2_k","attn2_v","attn2_o","ff_0","ff_2"]
    de_block = {k: de() for k in de_ks}

    # Set on reference
    ref.proj_in.de_mod  = de_proj_in
    ref.proj_out.de_mod = de_proj_out
    ref_block.attn1.to_q.de_mod = de_block["attn1_q"]
    ref_block.attn1.to_k.de_mod = de_block["attn1_k"]
    ref_block.attn1.to_v.de_mod = de_block["attn1_v"]
    ref_block.attn1.to_out[0].de_mod = de_block["attn1_o"]
    ref_block.attn2.to_q.de_mod = de_block["attn2_q"]
    ref_block.attn2.to_k.de_mod = de_block["attn2_k"]
    ref_block.attn2.to_v.de_mod = de_block["attn2_v"]
    ref_block.attn2.to_out[0].de_mod = de_block["attn2_o"]
    ref_block.ff.net[0].proj.de_mod  = de_block["ff_0"]
    ref_block.ff.net[2].de_mod       = de_block["ff_2"]

    # Input
    B, C, H, W = 1, 320, 16, 16
    x = torch.randn(B, C, H, W)
    enc = torch.randn(B, 77, cross_dim)

    with torch.no_grad():
        ref_out = ref(
            hidden_states=x,
            encoder_hidden_states=enc,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]

        new_out = new(
            hidden_states=x,
            encoder_hidden_states=enc,
            de_mod_proj_in=de_proj_in,
            de_mod_proj_out=de_proj_out,
            de_mod_attn1_q=de_block["attn1_q"],
            de_mod_attn1_k=de_block["attn1_k"],
            de_mod_attn1_v=de_block["attn1_v"],
            de_mod_attn1_o=de_block["attn1_o"],
            de_mod_attn2_q=de_block["attn2_q"],
            de_mod_attn2_k=de_block["attn2_k"],
            de_mod_attn2_v=de_block["attn2_v"],
            de_mod_attn2_o=de_block["attn2_o"],
            de_mod_ff_0=de_block["ff_0"],
            de_mod_ff_2=de_block["ff_2"],
        )

    c = cos(ref_out, new_out)
    md = (ref_out - new_out).abs().max().item()
    mn = (ref_out - new_out).abs().mean().item()
    print(f"Transformer2D: cos={c:.6f} max|diff|={md:.2e} mean|diff|={mn:.2e}")
    if c > 0.9999:
        print("[PASS]")
    else:
        print("[FAIL]")
        sys.exit(1)


def test_resnet(net_sr):
    from s3diff_resnet import NeuronS3DiffResnetBlock2D

    print("\n=== ResnetBlock test ===")
    # Pick a resnet with channel change (conv_shortcut required)
    ref = net_sr.unet.down_blocks[1].resnets[0]  # 320->640 likely
    print(f"ref: {type(ref).__name__}, in={ref.in_channels}, out={ref.out_channels}")
    print(f"ref.conv_shortcut: {ref.conv_shortcut}")

    r = 32
    LORA_SCALING = 0.25
    temb_ch = 1280

    # Inspect GN num_groups + eps
    print(f"ref.norm1 groups={ref.norm1.num_groups} eps={ref.norm1.eps}")

    new = NeuronS3DiffResnetBlock2D(
        in_channels=ref.in_channels,
        out_channels=ref.out_channels,
        temb_channels=temb_ch,
        lora_rank=r,
        lora_scaling=LORA_SCALING,
        groups=ref.norm1.num_groups,
        eps=ref.norm1.eps,
        use_shortcut=(ref.conv_shortcut is not None),
    ).eval()

    # Copy weights
    new.norm1.load_state_dict(ref.norm1.state_dict())
    new.norm2.load_state_dict(ref.norm2.state_dict())
    new.time_emb_proj.load_state_dict(ref.time_emb_proj.state_dict())
    copy_lora_conv(new.conv1, ref.conv1)
    copy_lora_conv(new.conv2, ref.conv2)
    if new.use_shortcut:
        copy_lora_conv(new.conv_shortcut, ref.conv_shortcut)

    # de_mods (r, r) per LoRA site
    torch.manual_seed(2)
    de_conv1 = torch.randn(1, r, r)
    de_conv2 = torch.randn(1, r, r)
    de_conv_shortcut = torch.randn(1, r, r) if new.use_shortcut else None

    ref.conv1.de_mod = de_conv1
    ref.conv2.de_mod = de_conv2
    if new.use_shortcut:
        ref.conv_shortcut.de_mod = de_conv_shortcut

    # Input
    B, H, W = 1, ref.in_channels, 32
    x = torch.randn(1, ref.in_channels, H, W)
    t = torch.randn(1, temb_ch)

    with torch.no_grad():
        ref_out = ref(x, t)
        new_out = new(x, t,
                      de_mod_conv1=de_conv1, de_mod_conv2=de_conv2,
                      de_mod_conv_shortcut=de_conv_shortcut)

    c = cos(ref_out, new_out)
    md = (ref_out - new_out).abs().max().item()
    mn = (ref_out - new_out).abs().mean().item()
    print(f"ResnetBlock: cos={c:.6f} max|diff|={md:.2e} mean|diff|={mn:.2e}")
    if c > 0.9999:
        print("[PASS]")
    else:
        print("[FAIL]")
        sys.exit(1)


if __name__ == "__main__":
    net_sr = load_s3diff()
    test_transformer2d(net_sr)
    test_resnet(net_sr)
    print("\n[ALL PASS]")
