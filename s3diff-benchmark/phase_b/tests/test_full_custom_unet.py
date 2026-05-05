"""End-to-end test: NeuronS3DiffUNet vs diffusers UNet2DConditionModel.

Steps:
1. Load S3Diff and its live diffusers UNet (peft LoRA wrapped).
2. Build our NeuronS3DiffUNet.
3. Build weight mapping: for every parameter in our UNet, find the matching
   parameter in the diffusers state_dict (via `.base_layer.*` / `.default.*` etc).
4. Assign de_mod on ref + build de_mod_map for ours.
5. Run both on a random sample, verify cosine > 0.999 and max|diff| small.

This validates the whole B1+B2 assembly before the Neuron bench.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")


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


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _copy_lora_linear(dst, src):
    with torch.no_grad():
        dst.base.weight.copy_(src.base_layer.weight)
        if src.base_layer.bias is not None:
            dst.base.bias.copy_(src.base_layer.bias)
        dst.lora_A.weight.copy_(src.lora_A.default.weight)
        dst.lora_B.weight.copy_(src.lora_B.default.weight)


def _copy_lora_conv(dst, src):
    with torch.no_grad():
        dst.base.weight.copy_(src.base_layer.weight)
        if src.base_layer.bias is not None:
            dst.base.bias.copy_(src.base_layer.bias)
        dst.lora_A.weight.copy_(src.lora_A.default.weight)
        dst.lora_B.weight.copy_(src.lora_B.default.weight)


def copy_weights(new_unet, ref_unet):
    """Copy weights from diffusers peft-wrapped UNet to our NeuronS3DiffUNet."""
    # conv_in (plain Conv2d, no LoRA)
    new_unet.conv_in.load_state_dict(ref_unet.conv_in.state_dict())

    # time_embedding
    new_unet.time_embedding.linear_1.load_state_dict(ref_unet.time_embedding.linear_1.state_dict())
    new_unet.time_embedding.linear_2.load_state_dict(ref_unet.time_embedding.linear_2.state_dict())

    # down_blocks
    for i, (new_blk, ref_blk) in enumerate(zip(new_unet.down_blocks, ref_unet.down_blocks)):
        for j, (n_res, r_res) in enumerate(zip(new_blk.resnets, ref_blk.resnets)):
            n_res.norm1.load_state_dict(r_res.norm1.state_dict())
            n_res.norm2.load_state_dict(r_res.norm2.state_dict())
            n_res.time_emb_proj.load_state_dict(r_res.time_emb_proj.state_dict())
            _copy_lora_conv(n_res.conv1, r_res.conv1)
            _copy_lora_conv(n_res.conv2, r_res.conv2)
            if n_res.use_shortcut:
                _copy_lora_conv(n_res.conv_shortcut, r_res.conv_shortcut)
        if hasattr(new_blk, "attentions"):
            for j, (n_attn, r_attn) in enumerate(zip(new_blk.attentions, ref_blk.attentions)):
                n_attn.norm.load_state_dict(r_attn.norm.state_dict())
                _copy_lora_linear(n_attn.proj_in, r_attn.proj_in)
                _copy_lora_linear(n_attn.proj_out, r_attn.proj_out)
                ref_tb = r_attn.transformer_blocks[0]
                nb = n_attn.transformer_blocks[0]
                nb.norm1.load_state_dict(ref_tb.norm1.state_dict())
                nb.norm2.load_state_dict(ref_tb.norm2.state_dict())
                nb.norm3.load_state_dict(ref_tb.norm3.state_dict())
                for a in ["attn1", "attn2"]:
                    s = getattr(ref_tb, a)
                    d = getattr(nb, a)
                    _copy_lora_linear(d.to_q, s.to_q)
                    _copy_lora_linear(d.to_k, s.to_k)
                    _copy_lora_linear(d.to_v, s.to_v)
                    _copy_lora_linear(d.to_out_0, s.to_out[0])
                _copy_lora_linear(nb.ff.net_0.proj, ref_tb.ff.net[0].proj)
                _copy_lora_linear(nb.ff.net_2, ref_tb.ff.net[2])
        if new_blk.downsamplers is not None:
            _copy_lora_conv(new_blk.downsamplers[0].conv, ref_blk.downsamplers[0].conv)

    # mid_block
    n_mid, r_mid = new_unet.mid_block, ref_unet.mid_block
    for j, (n_res, r_res) in enumerate(zip(n_mid.resnets, r_mid.resnets)):
        n_res.norm1.load_state_dict(r_res.norm1.state_dict())
        n_res.norm2.load_state_dict(r_res.norm2.state_dict())
        n_res.time_emb_proj.load_state_dict(r_res.time_emb_proj.state_dict())
        _copy_lora_conv(n_res.conv1, r_res.conv1)
        _copy_lora_conv(n_res.conv2, r_res.conv2)
        if n_res.use_shortcut:
            _copy_lora_conv(n_res.conv_shortcut, r_res.conv_shortcut)
    for j, (n_attn, r_attn) in enumerate(zip(n_mid.attentions, r_mid.attentions)):
        n_attn.norm.load_state_dict(r_attn.norm.state_dict())
        _copy_lora_linear(n_attn.proj_in, r_attn.proj_in)
        _copy_lora_linear(n_attn.proj_out, r_attn.proj_out)
        ref_tb = r_attn.transformer_blocks[0]
        nb = n_attn.transformer_blocks[0]
        nb.norm1.load_state_dict(ref_tb.norm1.state_dict())
        nb.norm2.load_state_dict(ref_tb.norm2.state_dict())
        nb.norm3.load_state_dict(ref_tb.norm3.state_dict())
        for a in ["attn1", "attn2"]:
            s = getattr(ref_tb, a)
            d = getattr(nb, a)
            _copy_lora_linear(d.to_q, s.to_q)
            _copy_lora_linear(d.to_k, s.to_k)
            _copy_lora_linear(d.to_v, s.to_v)
            _copy_lora_linear(d.to_out_0, s.to_out[0])
        _copy_lora_linear(nb.ff.net_0.proj, ref_tb.ff.net[0].proj)
        _copy_lora_linear(nb.ff.net_2, ref_tb.ff.net[2])

    # up_blocks
    for i, (new_blk, ref_blk) in enumerate(zip(new_unet.up_blocks, ref_unet.up_blocks)):
        for j, (n_res, r_res) in enumerate(zip(new_blk.resnets, ref_blk.resnets)):
            n_res.norm1.load_state_dict(r_res.norm1.state_dict())
            n_res.norm2.load_state_dict(r_res.norm2.state_dict())
            n_res.time_emb_proj.load_state_dict(r_res.time_emb_proj.state_dict())
            _copy_lora_conv(n_res.conv1, r_res.conv1)
            _copy_lora_conv(n_res.conv2, r_res.conv2)
            if n_res.use_shortcut:
                _copy_lora_conv(n_res.conv_shortcut, r_res.conv_shortcut)
        if hasattr(new_blk, "attentions"):
            for j, (n_attn, r_attn) in enumerate(zip(new_blk.attentions, ref_blk.attentions)):
                n_attn.norm.load_state_dict(r_attn.norm.state_dict())
                _copy_lora_linear(n_attn.proj_in, r_attn.proj_in)
                _copy_lora_linear(n_attn.proj_out, r_attn.proj_out)
                ref_tb = r_attn.transformer_blocks[0]
                nb = n_attn.transformer_blocks[0]
                nb.norm1.load_state_dict(ref_tb.norm1.state_dict())
                nb.norm2.load_state_dict(ref_tb.norm2.state_dict())
                nb.norm3.load_state_dict(ref_tb.norm3.state_dict())
                for a in ["attn1", "attn2"]:
                    s = getattr(ref_tb, a)
                    d = getattr(nb, a)
                    _copy_lora_linear(d.to_q, s.to_q)
                    _copy_lora_linear(d.to_k, s.to_k)
                    _copy_lora_linear(d.to_v, s.to_v)
                    _copy_lora_linear(d.to_out_0, s.to_out[0])
                _copy_lora_linear(nb.ff.net_0.proj, ref_tb.ff.net[0].proj)
                _copy_lora_linear(nb.ff.net_2, ref_tb.ff.net[2])
        if new_blk.upsamplers is not None:
            _copy_lora_conv(new_blk.upsamplers[0].conv, ref_blk.upsamplers[0].conv)

    # output conv
    new_unet.conv_norm_out.load_state_dict(ref_unet.conv_norm_out.state_dict())
    _copy_lora_conv(new_unet.conv_out, ref_unet.conv_out)


def gather_de_mods(ref_unet, rank=32, seed=42):
    """Assign random de_mod to every peft module on ref_unet, and return dict {site_name: tensor}.

    Keys match the LoRA site names as S3Diff's assignment loop writes them.
    """
    from peft.tuners.lora.layer import Linear as PeftLinear, Conv2d as PeftConv2d
    torch.manual_seed(seed)
    out = {}
    for name, m in ref_unet.named_modules():
        if isinstance(m, (PeftLinear, PeftConv2d)):
            t = torch.randn(1, rank, rank)
            m.de_mod = t
            out[name] = t
    return out


def main():
    print("[load] S3Diff CPU fp32 (peft ref)...", flush=True)
    net_sr = build_s3diff()
    ref_unet = net_sr.unet

    from s3diff_unet import NeuronS3DiffUNet
    print("[build] NeuronS3DiffUNet (scratch)...", flush=True)
    new_unet = NeuronS3DiffUNet(lora_rank=32, lora_scaling=0.25)
    new_unet.eval()

    print("[copy] weights from ref -> new...", flush=True)
    t0 = time.perf_counter()
    copy_weights(new_unet, ref_unet)
    print(f"[copy] done {time.perf_counter()-t0:.1f}s", flush=True)

    # Gather de_mod
    print("[de_mod] assign random to ref + build map for new...", flush=True)
    de_mod_map = gather_de_mods(ref_unet, rank=32, seed=42)
    print(f"[de_mod] {len(de_mod_map)} sites assigned")

    # Forward test
    torch.manual_seed(0)
    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([999])
    enc = torch.randn(1, 77, 1024)

    print("[run] ref UNet forward...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_out = ref_unet(sample=sample, timestep=timestep,
                           encoder_hidden_states=enc, return_dict=False)[0]
    print(f"[run] ref done {time.perf_counter()-t0:.1f}s shape={tuple(ref_out.shape)}", flush=True)

    print("[run] new UNet forward...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        new_out = new_unet(sample=sample, timestep=timestep,
                           encoder_hidden_states=enc, de_mod_map=de_mod_map)
    print(f"[run] new done {time.perf_counter()-t0:.1f}s shape={tuple(new_out.shape)}", flush=True)

    c = cos(ref_out, new_out)
    md = (ref_out - new_out).abs().max().item()
    mn = (ref_out - new_out).abs().mean().item()
    print(f"\n[compare] cos={c:.6f} max|diff|={md:.2e} mean|diff|={mn:.2e}")
    GATE = 0.999
    if c > GATE:
        print(f"[PASS] cos {c:.6f} > {GATE}")
    else:
        print(f"[FAIL] cos {c:.6f} <= {GATE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
