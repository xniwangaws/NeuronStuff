"""End-to-end test: one NeuronS3DiffBasicTransformerBlock matches diffusers reference.

Strategy:
1. Load the S3Diff model (CPU fp32).
2. Pick one attention Transformer2DModel block (e.g., `unet.down_blocks.0.attentions.0`).
3. Its first BasicTransformerBlock: unet.down_blocks.0.attentions.0.transformer_blocks.0.
4. Run the diffusers block as reference (via setting module.de_mod attributes as S3Diff normally does).
5. Build our NeuronS3DiffBasicTransformerBlock, load converted state_dict, feed same de_mod tensors as forward args.
6. Compare outputs.
Gate: cosine > 0.9999 (same math, fp32, should be essentially identical).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Expose S3Diff repo
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")

# Expose our new modules
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


def cosine(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main():
    print("[load] S3Diff CPU fp32...", flush=True)
    net_sr = load_s3diff()
    net_sr.eval()

    # Pick the reference block
    ref_mod = net_sr.unet.down_blocks[0].attentions[0].transformer_blocks[0]
    print(f"[ref] block: {type(ref_mod).__name__}")
    print(f"[ref] attn1 heads={ref_mod.attn1.heads} dim_head={ref_mod.attn1.to_q.base_layer.out_features // ref_mod.attn1.heads}")

    # Shapes for down_blocks.0 at latent 128x128: channels=320, tokens=16384, cross_dim=1024
    dim = 320
    heads = 5  # 320/64=5
    dim_head = 64
    cross_dim = 1024  # CLIP for sd-turbo
    seq_q = 16384  # too big for CPU; use smaller test-size
    seq_q_test = 256  # 16x16 latent-sized for speed
    seq_kv = 77

    # Set de_mod on the diffusers reference LoRA modules (the way S3Diff normally does)
    # Each LoRA-wrapped Linear inside this block has .de_mod attribute consumed by my_lora_fwd.
    # We'll generate 10 distinct random de_mod tensors for the 10 LoRA sites.
    torch.manual_seed(123)
    r = 32

    # Pre-generate de_mod tensors (one per LoRA site)
    de = lambda: torch.randn(1, r, r)
    de_mod_dict = {
        "attn1_to_q": de(),  "attn1_to_k": de(),  "attn1_to_v": de(),  "attn1_to_out": de(),
        "attn2_to_q": de(),  "attn2_to_k": de(),  "attn2_to_v": de(),  "attn2_to_out": de(),
        "ff_0_proj":  de(),  "ff_2":       de(),
    }
    # Assign to reference module
    ref_mod.attn1.to_q.de_mod   = de_mod_dict["attn1_to_q"]
    ref_mod.attn1.to_k.de_mod   = de_mod_dict["attn1_to_k"]
    ref_mod.attn1.to_v.de_mod   = de_mod_dict["attn1_to_v"]
    ref_mod.attn1.to_out[0].de_mod = de_mod_dict["attn1_to_out"]
    ref_mod.attn2.to_q.de_mod   = de_mod_dict["attn2_to_q"]
    ref_mod.attn2.to_k.de_mod   = de_mod_dict["attn2_to_k"]
    ref_mod.attn2.to_v.de_mod   = de_mod_dict["attn2_to_v"]
    ref_mod.attn2.to_out[0].de_mod = de_mod_dict["attn2_to_out"]
    ref_mod.ff.net[0].proj.de_mod  = de_mod_dict["ff_0_proj"]
    ref_mod.ff.net[2].de_mod       = de_mod_dict["ff_2"]

    # Random input + encoder_hidden_states
    x = torch.randn(1, seq_q_test, dim)
    enc = torch.randn(1, seq_kv, cross_dim)

    print("[ref] forward diffusers block...", flush=True)
    with torch.no_grad():
        ref_out = ref_mod(
            hidden_states=x,
            attention_mask=None,
            encoder_hidden_states=enc,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        )
    print(f"[ref] out shape: {tuple(ref_out.shape)}  mean={ref_out.mean():.4e}  std={ref_out.std():.4e}")

    # Now build our block and load weights from converted state_dict (or directly from ref module)
    print("[new] build NeuronS3DiffBasicTransformerBlock...", flush=True)
    from s3diff_transformer_block import NeuronS3DiffBasicTransformerBlock
    # S3Diff LoRA scaling is lora_alpha / r = 8 / 32 = 0.25 uniformly across UNet.
    LORA_SCALING = 0.25
    new_mod = NeuronS3DiffBasicTransformerBlock(
        dim=dim, num_attention_heads=heads, attention_head_dim=dim_head,
        cross_attention_dim=cross_dim, lora_rank=r, lora_scaling=LORA_SCALING,
    )
    new_mod.eval()

    # Copy weights from reference (peft-wrapped diffusers block) -> new module
    # Mapping:
    #   attn1.to_q.base_layer.* -> new.attn1.to_q.base.*
    #   attn1.to_q.lora_A.default.weight -> new.attn1.to_q.lora_A.weight
    #   attn1.to_q.lora_B.default.weight -> new.attn1.to_q.lora_B.weight
    #   ... same for k/v/out
    #   attn2.* -> new.attn2.*
    #   ff.net.0.proj.base_layer.* -> new.ff.net_0.proj.base.*
    #   ff.net.0.proj.lora_A.default.weight -> new.ff.net_0.proj.lora_A.weight
    #   ff.net.0.proj.lora_B.default.weight -> new.ff.net_0.proj.lora_B.weight
    #   ff.net.2.* -> new.ff.net_2.*
    #   norm1/2/3 -> norm1/2/3
    with torch.no_grad():
        # norms
        new_mod.norm1.load_state_dict(ref_mod.norm1.state_dict())
        new_mod.norm2.load_state_dict(ref_mod.norm2.state_dict())
        new_mod.norm3.load_state_dict(ref_mod.norm3.state_dict())

        def copy_lora_linear(dst, src):
            dst.base.weight.copy_(src.base_layer.weight)
            if src.base_layer.bias is not None:
                dst.base.bias.copy_(src.base_layer.bias)
            dst.lora_A.weight.copy_(src.lora_A.default.weight)
            dst.lora_B.weight.copy_(src.lora_B.default.weight)

        for attn_name in ["attn1", "attn2"]:
            src_attn = getattr(ref_mod, attn_name)
            dst_attn = getattr(new_mod, attn_name)
            copy_lora_linear(dst_attn.to_q, src_attn.to_q)
            copy_lora_linear(dst_attn.to_k, src_attn.to_k)
            copy_lora_linear(dst_attn.to_v, src_attn.to_v)
            copy_lora_linear(dst_attn.to_out_0, src_attn.to_out[0])

        copy_lora_linear(new_mod.ff.net_0.proj, ref_mod.ff.net[0].proj)
        copy_lora_linear(new_mod.ff.net_2, ref_mod.ff.net[2])

    # Forward with de_mod args
    print("[new] forward new block...", flush=True)
    with torch.no_grad():
        new_out = new_mod(
            hidden_states=x,
            encoder_hidden_states=enc,
            de_mod_attn1_q=de_mod_dict["attn1_to_q"],
            de_mod_attn1_k=de_mod_dict["attn1_to_k"],
            de_mod_attn1_v=de_mod_dict["attn1_to_v"],
            de_mod_attn1_o=de_mod_dict["attn1_to_out"],
            de_mod_attn2_q=de_mod_dict["attn2_to_q"],
            de_mod_attn2_k=de_mod_dict["attn2_to_k"],
            de_mod_attn2_v=de_mod_dict["attn2_to_v"],
            de_mod_attn2_o=de_mod_dict["attn2_to_out"],
            de_mod_ff_0=de_mod_dict["ff_0_proj"],
            de_mod_ff_2=de_mod_dict["ff_2"],
        )

    diff = (ref_out - new_out).abs()
    cos = cosine(ref_out, new_out)
    print(f"\n[compare]")
    print(f"  max|diff| = {diff.max().item():.4e}")
    print(f"  mean|diff|= {diff.mean().item():.4e}")
    print(f"  cosine    = {cos:.6f}")

    gate_cos = 0.9999
    if cos > gate_cos:
        print(f"[PASS] cosine {cos:.6f} > {gate_cos}")
    else:
        print(f"[FAIL] cosine {cos:.6f} <= {gate_cos}")
        sys.exit(1)


if __name__ == "__main__":
    main()
