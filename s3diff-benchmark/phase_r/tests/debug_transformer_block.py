"""Debug which sub-stage of the block diverges: attn1 / attn2 / ff / norms."""
import sys
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")

import torch
import torch.nn.functional as F


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main():
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

    ref = net_sr.unet.down_blocks[0].attentions[0].transformer_blocks[0]

    # Print the structure of the block
    print("[ref] Block repr:")
    # Show just direct children names
    for name, child in ref.named_children():
        print(f"  {name}: {type(child).__name__}")
    print()
    print("[ref] ff.net structure:")
    for i, sub in enumerate(ref.ff.net):
        print(f"  ff.net[{i}] = {type(sub).__name__}")
    print()
    print("[ref] block attrs:")
    for k in ["only_cross_attention", "use_ada_layer_norm", "use_ada_layer_norm_zero",
              "use_ada_layer_norm_single", "use_layer_norm", "use_ada_layer_norm_continuous",
              "norm_type", "num_embeds_ada_norm", "ada_norm_bias", "ada_norm_continous_conditioning_embedding_dim",
              "pos_embed", "attention_type", "_chunk_size", "_chunk_dim"]:
        v = getattr(ref, k, "<missing>")
        print(f"  {k} = {v}")

    # Import our block
    from s3diff_transformer_block import NeuronS3DiffBasicTransformerBlock

    # Setup
    torch.manual_seed(123)
    r, dim, heads, dim_head, cross_dim = 32, 320, 5, 64, 1024
    seq_q, seq_kv = 256, 77
    x = torch.randn(1, seq_q, dim)
    enc = torch.randn(1, seq_kv, cross_dim)

    def de(): return torch.randn(1, r, r)
    de_dict = {k: de() for k in ["a1q","a1k","a1v","a1o","a2q","a2k","a2v","a2o","ff0","ff2"]}

    # Wire de_mod on ref
    ref.attn1.to_q.de_mod    = de_dict["a1q"]
    ref.attn1.to_k.de_mod    = de_dict["a1k"]
    ref.attn1.to_v.de_mod    = de_dict["a1v"]
    ref.attn1.to_out[0].de_mod = de_dict["a1o"]
    ref.attn2.to_q.de_mod    = de_dict["a2q"]
    ref.attn2.to_k.de_mod    = de_dict["a2k"]
    ref.attn2.to_v.de_mod    = de_dict["a2v"]
    ref.attn2.to_out[0].de_mod = de_dict["a2o"]
    ref.ff.net[0].proj.de_mod  = de_dict["ff0"]
    ref.ff.net[2].de_mod       = de_dict["ff2"]

    new = NeuronS3DiffBasicTransformerBlock(
        dim=dim, num_attention_heads=heads, attention_head_dim=dim_head,
        cross_attention_dim=cross_dim, lora_rank=r, lora_scaling=1.0,
    ).eval()

    # Copy weights
    with torch.no_grad():
        new.norm1.load_state_dict(ref.norm1.state_dict())
        new.norm2.load_state_dict(ref.norm2.state_dict())
        new.norm3.load_state_dict(ref.norm3.state_dict())
        def copy_l(d, s):
            d.base.weight.copy_(s.base_layer.weight)
            if s.base_layer.bias is not None:
                d.base.bias.copy_(s.base_layer.bias)
            d.lora_A.weight.copy_(s.lora_A.default.weight)
            d.lora_B.weight.copy_(s.lora_B.default.weight)
        for a in ["attn1", "attn2"]:
            copy_l(getattr(new, a).to_q,   getattr(ref, a).to_q)
            copy_l(getattr(new, a).to_k,   getattr(ref, a).to_k)
            copy_l(getattr(new, a).to_v,   getattr(ref, a).to_v)
            copy_l(getattr(new, a).to_out_0, getattr(ref, a).to_out[0])
        copy_l(new.ff.net_0.proj, ref.ff.net[0].proj)
        copy_l(new.ff.net_2,     ref.ff.net[2])

    # Stage 1: norm1 comparison
    with torch.no_grad():
        r_n1 = ref.norm1(x)
        n_n1 = new.norm1(x)
        print(f"\n[stage norm1]     cos={cos(r_n1, n_n1):.6f}  max|diff|={(r_n1-n_n1).abs().max():.2e}")

        # Stage 2: attn1 only
        r_a1 = ref.attn1(r_n1)  # diffusers attn processor signature
        n_a1 = new.attn1(n_n1, None, de_dict["a1q"], de_dict["a1k"], de_dict["a1v"], de_dict["a1o"])
        print(f"[stage attn1]     cos={cos(r_a1, n_a1):.6f}  max|diff|={(r_a1-n_a1).abs().max():.2e}")

        # Stage 3: after attn1 residual
        r_after_a1 = r_a1 + x
        n_after_a1 = n_a1 + x
        print(f"[stage after-a1] cos={cos(r_after_a1, n_after_a1):.6f}  max|diff|={(r_after_a1-n_after_a1).abs().max():.2e}")

        # Stage 4: norm2
        r_n2 = ref.norm2(r_after_a1)
        n_n2 = new.norm2(n_after_a1)
        print(f"[stage norm2]     cos={cos(r_n2, n_n2):.6f}  max|diff|={(r_n2-n_n2).abs().max():.2e}")

        # Stage 5: attn2
        r_a2 = ref.attn2(r_n2, encoder_hidden_states=enc)
        n_a2 = new.attn2(n_n2, enc, de_dict["a2q"], de_dict["a2k"], de_dict["a2v"], de_dict["a2o"])
        print(f"[stage attn2]     cos={cos(r_a2, n_a2):.6f}  max|diff|={(r_a2-n_a2).abs().max():.2e}")

        # Stage 6: after attn2 residual
        r_after_a2 = r_a2 + r_after_a1
        n_after_a2 = n_a2 + n_after_a1
        print(f"[stage after-a2] cos={cos(r_after_a2, n_after_a2):.6f}")

        # Stage 7: norm3
        r_n3 = ref.norm3(r_after_a2)
        n_n3 = new.norm3(n_after_a2)
        print(f"[stage norm3]     cos={cos(r_n3, n_n3):.6f}")

        # Stage 8: ff
        r_ff = ref.ff(r_n3)
        n_ff = new.ff(n_n3, de_dict["ff0"], de_dict["ff2"])
        print(f"[stage ff]        cos={cos(r_ff, n_ff):.6f}  max|diff|={(r_ff-n_ff).abs().max():.2e}")


if __name__ == "__main__":
    main()
