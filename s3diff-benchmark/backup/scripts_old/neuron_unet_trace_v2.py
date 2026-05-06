"""
Phase 1 v2: Trace S3Diff UNet with de_mod baked in as constants.

Approach:
  1. Compute degradation score once on a reference LQ image
  2. Run S3Diff's de_mod computation path (vae_de_mlp -> unet_fuse_mlp etc.)
  3. Write each per-layer de_mod tensor as a non-trainable buffer on the LoRA module
  4. Monkey-patch my_lora_fwd to read that buffer (already does — no change needed)
  5. torch_neuronx.trace() now sees de_mod as a constant in the graph

Limitation: traced UNet is only valid for that specific deg_score.  At benchmark
time we use a fixed LQ reference image + seed; at production time we'd either
re-trace per image (expensive) or split the degradation-dependent LoRA into a
separate small Neuron-traced module that stays in Python (future work).

Scope: 1K bucket (LQ 256 -> SR 1024, VAE latent 32x32, no latent tiling).
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/lq_00.png")
    p.add_argument("--out_dir", default="/home/ubuntu/s3diff/neuron_out")
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    p.add_argument("--resolution", choices=["1K", "2K", "4K"], default="1K")
    return p.parse_args()


def compute_and_bake_de_mod(net_sr, deg_score):
    """Replicate S3Diff.forward's de_mod computation and write into each LoRA layer."""
    # from s3diff_tile.py lines 252-297 (de_mod derivation)
    # 1) fourier embedding of degradation score
    deg_proj = deg_score[..., None] * net_sr.W[None, None, :] * 2 * np.pi
    deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
    deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

    vae_de_c_embed = net_sr.vae_de_mlp(deg_proj)
    unet_de_c_embed = net_sr.unet_de_mlp(deg_proj)

    vae_block_c_embeds = net_sr.vae_block_mlp(net_sr.vae_block_embeddings.weight)
    unet_block_c_embeds = net_sr.unet_block_mlp(net_sr.unet_block_embeddings.weight)

    vae_embeds = net_sr.vae_fuse_mlp(torch.cat([
        vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1),
        vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0], 1, 1),
    ], -1))
    unet_embeds = net_sr.unet_fuse_mlp(torch.cat([
        unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),
        unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0], 1, 1),
    ], -1))

    # 2) write per-layer de_mod into each LoRA module as a buffer (so trace sees it)
    for layer_name, module in net_sr.vae.named_modules():
        if layer_name in net_sr.vae_lora_layers:
            split_name = layer_name.split(".")
            if split_name[1] == "down_blocks":
                block_id = int(split_name[2])
                vae_embed = vae_embeds[:, block_id]
            elif split_name[1] == "mid_block":
                vae_embed = vae_embeds[:, -2]
            else:
                vae_embed = vae_embeds[:, -1]
            de_mod = vae_embed.reshape(-1, net_sr.lora_rank_vae, net_sr.lora_rank_vae).detach().clone()
            # register as buffer so it tracks device moves during trace
            if hasattr(module, "de_mod") and "de_mod" in module._buffers:
                module._buffers.pop("de_mod")
            module.register_buffer("de_mod", de_mod, persistent=False)

    for layer_name, module in net_sr.unet.named_modules():
        if layer_name in net_sr.unet_lora_layers:
            split_name = layer_name.split(".")
            if split_name[0] == "down_blocks":
                block_id = int(split_name[1])
                unet_embed = unet_embeds[:, block_id]
            elif split_name[0] == "mid_block":
                unet_embed = unet_embeds[:, 4]
            elif split_name[0] == "up_blocks":
                block_id = int(split_name[1]) + 5
                unet_embed = unet_embeds[:, block_id]
            else:
                unet_embed = unet_embeds[:, -1]
            de_mod = unet_embed.reshape(-1, net_sr.lora_rank_unet, net_sr.lora_rank_unet).detach().clone()
            if hasattr(module, "de_mod") and "de_mod" in module._buffers:
                module._buffers.pop("de_mod")
            module.register_buffer("de_mod", de_mod, persistent=False)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from s3diff_tile import S3Diff
    from de_net import DEResNet

    # latent side per resolution bucket (S3Diff x4 upscale, VAE 8x downsample)
    # LQ side -> resize to x4 -> VAE encode /8
    # 1K: LQ 256 -> x4=1024, vae->128.  But S3Diff also works on `c_t` = x4-upsampled LQ
    # Actually looking at S3Diff.forward: c_t is the LQ *after* x4 bicubic resize.
    # For 1K LQ=256, c_t=1024 -> vae_encode -> 128x128 latent. That's > 96 tile -> tiled.
    # Hmm, but the tile size is 96 for latent side. At 1K we'd tile.
    # Let me think again: smoke test at 1K ran without tiling (log said "tiny, no tile").
    # That was because LQ 256x256 -> c_t 1024x1024 bicubic -> VAE latent 128x128. 128*128=16384.
    # tile_size*tile_size = 96*96 = 9216. 16384 > 9216 -> should tile.
    # But the earlier "untiled" log said size 1024x1024 = c_t. Not latent. So the tile check
    # uses latent h*w vs tile**2.  128*128=16384 > 9216 -> should tile.  Yet log said "no tile"?
    # Need to check carefully.  For now pick a tile-sized forward: latent 96x96.
    latent_side_map = {"1K": 32, "2K": 64, "4K": 96}  # latent at each bucket
    # Actually at 1K: LQ=256, c_t=1024, vae_latent = 128.  S3Diff's latent_tiled_size = 96,
    # so latent 128 would tile (ceil(128/(96-32))*...).  The traced UNet forward takes 1 tile
    # at a time, which is 96x96.  So latent_side=96 is the right trace shape.  1K-untiled
    # is the special case smoke test hit earlier — probably because the check compares h*w
    # vs tile**2 and we'll need to verify.
    # For benchmark trace purposes pick 96 to match tile forward (works for 1K/2K/4K).
    latent_side = 96

    print(f"[load] S3Diff...", flush=True)
    s3 = argparse.Namespace(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        sd_path=args.sd_path,
        pretrained_path=args.pretrained_path,
    )
    t0 = time.perf_counter()
    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    # Compute a real degradation score from LQ image
    print("[deg] running DEResNet on reference LQ...", flush=True)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    # NCHW, [0,1]
    im_t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    im_t = im_t.unsqueeze(0)
    with torch.no_grad():
        deg_score = net_de(im_t)
    print(f"[deg] deg_score shape {tuple(deg_score.shape)} values {deg_score.flatten().tolist()}", flush=True)
    # Bake de_mod at B=1 (trace batch).  CFG done in Python: call traced UNet
    # twice (once with pos enc_hidden, once with neg).
    compute_and_bake_de_mod(net_sr, deg_score)
    print(f"[bake] de_mod baked into {len(net_sr.unet_lora_layers)} UNet + {len(net_sr.vae_lora_layers)} VAE LoRA layers", flush=True)

    # Trace at B=1 (lower compile memory). CFG pos/neg done in Python by calling
    # traced UNet twice. This cuts compile-time memory in half and lets the trn2
    # system RAM (124 GB) handle SD-Turbo UNet at latent 96x96.
    # But de_mod was computed at B=2 (CFG).  Slice it back to B=1 for trace.
    # At run time we still need both pos and neg de_mods — they're the same because
    # deg_score is identical per image.  So B=1 is fine.
    B = 1
    latent = torch.randn(B, 4, latent_side, latent_side)
    t = torch.tensor([999], dtype=torch.long)
    # Text embed from S3Diff's text encoder
    with torch.no_grad():
        tokens = net_sr.tokenizer(
            [s3.pos_prompt] * B, max_length=77, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids
        enc = net_sr.text_encoder(tokens)[0]

    print(f"[inputs] latent {tuple(latent.shape)}  t {t.shape}  enc {tuple(enc.shape)}", flush=True)

    print("[cpu-eager] UNet forward...", flush=True)
    unet = net_sr.unet.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_cpu = unet(latent, t, encoder_hidden_states=enc).sample
    print(f"[cpu-eager] out {tuple(y_cpu.shape)}  time {time.perf_counter()-t0:.2f}s", flush=True)

    # Trace on Neuron
    print("[neuron] tracing UNet (latent {}x{})...".format(latent_side, latent_side), flush=True)
    import torch_neuronx

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    wrapped = UNetWrapper(unet).eval()
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapped, (latent, t, enc),
        compiler_args=["--auto-cast", "none"],
    )
    compile_s = time.perf_counter() - t0
    print(f"[neuron] compile {compile_s:.1f}s", flush=True)

    save_path = os.path.join(args.out_dir, f"unet_latent{latent_side}.pt2")
    torch.jit.save(traced, save_path)
    print(f"[save] {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)", flush=True)

    print("[neuron] timing 3 forwards...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        y_n = traced(latent, t, enc)
        dt = time.perf_counter() - t0
        print(f"  run {i+1}: {dt:.3f}s", flush=True)

    diff = (y_n.float() - y_cpu.float()).abs()
    print(f"[acc] max|diff| {diff.max().item():.5f}  mean {diff.mean().item():.5f}", flush=True)
    print("\nPHASE-1 DONE.")


if __name__ == "__main__":
    main()
