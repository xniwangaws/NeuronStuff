"""Phase 3: trace S3Diff UNet with de_mod as explicit tensor input.

Produces one NEFF per resolution bucket.  UNet forward signature becomes
(sample, timestep, encoder_hidden_states, unet_de_mods_stack) where
unet_de_mods_stack is [N_lora_layers, B, R, R].

Usage:
  python neuron_unet_trace_v3.py --latent 128 --out unet_1K.pt2 \
    --lq_image /home/ubuntu/s3diff/smoke_in/cat_LQ_256.png
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image


def compute_de_mods_stacked(net_sr, deg_score):
    """Return (unet_de_mods, vae_de_mods, unet_layer_names, vae_layer_names).

    unet_de_mods: Tensor [N_unet, B, R_unet, R_unet]
    vae_de_mods:  Tensor [N_vae,  B, R_vae,  R_vae]

    Identical math to neuron_unet_trace_v2.compute_and_bake_de_mod but returns
    stacked tensors instead of writing to module buffers.
    """
    # fourier embed of degradation score
    deg_proj = deg_score[..., None] * net_sr.W[None, None, :] * 2 * np.pi
    deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
    deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

    vae_de_c = net_sr.vae_de_mlp(deg_proj)
    unet_de_c = net_sr.unet_de_mlp(deg_proj)

    vae_block_c = net_sr.vae_block_mlp(net_sr.vae_block_embeddings.weight)
    unet_block_c = net_sr.unet_block_mlp(net_sr.unet_block_embeddings.weight)

    vae_embeds = net_sr.vae_fuse_mlp(torch.cat([
        vae_de_c.unsqueeze(1).repeat(1, vae_block_c.shape[0], 1),
        vae_block_c.unsqueeze(0).repeat(vae_de_c.shape[0], 1, 1),
    ], -1))
    unet_embeds = net_sr.unet_fuse_mlp(torch.cat([
        unet_de_c.unsqueeze(1).repeat(1, unet_block_c.shape[0], 1),
        unet_block_c.unsqueeze(0).repeat(unet_de_c.shape[0], 1, 1),
    ], -1))

    vae_names = list(net_sr.vae_lora_layers)
    vae_list = []
    for name in vae_names:
        split = name.split(".")
        if split[1] == "down_blocks":
            block_id = int(split[2])
            emb = vae_embeds[:, block_id]
        elif split[1] == "mid_block":
            emb = vae_embeds[:, -2]
        else:
            emb = vae_embeds[:, -1]
        vae_list.append(emb.reshape(-1, net_sr.lora_rank_vae, net_sr.lora_rank_vae))
    vae_stack = torch.stack(vae_list, dim=0)  # [N_vae, B, R_vae, R_vae]

    unet_names = list(net_sr.unet_lora_layers)
    unet_list = []
    for name in unet_names:
        split = name.split(".")
        if split[0] == "down_blocks":
            block_id = int(split[1])
            emb = unet_embeds[:, block_id]
        elif split[0] == "mid_block":
            emb = unet_embeds[:, 4]
        elif split[0] == "up_blocks":
            block_id = int(split[1]) + 5
            emb = unet_embeds[:, block_id]
        else:
            emb = unet_embeds[:, -1]
        unet_list.append(emb.reshape(-1, net_sr.lora_rank_unet, net_sr.lora_rank_unet))
    unet_stack = torch.stack(unet_list, dim=0)  # [N_unet, B, R_unet, R_unet]

    return unet_stack, vae_stack, unet_names, vae_names


class UNetWithDeMod(torch.nn.Module):
    """Wraps diffusers UNet to accept unet_de_mods as a 4th input.

    Sets DE_MOD_CTX before calling unet, so my_lora_fwd reads traced tensors
    from context instead of self.de_mod.
    """
    def __init__(self, unet, layer_names):
        super().__init__()
        self.unet = unet
        self.layer_names = ["unet::" + n for n in layer_names]

    def forward(self, sample, timestep, encoder_hidden_states, unet_de_mods):
        # unet_de_mods: [N, B, R, R]
        from de_mod_ctx import set_de_mods, reset_de_mods
        ctx = {name: unet_de_mods[i] for i, name in enumerate(self.layer_names)}
        token = set_de_mods(ctx)
        try:
            return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample
        finally:
            reset_de_mods(token)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png",
                   help="Any LQ image with non-degenerate deg_score (used only to get sensible example de_mod tensors for tracing; trace works for any deg_score afterward)")
    p.add_argument("--latent", type=int, default=128,
                   help="Latent side. 128 for 1K/single-tile, 256 for 2K/single-tile")
    p.add_argument("--out", required=True, help="Output NEFF path")
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    from s3diff_tile import S3Diff
    from de_net import DEResNet

    s3 = argparse.Namespace(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="x", neg_prompt="y",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    print(f"[load] S3Diff...", flush=True)
    t0 = time.perf_counter()
    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    # Compute example de_mods (for trace inputs).  Trace captures them as tensors
    # so any runtime deg_score's de_mods can replace these later.
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()
    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        deg_score = net_de(im_t.unsqueeze(0))
    unet_de_mods, vae_de_mods, unet_names, vae_names = compute_de_mods_stacked(net_sr, deg_score)
    print(f"[de_mod] unet stack {tuple(unet_de_mods.shape)}  vae stack {tuple(vae_de_mods.shape)}", flush=True)

    # Eager sanity: run UNet forward with de_mods from ctx
    wrapped = UNetWithDeMod(net_sr.unet, unet_names).eval()

    B = 1
    H = W = args.latent
    latent = torch.randn(B, 4, H, W)
    t = torch.tensor([999], dtype=torch.long)
    # Text embeds via CLIP (single positive prompt is fine; CFG done outside)
    with torch.no_grad():
        tokens = net_sr.tokenizer(
            ["A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."],
            max_length=77, padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids
        enc = net_sr.text_encoder(tokens)[0]

    print(f"[inputs] latent {tuple(latent.shape)} enc {tuple(enc.shape)} de_mods {tuple(unet_de_mods.shape)}", flush=True)

    print(f"[cpu-eager] single forward (to verify ctx plumbing)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        y_cpu = wrapped(latent, t, enc, unet_de_mods)
    print(f"[cpu-eager] {tuple(y_cpu.shape)}  time {time.perf_counter()-t0:.2f}s", flush=True)

    # Trace on Neuron
    print(f"[neuron] tracing at latent {H}x{W}...", flush=True)
    import torch_neuronx
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapped, (latent, t, enc, unet_de_mods),
        compiler_args=["--auto-cast", "none"],
    )
    print(f"[neuron] compile {time.perf_counter()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.jit.save(traced, args.out)
    print(f"[save] {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)

    print(f"[neuron] timing 3 runs...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        y_n = traced(latent, t, enc, unet_de_mods)
        print(f"  run {i+1}: {time.perf_counter()-t0:.3f}s", flush=True)

    # Critical accuracy check: feed a DIFFERENT de_mod and confirm output changes.
    # If ctx plumbing is broken (trace captured constant), output won't change.
    other_deg = deg_score + 0.5
    other_unet_de_mods, _, _, _ = compute_de_mods_stacked(net_sr, other_deg.clamp(0, 1))
    y_other = traced(latent, t, enc, other_unet_de_mods)
    diff_from_change = (y_other.float() - y_n.float()).abs().mean().item()
    print(f"[ctx-check] changing de_mods changes output by mean|diff|={diff_from_change:.5f}")
    print(f"  (If this is ~0, ContextVar wasn't captured; traced NEFF bakes constants.)")

    diff = (y_n.float() - y_cpu.float()).abs()
    print(f"[acc] Neuron vs CPU eager (same de_mods): max|diff| {diff.max():.5f}  mean {diff.mean():.5f}", flush=True)
    print(f"\nPHASE-3 UNet trace {args.out} DONE.")


if __name__ == "__main__":
    main()
