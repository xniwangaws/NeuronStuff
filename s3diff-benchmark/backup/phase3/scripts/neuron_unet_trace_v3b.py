"""Phase 3 Plan B: attribute-routing trace (fallback if ContextVar fails).

Writes unet_de_mods[i] to each LoRA layer's .de_mod inside the traced forward.
Because the write happens inside forward() and the right-hand side is a slice
of a traced input tensor, XLA captures the data flow: input -> slice -> attr -> einsum.

Advantage over v3 (Plan A / ContextVar): no patching of model.py or s3diff_tile.py
required.  Only changes the UNet wrapper.

Usage:
  python neuron_unet_trace_v3b.py --latent 128 --out unet_1K_v3b.pt2 \
    --lq_image /home/ubuntu/s3diff/smoke_in/cat_LQ_256.png
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image


def compute_de_mods_stacked(net_sr, deg_score):
    """Stack per-layer de_mods into a single tensor. Returns (unet_stack, vae_stack, unet_modules, vae_modules)."""
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

    # Collect UNet LoRA modules IN ORDER (dict iteration order is insertion order in Py3.7+)
    unet_mods = []
    unet_list = []
    for name, mod in net_sr.unet.named_modules():
        if name in net_sr.unet_lora_layers:
            split = name.split(".")
            if split[0] == "down_blocks":
                emb = unet_embeds[:, int(split[1])]
            elif split[0] == "mid_block":
                emb = unet_embeds[:, 4]
            elif split[0] == "up_blocks":
                emb = unet_embeds[:, int(split[1]) + 5]
            else:
                emb = unet_embeds[:, -1]
            unet_mods.append(mod)
            unet_list.append(emb.reshape(-1, net_sr.lora_rank_unet, net_sr.lora_rank_unet))
    unet_stack = torch.stack(unet_list, dim=0)

    vae_mods = []
    vae_list = []
    for name, mod in net_sr.vae.named_modules():
        if name in net_sr.vae_lora_layers:
            split = name.split(".")
            if split[1] == "down_blocks":
                emb = vae_embeds[:, int(split[2])]
            elif split[1] == "mid_block":
                emb = vae_embeds[:, -2]
            else:
                emb = vae_embeds[:, -1]
            vae_mods.append(mod)
            vae_list.append(emb.reshape(-1, net_sr.lora_rank_vae, net_sr.lora_rank_vae))
    vae_stack = torch.stack(vae_list, dim=0)

    return unet_stack, vae_stack, unet_mods, vae_mods


class UNetAttrRouted(torch.nn.Module):
    """Plan B: write unet_de_mods[i] to each LoRA module's .de_mod INSIDE forward.

    This is critical: the assignment must be inside the traced call so that XLA
    sees the data flow input -> slice -> einsum (via self.de_mod in my_lora_fwd).
    """
    def __init__(self, unet, lora_modules):
        super().__init__()
        self.unet = unet
        # Keep references to the LoRA modules we'll set .de_mod on.
        # Use a list (not nn.ModuleList) because these modules are already part
        # of self.unet and we don't want to re-register them.
        self._lora_module_refs = lora_modules

    def forward(self, sample, timestep, encoder_hidden_states, unet_de_mods):
        # unet_de_mods: [N, B, R, R]
        for i, mod in enumerate(self._lora_module_refs):
            mod.de_mod = unet_de_mods[i]
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--out", required=True)
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
        padding_offset=32, pos_prompt="x", neg_prompt="y",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    print("[load] S3Diff...", flush=True)
    t0 = time.perf_counter()
    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    # Use cat image as source of example de_mod tensors (used only for trace shape/dtype)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()
    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        deg_score = net_de(im_t.unsqueeze(0))
    unet_de_mods, _, unet_mods, _ = compute_de_mods_stacked(net_sr, deg_score)
    print(f"[de_mod] unet stack {tuple(unet_de_mods.shape)}", flush=True)

    wrapped = UNetAttrRouted(net_sr.unet, unet_mods).eval()

    B, C = 1, 4
    H = W = args.latent
    latent = torch.randn(B, C, H, W)
    t = torch.tensor([999], dtype=torch.long)
    with torch.no_grad():
        tokens = net_sr.tokenizer(
            ["A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."],
            max_length=77, padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids
        enc = net_sr.text_encoder(tokens)[0]

    print(f"[inputs] latent {tuple(latent.shape)}  enc {tuple(enc.shape)}  de_mods {tuple(unet_de_mods.shape)}", flush=True)

    print("[cpu-eager] single forward (wrap+attr)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        y_cpu = wrapped(latent, t, enc, unet_de_mods)
    print(f"[cpu-eager] {tuple(y_cpu.shape)}  time {time.perf_counter()-t0:.2f}s", flush=True)

    print("[neuron] tracing...", flush=True)
    import torch_neuronx
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapped, (latent, t, enc, unet_de_mods),
        compiler_args=["--auto-cast", "none"],
    )
    print(f"[neuron] compile {time.perf_counter()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.jit.save(traced, args.out)
    print(f"[save] {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)

    # Timing
    print("[neuron] 3 timed runs...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        y_n = traced(latent, t, enc, unet_de_mods)
        print(f"  run {i+1}: {time.perf_counter()-t0:.3f}s", flush=True)

    # ========================= CRITICAL CHECK =========================
    # Per TEXT_TO_VIDEO_MODEL_PORTING.md §12.2, XLA silently drops unused inputs.
    # If attribute routing failed, traced NEFF uses the baked de_mods regardless
    # of the unet_de_mods input.  Verify by feeding a DIFFERENT de_mod tensor
    # and checking the output changes.
    other_deg = (deg_score + 0.3).clamp(0, 1)
    other_unet_de_mods, _, _, _ = compute_de_mods_stacked(net_sr, other_deg)
    y_other = traced(latent, t, enc, other_unet_de_mods)
    change = (y_other.float() - y_n.float()).abs().mean().item()
    print(f"\n[CTX-CHECK] de_mods input sensitivity: mean|delta|={change:.6f}")
    if change < 0.001:
        print("  ❌ FAILED: de_mods is NOT reaching the traced graph. Trace baked constants.")
        print("     Plan B attribute routing did NOT work on this SDK version.")
        print("     Next: try Plan C (diffusers forward kwarg patching)")
    else:
        print("  ✅ PASSED: de_mods flows through trace.  Plan B works.")

    diff = (y_n.float() - y_cpu.float()).abs()
    print(f"[acc] Neuron vs CPU (same de_mods): max|diff| {diff.max():.5f}  mean {diff.mean():.5f}")
    print("\nPHASE-3 PLAN B DONE.")


if __name__ == "__main__":
    main()
