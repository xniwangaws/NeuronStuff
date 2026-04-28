"""
Trace S3Diff VAE encoder + decoder on Neuron.

The S3Diff VAE is `stabilityai/sd-turbo`'s AutoencoderKL (~84M params)
with S3Diff's LoRA applied to encoder only (per s3diff_tile.py vae_lora_layers).

Strategy:
  - Encoder: trace at fixed pixel tile 256x256 -> latent 32x32
  - Decoder: trace at fixed latent tile 224x224 -> pixel 1792x1792 (S3Diff default)
    OR smaller (e.g. 128x128 latent -> 1024x1024 pixel) to stay within compile RAM

Saves both as TorchScript archives for use in neuron_e2e.
"""
import argparse
import os
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/bridge_LQ_256.png")
    p.add_argument("--out_dir", default="/home/ubuntu/s3diff/neuron_out")
    p.add_argument("--component", choices=["encoder", "decoder", "both"], default="both")
    p.add_argument("--enc_tile", type=int, default=256, help="Encoder pixel tile side")
    p.add_argument("--dec_tile", type=int, default=96, help="Decoder latent tile side (96 matches UNet tile)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    import numpy as np
    from PIL import Image
    import importlib.util
    spec = importlib.util.spec_from_file_location("bake", "/home/ubuntu/s3diff/neuron_unet_trace_v2.py")
    bake_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(bake_mod)

    print("[load] S3Diff...", flush=True)
    s3 = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="x", neg_prompt="y",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    net_sr = S3Diff(
        lora_rank_unet=32, lora_rank_vae=16,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()

    # Bake VAE de_mod using DEResNet on the LQ image
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()
    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        deg_score = net_de(im_t.unsqueeze(0))
    bake_mod.compute_and_bake_de_mod(net_sr, deg_score)
    print(f"[bake] deg_score baked into VAE/UNet LoRA", flush=True)

    import torch_neuronx

    # ---------------- Encoder ----------------
    if args.component in ("encoder", "both"):
        print(f"\n=== Encoder trace (pixel tile {args.enc_tile}x{args.enc_tile}) ===", flush=True)

        # Encoder takes pixel [-1,1] input, returns DiagonalGaussianDistribution.
        # For trace we need a tensor-returning wrapper: just call encoder conv stack
        # and return sample(). The encoder ends with a "quant_conv" producing mean/var.
        # S3Diff replaces vae.encoder.forward with a tile-hook __call__ that calls
        # net.to(device) at runtime (trace-incompatible).  The real forward is
        # stored as vae.encoder.original_forward.  We wrap that directly.
        encoder_real_fwd = net_sr.vae.encoder.original_forward if hasattr(net_sr.vae.encoder, "original_forward") else net_sr.vae.encoder.forward

        class VAEEncoderWrapper(torch.nn.Module):
            def __init__(self, encoder, quant_conv, scaling):
                super().__init__()
                self.encoder = encoder
                self.quant_conv = quant_conv
                self.scaling = scaling
            def forward(self, x):
                # Directly call the un-hooked forward via Module.__call__ convention:
                # encoder module's forward is nn.Module.forward which = original conv stack
                h = self.encoder._orig_forward(x) if hasattr(self.encoder, "_orig_forward") else self.encoder.forward(x)
                moments = self.quant_conv(h)
                mean = moments.chunk(2, dim=1)[0]
                return mean * self.scaling

        # Inject _orig_forward reference so wrapper can call it even if forward is patched
        if hasattr(net_sr.vae.encoder, "original_forward"):
            net_sr.vae.encoder._orig_forward = net_sr.vae.encoder.original_forward

        enc = VAEEncoderWrapper(net_sr.vae.encoder, net_sr.vae.quant_conv, net_sr.vae.config.scaling_factor).eval()
        x_ex = torch.randn(1, 3, args.enc_tile, args.enc_tile)

        # CPU eager reference
        with torch.no_grad():
            y_cpu = enc(x_ex)
        print(f"[enc-cpu] in {tuple(x_ex.shape)} -> out {tuple(y_cpu.shape)}", flush=True)

        t0 = time.perf_counter()
        enc_traced = torch_neuronx.trace(
            enc, (x_ex,), compiler_args=["--auto-cast", "none"],
        )
        print(f"[enc-neuron] compile {time.perf_counter()-t0:.1f}s", flush=True)

        save_path = os.path.join(args.out_dir, f"vae_encoder_{args.enc_tile}.pt2")
        torch.jit.save(enc_traced, save_path)
        print(f"[enc-save] {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)", flush=True)

        print("[enc-neuron] timing 3 runs...", flush=True)
        for i in range(3):
            t0 = time.perf_counter()
            y_n = enc_traced(x_ex)
            print(f"  run {i+1}: {time.perf_counter()-t0:.3f}s", flush=True)

        diff = (y_n.float() - y_cpu.float()).abs()
        print(f"[enc-acc] max|diff| {diff.max():.5f}  mean {diff.mean():.5f}", flush=True)

    # ---------------- Decoder ----------------
    if args.component in ("decoder", "both"):
        print(f"\n=== Decoder trace (latent tile {args.dec_tile}x{args.dec_tile}) ===", flush=True)

        if hasattr(net_sr.vae.decoder, "original_forward"):
            net_sr.vae.decoder._orig_forward = net_sr.vae.decoder.original_forward

        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, decoder, post_quant_conv, scaling):
                super().__init__()
                self.decoder = decoder
                self.post_quant_conv = post_quant_conv
                self.scaling = scaling
            def forward(self, z):
                z = z / self.scaling
                z = self.post_quant_conv(z)
                return self.decoder._orig_forward(z) if hasattr(self.decoder, "_orig_forward") else self.decoder.forward(z)

        dec = VAEDecoderWrapper(net_sr.vae.decoder, net_sr.vae.post_quant_conv, net_sr.vae.config.scaling_factor).eval()
        z_ex = torch.randn(1, 4, args.dec_tile, args.dec_tile)

        with torch.no_grad():
            y_cpu = dec(z_ex)
        print(f"[dec-cpu] in {tuple(z_ex.shape)} -> out {tuple(y_cpu.shape)}", flush=True)

        t0 = time.perf_counter()
        dec_traced = torch_neuronx.trace(
            dec, (z_ex,), compiler_args=["--auto-cast", "none"],
        )
        print(f"[dec-neuron] compile {time.perf_counter()-t0:.1f}s", flush=True)

        save_path = os.path.join(args.out_dir, f"vae_decoder_{args.dec_tile}.pt2")
        torch.jit.save(dec_traced, save_path)
        print(f"[dec-save] {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)", flush=True)

        print("[dec-neuron] timing 3 runs...", flush=True)
        for i in range(3):
            t0 = time.perf_counter()
            y_n = dec_traced(z_ex)
            print(f"  run {i+1}: {time.perf_counter()-t0:.3f}s", flush=True)

        diff = (y_n.float() - y_cpu.float()).abs()
        print(f"[dec-acc] max|diff| {diff.max():.5f}  mean {diff.mean():.5f}", flush=True)

    print("\nPHASE-2 (VAE trace) DONE.")


if __name__ == "__main__":
    main()
