"""Phase R1 step 3: extract UNet + VAE decoder state_dicts from loaded S3Diff.

CPU-only. Saves:
  unet_sd.pt     — raw diffusers + peft LoRA state_dict
  vae_dec_sd.pt  — raw VAE decoder state_dict (LoRA applied via peft same way)
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--out_dir", default="/home/ubuntu/workspace/s3diff_nxdi/data")
    args = p.parse_args()

    import argparse as _ap
    s3args = _ap.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32, pos_prompt="x", neg_prompt="y",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    from s3diff_tile import S3Diff
    print("[load] S3Diff...", flush=True)
    net_sr = S3Diff(
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
        lora_rank_unet=32, lora_rank_vae=16, args=s3args,
    )
    net_sr.set_eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_sd = net_sr.unet.state_dict()
    torch.save(unet_sd, out_dir / "unet_sd.pt")
    print(f"[save] UNet state_dict: {len(unet_sd)} keys -> {out_dir / 'unet_sd.pt'}", flush=True)

    vae_dec_sd = net_sr.vae.decoder.state_dict()
    torch.save(vae_dec_sd, out_dir / "vae_dec_sd.pt")
    print(f"[save] VAE decoder state_dict: {len(vae_dec_sd)} keys -> {out_dir / 'vae_dec_sd.pt'}", flush=True)

    # Also save the post_quant_conv (VAE decoder front-end)
    pqc_sd = net_sr.vae.post_quant_conv.state_dict()
    torch.save(pqc_sd, out_dir / "vae_post_quant_conv_sd.pt")
    print(f"[save] post_quant_conv state_dict: {len(pqc_sd)} keys -> {out_dir / 'vae_post_quant_conv_sd.pt'}", flush=True)


if __name__ == "__main__":
    main()
