"""Phase R1 step 1: dump UNet state_dict keys + shapes + dtype.

Also dumps what S3Diff's target_modules_unet is (the LoRA sites) and what
unet_lora_layers looks like at runtime. All CPU, no Neuron.
"""
import argparse, pickle, sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--out_dir", default="/home/ubuntu/workspace/s3diff_nxdi/data")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import argparse as _ap
    s3args = _ap.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="x", neg_prompt="y",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    from s3diff_tile import S3Diff
    print("[load] S3Diff CPU fp32...", flush=True)
    net_sr = S3Diff(
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
        lora_rank_unet=32, lora_rank_vae=16, args=s3args,
    )
    net_sr.set_eval()

    # 1) UNet state_dict
    sd = net_sr.unet.state_dict()
    with open(out_dir / "unet_keys.txt", "w") as f:
        total = 0
        for k, v in sd.items():
            f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\t{v.numel()}\n")
            total += v.numel()
        f.write(f"# TOTAL params: {total:,}\n")
    print(f"[save] UNet state_dict keys -> {out_dir / 'unet_keys.txt'}  ({len(sd)} entries, {total:,} params)", flush=True)

    # 2) VAE state_dict (decoder only — encoder stays eager)
    vsd = net_sr.vae.decoder.state_dict()
    with open(out_dir / "vae_decoder_keys.txt", "w") as f:
        total = 0
        for k, v in vsd.items():
            f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\t{v.numel()}\n")
            total += v.numel()
        f.write(f"# TOTAL params: {total:,}\n")
    print(f"[save] VAE decoder keys -> {out_dir / 'vae_decoder_keys.txt'}  ({len(vsd)} entries, {total:,} params)", flush=True)

    # 3) LoRA target module lists
    unet_targets = net_sr.target_modules_unet if hasattr(net_sr, "target_modules_unet") else None
    vae_targets = net_sr.target_modules_vae if hasattr(net_sr, "target_modules_vae") else None
    with open(out_dir / "lora_targets.txt", "w") as f:
        f.write(f"target_modules_unet = {unet_targets!r}\n")
        f.write(f"target_modules_vae  = {vae_targets!r}\n")
        f.write(f"lora_rank_unet = {net_sr.lora_rank_unet}\n")
        f.write(f"lora_rank_vae  = {net_sr.lora_rank_vae}\n")

    # 4) Enumerate LoRA sites (unet_lora_layers attribute)
    if hasattr(net_sr, "unet_lora_layers"):
        unet_sites = list(net_sr.unet_lora_layers)
        print(f"[info] UNet LoRA sites: {len(unet_sites)}", flush=True)
        with open(out_dir / "unet_lora_sites.txt", "w") as f:
            for s in unet_sites:
                f.write(f"{s}\n")

    if hasattr(net_sr, "vae_lora_layers"):
        vae_sites = list(net_sr.vae_lora_layers)
        print(f"[info] VAE  LoRA sites: {len(vae_sites)}", flush=True)
        with open(out_dir / "vae_lora_sites.txt", "w") as f:
            for s in vae_sites:
                f.write(f"{s}\n")

    # 5) Count unique module types in LoRA sites
    from collections import Counter
    def summarize(sites):
        # Classify by last part of the name and second-to-last
        parts_last = Counter(s.split(".")[-1] for s in sites)
        return parts_last
    if hasattr(net_sr, "unet_lora_layers"):
        print("[summary] UNet LoRA by last name-part:", dict(summarize(list(net_sr.unet_lora_layers))), flush=True)
    if hasattr(net_sr, "vae_lora_layers"):
        print("[summary] VAE  LoRA by last name-part:", dict(summarize(list(net_sr.vae_lora_layers))), flush=True)

    # 6) UNet class hierarchy for modeling guidance
    with open(out_dir / "unet_class_summary.txt", "w") as f:
        from collections import Counter
        cls_count = Counter()
        for name, m in net_sr.unet.named_modules():
            cls_count[type(m).__name__] += 1
        for cls, n in cls_count.most_common():
            f.write(f"{cls}\t{n}\n")
    print(f"[save] UNet class summary -> {out_dir / 'unet_class_summary.txt'}", flush=True)

    # 7) pickle checkpoint peek
    with open(args.pretrained_path, "rb") as f:
        ckpt = pickle.load(f)
    with open(out_dir / "s3diff_pkl_keys.txt", "w") as f:
        for k, v in ckpt.items():
            if isinstance(v, dict):
                f.write(f"{k} (dict, {len(v)} entries)\n")
                for k2, v2 in list(v.items())[:3]:
                    if hasattr(v2, "shape"):
                        f.write(f"  {k2}\t{tuple(v2.shape)}\t{v2.dtype}\n")
                    else:
                        f.write(f"  {k2}\t{type(v2).__name__}\t{v2!r:.50}\n")
                if len(v) > 3:
                    f.write(f"  ... ({len(v) - 3} more)\n")
            elif hasattr(v, "shape"):
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
            else:
                f.write(f"{k}\t{type(v).__name__}\t{v!r:.100}\n")
    print(f"[save] s3diff.pkl key overview -> {out_dir / 's3diff_pkl_keys.txt'}", flush=True)

    print("[done] R1 step 1 — all dumps written.", flush=True)


if __name__ == "__main__":
    main()
