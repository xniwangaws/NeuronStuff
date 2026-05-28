"""Validation with a realistic latent: encode an image first, then decode through chain.
Real latents have bounded distribution (~mean 0 std 1) post scaling, so accumulated
bf16 error is more representative of production workload."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL

import sys
sys.path.insert(0, "/home/ubuntu/work_a")
from chained_vae_decoder import ChainedVAEDecoder, DEFAULT_ORDER

VAE_PATH = "/home/ubuntu/sdxl-base"
WORK = Path("/home/ubuntu/work_a")


def cosine(a, b):
    a = a.flatten().to(torch.float32); b = b.flatten().to(torch.float32)
    # Normalize manually to avoid bf16 norm overflow.
    return float((a*b).sum() / (a.norm() * b.norm()))


def to_png(image, path):
    x = image.detach().to(torch.float32).clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)


def main():
    vae = AutoencoderKL.from_pretrained(
        VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
    )
    vae.eval()

    # Make a 2048x2048 "bus" image: simple synthetic scene.
    H = W = 2048
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = 0.6  # reddish background
    img[200:1800, 400:1600, :] = 0.9  # bus body
    img[400:600, 600:1400, :] = 0.2   # windows
    img[1700:1850, 600:800, :] = 0.1  # wheel
    img[1700:1850, 1200:1400, :] = 0.1
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil.save(WORK / "input_bus.png")

    # Encode to latent.
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16) * 2 - 1
    print("Encoding 2048x2048 image to latent on CPU bf16 (slow)...")
    t0 = time.time()
    with torch.no_grad():
        latent = vae.encode(img_t).latent_dist.mean * vae.config.scaling_factor
    print(f"  encode: {time.time()-t0:.1f}s  latent shape={list(latent.shape)} mean={latent.float().mean().item():.3f} std={latent.float().std().item():.3f}")

    # Inverse scaling for decode (matches diffusers pipeline).
    decode_in = latent / vae.config.scaling_factor

    # CPU reference.
    print("CPU bf16 reference decode...")
    t0 = time.time()
    with torch.no_grad():
        ref = vae.decode(decode_in).sample
    cpu_t = time.time() - t0
    print(f"  cpu decode: {cpu_t:.1f}s")

    # Neuron chain.
    chain = ChainedVAEDecoder(WORK)
    print("Neuron chained decode...")
    t0 = time.time()
    with torch.no_grad():
        out_cold = chain(decode_in)
    cold_t = time.time() - t0
    warm = []
    for i in range(5):
        t0 = time.time()
        with torch.no_grad():
            out = chain(decode_in)
        warm.append(time.time()-t0)
    print(f"  cold {cold_t:.2f}s  warm {warm}")

    cos = cosine(ref, out)
    md  = float((ref.float() - out.float()).abs().max())
    me  = float((ref.float() - out.float()).abs().mean())
    print(f"\n=== Realistic-latent accuracy ===")
    print(f"  cosine_similarity = {cos:.6f}  (target >= 0.999)")
    print(f"  max_abs_diff      = {md:.6f}")
    print(f"  mean_abs_diff     = {me:.6f}")

    to_png(ref, WORK / "real_ref_cpu.png")
    to_png(out, WORK / "real_neuron.png")

    summary = {
        "input_shape": list(decode_in.shape),
        "output_shape": list(out.shape),
        "cosine_similarity": cos,
        "max_abs_diff": md,
        "mean_abs_diff": me,
        "cpu_decode_seconds": cpu_t,
        "neuron_cold_seconds": cold_t,
        "neuron_warm_seconds": warm,
        "neuron_warm_mean_seconds": float(np.mean(warm)),
    }
    (WORK / "test_summary_realistic.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {WORK / 'test_summary_realistic.json'}")


if __name__ == "__main__":
    main()
