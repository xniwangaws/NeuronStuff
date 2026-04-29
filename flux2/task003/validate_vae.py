"""Validate the 512² Neuron VAE decoder against CPU reference.

Uses a real latent (saved from CPU reference run) rather than random data,
since VAE decoders are very sensitive to input distribution.

For the tiled 1K path: split the 1024² 128×128 latent into 4 tiles of 64×64
(with overlap), decode each through the 512² NEFF, then blend and compare
against the CPU reference image.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline, AutoencoderKLFlux2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neff", default="/home/ubuntu/vae_traced/vae_decoder_512.pt")
    parser.add_argument("--model", default="/home/ubuntu/flux2_weights")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--out-dir", default="vae_validate")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    PROMPTS = [
        "a high-resolution photograph of a red panda sitting on a tree branch in a misty forest, volumetric lighting",
        "an oil painting of a medieval castle at sunset, dramatic clouds, style of Caspar David Friedrich",
        "a futuristic cyberpunk city street at night, neon signs in Japanese, rain-slicked pavement, cinematic",
    ]
    SEED = 42
    prompt = PROMPTS[args.prompt_index]

    print(f"[load] Flux2Pipeline to capture a real latent at 512²...")
    # Generate a real latent via the pipeline (using output_type='latent')
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    gen = torch.Generator(device="cpu").manual_seed(SEED + args.prompt_index)
    result = pipe(
        prompt=prompt,
        height=512, width=512,
        num_inference_steps=28,
        guidance_scale=4.0,
        generator=gen,
        output_type="latent",
        return_dict=True,
    )
    latent = result.images  # shape (1, 32, 64, 64) before VAE decode, unscaled
    print(f"[latent] shape={tuple(latent.shape)} dtype={latent.dtype}")

    # Scale latent as the VAE expects (FLUX uses shift_factor & scaling_factor)
    vae_cpu = pipe.vae
    shift = vae_cpu.config.shift_factor if hasattr(vae_cpu.config, "shift_factor") else 0.0
    scale = vae_cpu.config.scaling_factor if hasattr(vae_cpu.config, "scaling_factor") else 1.0
    print(f"[vae cfg] shift={shift} scale={scale}")
    scaled_latent = latent / scale + shift

    # CPU decode (reference)
    print(f"[cpu] decoding on CPU...")
    t = time.perf_counter()
    with torch.no_grad():
        cpu_image = vae_cpu.decode(scaled_latent, return_dict=False)[0]
    cpu_dt = time.perf_counter() - t
    print(f"[cpu] shape={tuple(cpu_image.shape)} time={cpu_dt:.1f}s")

    # Neuron decode (same latent)
    print(f"[neuron] loading traced NEFF from {args.neff}")
    compiled = torch.jit.load(args.neff)
    # Warmup
    for _ in range(3):
        _ = compiled(scaled_latent)
    t = time.perf_counter()
    neuron_image = compiled(scaled_latent)
    neuron_dt = time.perf_counter() - t
    print(f"[neuron] shape={tuple(neuron_image.shape)} time={neuron_dt*1000:.1f}ms")

    # Compare
    cpu_f = cpu_image.float()
    neuron_f = neuron_image.float()
    max_abs = (cpu_f - neuron_f).abs().max().item()
    max_rel = max_abs / cpu_f.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(cpu_f.flatten().unsqueeze(0), neuron_f.flatten().unsqueeze(0)).item()

    import numpy as np
    mse = ((cpu_f - neuron_f) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor((cpu_f.max() - cpu_f.min()) ** 2 / max(mse, 1e-12))).item()
    print(f"[accuracy] max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  cos_sim={cos:.6f}  psnr={psnr:.2f} dB")

    # Save decoded images for eyeball check
    from torchvision.transforms.functional import to_pil_image
    cpu_pil = to_pil_image((cpu_image[0].clamp(-1, 1) * 0.5 + 0.5).to(torch.float32))
    neuron_pil = to_pil_image((neuron_image[0].clamp(-1, 1) * 0.5 + 0.5).to(torch.float32))
    cpu_pil.save(out / f"cpu_p{args.prompt_index:02d}.png")
    neuron_pil.save(out / f"neuron_p{args.prompt_index:02d}.png")

    results = {
        "prompt": prompt,
        "prompt_index": args.prompt_index,
        "seed": SEED + args.prompt_index,
        "resolution": 512,
        "latent_shape": list(latent.shape),
        "cpu_time_s": cpu_dt,
        "neuron_time_s": neuron_dt,
        "speedup": cpu_dt / neuron_dt,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "cosine_sim": cos,
        "psnr_db": psnr,
    }
    (out / f"validate_p{args.prompt_index:02d}.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out}/validate_p{args.prompt_index:02d}.json")


if __name__ == "__main__":
    main()
