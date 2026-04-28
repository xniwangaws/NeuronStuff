"""Validate the tiled Neuron VAE decoder at 2048² using a REAL pipeline latent.

The prior benchmark used random latents and got cos_sim ~0.92 (BF16 noise amplification).
A real-signal latent should give much higher cos_sim, matching 512² (0.998).

Strategy:
1. Run Flux2Pipeline CPU at 2048² with output_type='latent' (slow, ~20-30 min on 192 cores)
2. Unpack + BN denorm → 4D (1, 32, 256, 256)
3. Decode on CPU (reference image)
4. Decode via tiled Neuron orchestrator (6x6 tiles of 512² NEFF)
5. Compare: cos_sim, max_rel_err, PSNR
"""
import os
import sys
import time
import json
import torch

sys.path.insert(0, "/home/ubuntu")
from vae_tile_decode import tiled_decode_neff, TILE_LATENT

WEIGHTS = "/home/ubuntu/flux2_weights"
NEFF = "/home/ubuntu/vae_traced/vae_decoder_512.pt"
OUT = "/home/ubuntu/vae_1k_validation.json"


def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()


def cos_max(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cs = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    denom = bf.abs().clamp(min=1e-6)
    mre = ((af - bf).abs() / denom).max().item()
    return cs, mre


def unpack_and_denorm(pipe, latents_packed, latent_ids):
    latents = pipe._unpack_latents_with_ids(latents_packed, latent_ids)
    vae = pipe.vae
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * bn_std + bn_mean
    latents = pipe._unpatchify_latents(latents)
    return latents


def main():
    from diffusers import Flux2Pipeline
    import torch_neuronx  # register Neuron model class

    # Validate at 1024² (real pipeline latent; ~15-20 min CPU denoise)
    # rather than 2048² which would take >90 min on CPU.
    # The tiling logic is exercised at 1024² (3x3 tiles of 512² NEFF).
    RES = 1024
    STEPS = 28
    GUIDE = 4.0
    PROMPT = "a high-resolution photograph of a red panda sitting on a tree branch in a misty forest, volumetric lighting"

    print(f"[load] Flux2Pipeline (BF16 on CPU) ...")
    t0 = time.time()
    pipe = Flux2Pipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
    pipe.to("cpu")
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"[gen] pipeline at {RES}² on CPU — THIS IS SLOW (~20-30 min) ...")
    gen = torch.Generator("cpu").manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        out = pipe(
            prompt=PROMPT,
            height=RES,
            width=RES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDE,
            generator=gen,
            output_type="latent",
        )
    gen_time = time.time() - t0
    packed = out.images if hasattr(out, "images") else out[0]
    print(f"  gen done in {gen_time:.1f}s; packed shape={tuple(packed.shape)} dtype={packed.dtype}")

    vae_sf = pipe.vae_scale_factor
    h = 2 * (RES // (vae_sf * 2))
    w = 2 * (RES // (vae_sf * 2))
    dummy = torch.zeros(1, pipe.vae.config.latent_channels * 4, h // 2, w // 2, dtype=packed.dtype)
    latent_ids = pipe._prepare_latent_ids(dummy)
    latents_4d = unpack_and_denorm(pipe, packed, latent_ids)
    print(f"  unpacked 4D latent: {tuple(latents_4d.shape)}")

    # Save the latent so we can reuse without re-running the pipeline
    torch.save(latents_4d, f"/home/ubuntu/flux2_{RES}_latent.pt")
    print(f"  saved latent to /home/ubuntu/flux2_{RES}_latent.pt")

    # Free pipeline memory (keep vae)
    vae_cpu = pipe.vae.eval()
    del pipe
    import gc
    gc.collect()

    # ---- CPU reference decode (tiled on CPU) ----
    print(f"[cpu] decoding {RES}² (tiled) on CPU — also slow ...")
    vae_cpu.tile_latent_min_size = TILE_LATENT
    vae_cpu.tile_sample_min_size = 512
    vae_cpu.tile_overlap_factor = 0.25
    vae_cpu.use_tiling = True
    t0 = time.time()
    with torch.no_grad():
        img_cpu = vae_cpu.tiled_decode(latents_4d, return_dict=False)[0]
    cpu_time = time.time() - t0
    print(f"  cpu tiled decode: {cpu_time:.1f}s, shape {tuple(img_cpu.shape)}")

    del vae_cpu
    gc.collect()

    # ---- Neuron tiled decode ----
    print(f"[neuron] loading NEFF and tiled decoding ...")
    neff = torch.jit.load(NEFF)

    def neff_call(x):
        with torch.no_grad():
            return neff(x)

    _ = neff_call(torch.randn(1, 32, 64, 64, dtype=torch.bfloat16))  # warmup
    lat_n = latents_4d.to(torch.bfloat16)
    t0 = time.time()
    img_neu = tiled_decode_neff(neff_call, lat_n)
    neu_time = time.time() - t0
    print(f"  neuron tiled: {neu_time:.2f}s, shape {tuple(img_neu.shape)}")

    cs, mre = cos_max(img_neu, img_cpu)
    p = psnr(img_neu.clamp(-1, 1), img_cpu.clamp(-1, 1))
    print(f"\n=== {RES}² VAE VALIDATION (real pipeline latent) ===")
    print(f"cos_sim     : {cs:.6f}")
    print(f"max_rel_err : {mre:.4f}")
    print(f"PSNR        : {p:.2f} dB")
    print(f"cpu time    : {cpu_time:.2f}s")
    print(f"neuron time : {neu_time*1000:.0f}ms")

    # Save image artifacts
    from torchvision.transforms.functional import to_pil_image

    def save_img(img, path):
        # img is (B, 3, H, W) in [-1, 1]. Map to [0, 1] PIL.
        pil = to_pil_image((img[0].clamp(-1, 1) * 0.5 + 0.5).float())
        pil.save(path)

    save_img(img_cpu, f"/home/ubuntu/vae_{RES}_cpu.png")
    save_img(img_neu, f"/home/ubuntu/vae_{RES}_neuron.png")

    report = {
        "resolution": RES,
        "prompt": PROMPT,
        "seed": 42,
        "steps": STEPS,
        "guidance": GUIDE,
        "pipeline_cpu_gen_time_s": gen_time,
        "vae_cpu_tiled_time_s": cpu_time,
        "vae_neuron_tiled_time_s": neu_time,
        "cos_sim": cs,
        "max_rel_err": mre,
        "psnr_db": p,
        "latents_4d_shape": list(latents_4d.shape),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
