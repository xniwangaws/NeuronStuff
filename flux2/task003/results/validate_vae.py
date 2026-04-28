"""Validate traced 512² VAE decoder against CPU reference."""
import os, time, json
import torch
import torch_neuronx  # registers __torch__.torch.classes.neuron.Model
from diffusers import Flux2Pipeline, AutoencoderKLFlux2

WEIGHTS = "/home/ubuntu/flux2_weights"
TRACE   = "/home/ubuntu/vae_traced/vae_decoder_512.pt"
PROMPTS = [
    "a cat holding a sign that says hello world",
    "a photorealistic portrait of an astronaut riding a horse on mars",
    "a cozy cabin in the snowy mountains at sunrise, oil painting",
]
SEED, STEPS, GUIDE, H, W = 42, 28, 4.0, 512, 512

def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse == 0: return float("inf")
    return 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()  # range [-1,1] => span 2 => 2^2=4

def cos_max(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cs = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    denom = bf.abs().clamp(min=1e-6)
    mre = ((af - bf).abs() / denom).max().item()
    return cs, mre

def unpack_and_denorm(pipe, latents_packed, latent_ids):
    """Replicates pipeline logic: scatter -> bn denorm -> unpatchify. Returns 4D (B,32,H,W)."""
    latents = pipe._unpack_latents_with_ids(latents_packed, latent_ids)
    vae = pipe.vae
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std  = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                         ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    latents = pipe._unpatchify_latents(latents)
    return latents

def main():
    print("[load] pipeline (bf16) on CPU ...")
    pipe = Flux2Pipeline.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16)
    pipe.to("cpu")

    print("[gen] packed latents via pipeline(output_type='latent') ...")
    gen = torch.Generator("cpu").manual_seed(SEED)
    with torch.no_grad():
        out = pipe(
            prompt=PROMPTS[0], height=H, width=W,
            num_inference_steps=STEPS, guidance_scale=GUIDE,
            generator=gen, output_type="latent",
        )
    packed = out.images if hasattr(out, "images") else out[0]
    print(f"  packed shape: {tuple(packed.shape)}, dtype: {packed.dtype}")

    # Reconstruct latent_ids for this shape (B=1, single-frame, H/2 x W/2 post-patchify)
    # Pipeline stored them inside, but we re-derive:
    vae_sf = pipe.vae_scale_factor  # 8
    h = 2 * (H // (vae_sf * 2))  # 64 for 512
    w = 2 * (W // (vae_sf * 2))
    # Build a dummy 4D latent to call _prepare_latent_ids
    dummy = torch.zeros(1, pipe.vae.config.latent_channels * 4, h // 2, w // 2,
                        dtype=packed.dtype)
    latent_ids = pipe._prepare_latent_ids(dummy)
    print(f"  latent_ids shape: {tuple(latent_ids.shape)}")

    latents_4d = unpack_and_denorm(pipe, packed, latent_ids)
    print(f"  unpacked 4D latent: {tuple(latents_4d.shape)}, dtype: {latents_4d.dtype}")

    # ---- CPU ground truth ----
    print("[cpu] decoding (slow) ...")
    pipe.vae.eval()
    t0 = time.time()
    with torch.no_grad():
        img_cpu = pipe.vae.decode(latents_4d, return_dict=False)[0]
    cpu_time = time.time() - t0
    print(f"  cpu decode: {cpu_time:.2f}s, out {tuple(img_cpu.shape)} {img_cpu.dtype}")

    # ---- Neuron traced ----
    print(f"[neuron] loading {TRACE} ...")
    neuron_vae = torch.jit.load(TRACE)
    # bf16 inputs (tracing was bf16)
    lat_n = latents_4d.to(torch.bfloat16)
    print("[neuron] warmup + decode ...")
    with torch.no_grad():
        _ = neuron_vae(lat_n)  # warmup
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    N = 5
    for _ in range(N):
        with torch.no_grad():
            img_neu = neuron_vae(lat_n)
    neu_time = (time.time() - t0) / N
    if isinstance(img_neu, (list, tuple)):
        img_neu = img_neu[0]
    print(f"  neuron decode: {neu_time*1000:.1f}ms, out {tuple(img_neu.shape)} {img_neu.dtype}")

    cs, mre = cos_max(img_neu, img_cpu)
    p = psnr(img_neu.clamp(-1,1), img_cpu.clamp(-1,1))
    print(f"\n=== 512² VAE VALIDATION ===")
    print(f"cos_sim     : {cs:.6f}")
    print(f"max_rel_err : {mre:.4f}")
    print(f"PSNR        : {p:.2f} dB")
    print(f"cpu_time    : {cpu_time:.2f}s")
    print(f"neu_time    : {neu_time*1000:.1f}ms  (speedup {cpu_time/neu_time:.1f}x)")

    # save report
    with open("/home/ubuntu/vae_validation_report.json","w") as f:
        json.dump({"cos_sim":cs,"max_rel_err":mre,"psnr_db":p,
                   "cpu_time_s":cpu_time,"neuron_time_s":neu_time,
                   "packed_shape":list(packed.shape),
                   "latents_4d_shape":list(latents_4d.shape)}, f, indent=2)
    print("report -> /home/ubuntu/vae_validation_report.json")

if __name__ == "__main__":
    main()
