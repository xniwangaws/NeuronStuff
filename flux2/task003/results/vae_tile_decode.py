"""Tiled VAE decode on Neuron using the 512² NEFF for 1024² / 2048² images.

Replicates diffusers' AutoencoderKLFlux2.tiled_decode logic but replaces
the internal `post_quant_conv + decoder` call with the traced NEFF, which
already wraps both.

Tile params:
    tile_latent_min_size  = 64   (64x64 latent tile -> 512x512 pixel tile)
    tile_sample_min_size  = 512
    tile_overlap_factor   = 0.25
  => overlap_size   (latent) = 48
  => blend_extent   (pixel)  = 128
  => row_limit      (pixel)  = 384
"""
import time
import torch

TILE_LATENT = 64
TILE_SAMPLE = 512
OVERLAP     = 0.25

def blend_v(a, b, extent):
    extent = min(a.shape[2], b.shape[2], extent)
    for y in range(extent):
        b[:, :, y, :] = a[:, :, -extent + y, :] * (1 - y/extent) + b[:, :, y, :] * (y/extent)
    return b

def blend_h(a, b, extent):
    extent = min(a.shape[3], b.shape[3], extent)
    for x in range(extent):
        b[:, :, :, x] = a[:, :, :, -extent + x] * (1 - x/extent) + b[:, :, :, x] * (x/extent)
    return b

def tiled_decode_neff(decoder_callable, z, tile_latent=TILE_LATENT, tile_sample=TILE_SAMPLE, overlap=OVERLAP):
    """z: (B, 32, H, W) bf16 latent.  Returns (B, 3, H*8, W*8) image in pixel space [-1,1]."""
    overlap_size  = int(tile_latent  * (1 - overlap))        # 48
    blend_extent  = int(tile_sample  * overlap)              # 128
    row_limit     = tile_sample - blend_extent               # 384

    # pad the latent to a multiple of overlap_size starting points; we follow
    # diffusers logic that iterates over exact starts and lets the final tile
    # be smaller than TILE_LATENT (NEFF is fixed-size, so we must pad to 64).
    B, C, H, W = z.shape

    def decode_one(latent_64):
        """Pad to (B,32,64,64) if needed, run NEFF, crop to match."""
        _, _, h, w = latent_64.shape
        pad_h = tile_latent - h
        pad_w = tile_latent - w
        if pad_h > 0 or pad_w > 0:
            # replicate padding on the right/bottom to avoid zero-bleed seams
            latent_64 = torch.nn.functional.pad(latent_64, (0, pad_w, 0, pad_h), mode="replicate")
        out = decoder_callable(latent_64)
        if isinstance(out, (list, tuple)):
            out = out[0]
        # crop pixel output to matching size
        out = out[:, :, : (h * 8), : (w * 8)]
        return out

    rows = []
    for i in range(0, H, overlap_size):
        row = []
        for j in range(0, W, overlap_size):
            tile = z[:, :, i : i + tile_latent, j : j + tile_latent]
            decoded = decode_one(tile)
            row.append(decoded)
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(rows[i-1][j], tile, blend_extent)
            if j > 0:
                tile = blend_h(row[j-1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    dec = torch.cat(result_rows, dim=2)
    # Final crop to expected output
    return dec[:, :, : H*8, : W*8]


if __name__ == "__main__":
    import torch_neuronx
    NEFF = "/home/ubuntu/vae_traced/vae_decoder_512.pt"
    print(f"[load] {NEFF}")
    neff = torch.jit.load(NEFF)

    def neff_call(x):
        with torch.no_grad():
            return neff(x)

    # Warmup the NEFF
    print("[warmup]")
    _ = neff_call(torch.randn(1, 32, 64, 64, dtype=torch.bfloat16))

    # Also load CPU VAE for reference (tiled_decode on CPU for 1024)
    from diffusers import AutoencoderKLFlux2
    vae_cpu = AutoencoderKLFlux2.from_pretrained(
        "/home/ubuntu/flux2_weights", subfolder="vae", torch_dtype=torch.bfloat16
    ).eval()
    vae_cpu.tile_latent_min_size  = TILE_LATENT
    vae_cpu.tile_sample_min_size  = TILE_SAMPLE
    vae_cpu.tile_overlap_factor   = OVERLAP
    vae_cpu.use_tiling            = True

    import json
    report = {}

    for res in [512, 1024, 2048]:
        Hl = Wl = res // 8
        print(f"\n=== {res}x{res} (latent {Hl}x{Wl}) ===")
        torch.manual_seed(0)
        z = torch.randn(1, 32, Hl, Wl, dtype=torch.bfloat16)

        # Neuron tiled (or direct for 512)
        times = []
        if res == 512:
            # direct call
            for _ in range(3): neff_call(z)
            for _ in range(5):
                t = time.time(); img_n = neff_call(z); times.append(time.time()-t)
        else:
            for _ in range(2): tiled_decode_neff(neff_call, z)
            for _ in range(3):
                t = time.time(); img_n = tiled_decode_neff(neff_call, z); times.append(time.time()-t)
        mean_n = sum(times)/len(times)
        print(f"  neuron tiled: mean={mean_n*1000:.0f}ms, shape={tuple(img_n.shape)}")

        # CPU reference — only for 512/1024 (2048 would be ~25min on CPU)
        if res <= 1024:
            t = time.time()
            with torch.no_grad():
                if res == 512:
                    img_c = vae_cpu.decode(z, return_dict=False)[0]
                else:
                    img_c = vae_cpu.tiled_decode(z, return_dict=False)[0]
            cpu_time = time.time() - t
            af, bf = img_n.float().flatten(), img_c.float().flatten()
            cs = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
            mse = (af - bf).pow(2).mean().item()
            psnr = 10.0 * torch.log10(torch.tensor(4.0/mse)).item() if mse > 0 else float("inf")
            print(f"  cpu tiled   : {cpu_time:.2f}s  cos_sim={cs:.6f}  psnr={psnr:.2f}dB")
            report[str(res)] = {"neuron_ms": mean_n*1000, "cpu_s": cpu_time,
                                "cos_sim": cs, "psnr_db": psnr,
                                "shape": list(img_n.shape)}
        else:
            report[str(res)] = {"neuron_ms": mean_n*1000, "shape": list(img_n.shape)}

    with open("/home/ubuntu/vae_tile_benchmark.json","w") as f:
        json.dump(report, f, indent=2)
    print("\n[done] -> /home/ubuntu/vae_tile_benchmark.json")
    print(json.dumps(report, indent=2))
