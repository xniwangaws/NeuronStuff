"""Single-run smoke test — task011 pipeline + 5/4 NEFFs + seed 42 cat prompt on DLAMI 20260410."""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch_neuronx  # noqa

sys.path.insert(0, "/home/ubuntu")
from neuron_flux2_pipeline import NeuronFlux2Pipeline

CAT_PROMPT = "A cat holding a sign that says hello world"
SEED = 42

def main():
    dit_neff = "/mnt/nvme/neff/dit_tp8_1024_v3.pt"
    te_neff = "/mnt/nvme/neff/text_encoder_tp8.pt"
    vae_neff = "/mnt/nvme/neff/vae_decoder_512.pt"
    weights_dir = "/home/ubuntu/flux2_weights"
    out_png = "/tmp/smoke_v3.png"

    print(f"[load] DiT={dit_neff}")
    t0 = time.perf_counter()
    pipe = NeuronFlux2Pipeline.from_traced(
        dit_neff=dit_neff, te_neff=te_neff, vae_neff=vae_neff,
        weights_dir=weights_dir)
    load_time = time.perf_counter() - t0
    print(f"[load] {load_time:.2f}s")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    t = time.perf_counter()
    image = pipe(prompt=CAT_PROMPT, height=1024, width=1024,
                 num_inference_steps=50, guidance_scale=4.0, generator=gen)
    gen_time = time.perf_counter() - t
    image.save(out_png)
    arr = np.asarray(image).astype(np.float32)
    print(f"[run] gen_time={gen_time:.2f}s")
    print(f"[image] shape={arr.shape} dtype={arr.dtype} min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f} std={arr.std():.2f}")
    print(f"[image] saved -> {out_png}")
    import hashlib
    md5 = hashlib.md5(open(out_png, "rb").read()).hexdigest()
    print(f"[image] md5={md5[:8]}")
    verdict = "IMAGE" if arr.std() > 50 else "NOISE"
    print(f"[verdict] {verdict} (std={arr.std():.2f})")

if __name__ == "__main__":
    main()
