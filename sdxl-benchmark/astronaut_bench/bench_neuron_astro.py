"""SDXL Neuron astronaut benchmark — seeds 42-51, resolution parametric."""
import argparse, json, os, time
import torch, torch_neuronx
from diffusers import DiffusionPipeline
# Reuse verbatim wrappers from trace_sdxl_res.py (matching repo trace_sdxl.py)
from trace_sdxl_res import NeuronUNet, UNetWrap, TextEncoderOutputWrapper, TraceableTextEncoder


ap = argparse.ArgumentParser()
ap.add_argument("--compile_dir", required=True)
ap.add_argument("--model", default="/home/ubuntu/models/sdxl-base")
ap.add_argument("--out", required=True)
ap.add_argument("--resolution", type=int, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--unet_cores", default="0,1,2,3")
args = ap.parse_args()

PROMPT = "An astronaut riding a green horse"
STEPS = 50
GUIDANCE = 7.5
RES = args.resolution
os.makedirs(args.out, exist_ok=True)
device_ids = [int(x) for x in args.unet_cores.split(",")]

print(f"[load] SDXL Neuron res={RES}")
t0 = time.time()
pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
# Only swap UNet + VAE for Neuron NEFFs; keep CPU text encoders (CLIP-L/G tiny, not worth Neuron).
# This bypasses the pool-embed shape mismatch from TextEncoderOutputWrapper.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
pipe.unet.unetwrap = torch_neuronx.DataParallel(
    torch.jit.load(os.path.join(args.compile_dir, "unet/model.pt")),
    device_ids, set_dynamic_batching=False)
pipe.vae.decoder = torch.jit.load(os.path.join(args.compile_dir, "vae_decoder/model.pt"))
pipe.vae.post_quant_conv = torch.jit.load(os.path.join(args.compile_dir, "vae_post_quant_conv/model.pt"))
load_s = time.time() - t0
print(f"[load] {load_s:.1f}s")

print(f"[warmup] {args.warmup} rounds")
for i in range(args.warmup):
    t = time.time()
    _ = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
             guidance_scale=GUIDANCE,
             generator=torch.Generator("cpu").manual_seed(0)).images[0]
    print(f"  warmup {i+1}/{args.warmup}: {time.time()-t:.2f}s")

runs = []
for seed in args.seeds:
    t = time.time()
    img = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    e2e = time.time() - t
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    std_v = float(arr.std())
    png = os.path.join(args.out, f"seed{seed}_astro.png")
    img.save(png)
    runs.append({"seed": seed, "time_s": round(e2e, 2), "std": round(std_v, 2), "png": png})
    print(f"[seed {seed}] {e2e:.2f}s std={std_v:.2f}")

summary = {
    "device": f"Neuron trn2.3xlarge res={RES}",
    "model": "SDXL-base-1.0 BF16",
    "prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE,
    "resolution": f"{RES}x{RES}", "load_s": round(load_s, 1),
    "mean_s": round(sum(r["time_s"] for r in runs)/len(runs), 2),
    "runs": runs,
}
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s over {len(runs)} seeds")
