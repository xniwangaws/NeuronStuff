import torch, time, json, os
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
import numpy as np

PIPE_PATH = "/home/ubuntu/flux2_klein"  # BF16 pipeline (for text encoder, vae, scheduler, tokenizer)
FP8_CKPT = "/home/ubuntu/flux2_klein_fp8/flux-2-klein-9b-fp8.safetensors"
PROMPT = "A cat holding a sign that says hello world"
STEPS = 50
GUIDANCE = 4.0
SEEDS = list(range(42, 52))

# Load FP8 transformer via single-file
print("Loading FP8 transformer from single-file...", flush=True)
transformer = Flux2Transformer2DModel.from_single_file(
    FP8_CKPT,
    config="/home/ubuntu/flux2_klein/transformer",
    torch_dtype=torch.bfloat16,
)
print(f"Transformer loaded. dtype={transformer.dtype}", flush=True)

# Load pipeline with replaced transformer
print("Loading full pipeline with FP8 transformer...", flush=True)
pipe = Flux2KleinPipeline.from_pretrained(PIPE_PATH, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("Pipeline on CUDA", flush=True)

for RES in [1024, 2048]:
    OUT = f"/home/ubuntu/klein_fp8_l4_{RES}"
    os.makedirs(OUT, exist_ok=True)
    if RES >= 2048:
        pipe.enable_vae_tiling()
    # warmup
    print(f"[{RES}] warmup", flush=True)
    pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
         generator=torch.Generator("cpu").manual_seed(0)).images[0]
    runs = []
    peak = 0
    for seed in SEEDS:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        dt = time.time()-t0
        pk = torch.cuda.max_memory_allocated()/1024**3
        peak = max(peak, pk)
        png = f"{OUT}/seed{seed}_cat.png"
        img.save(png)
        arr = np.asarray(img).astype(np.float32)
        runs.append({"seed": seed, "time_s": round(dt,2), "peak_vram_gb": round(pk,2), "std": round(float(arr.std()),2), "png": png})
        print(f"[{RES} seed {seed}] {dt:.2f}s peak={pk:.2f}GB std={arr.std():.2f}", flush=True)
    mean_s = sum(r["time_s"] for r in runs)/len(runs)
    summary = {"device":"L4 g6.4xlarge (FP8 official)","model":"FLUX.2-klein-9b-fp8",
               "resolution":f"{RES}x{RES}","prompt":PROMPT,"steps":STEPS,"guidance":GUIDANCE,
               "mean_s":round(mean_s,2),"peak_vram_gb":round(peak,2),"n_ok":len(runs),"runs":runs}
    json.dump(summary, open(f"{OUT}/results.json","w"), indent=2)
    print(f"[{RES} DONE] mean={mean_s:.2f}s peak={peak:.2f}GB", flush=True)
