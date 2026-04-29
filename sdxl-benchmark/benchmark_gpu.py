"""
SDXL GPU benchmark (H100 or L4), BF16, diffusers native.

Run:
    python benchmark_gpu.py \
        --device cuda:0 --dtype bf16 \
        --model /home/ubuntu/models/sdxl-base \
        --prompts prompts.json \
        --steps 25 \
        --out /home/ubuntu/sdxl_out_25 \
        --device_label h100     # or l4
"""

import argparse
import csv
import json
import os
import subprocess
import time

import torch
from diffusers import DiffusionPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device_label", required=True, help="e.g. h100 or l4")
    return p.parse_args()


def nvsmi_peak_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return float(out) / 1024
    except Exception:
        return -1.0


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    with open(args.prompts) as f:
        pdata = json.load(f)
    prompts = pdata["prompts"]
    seed = pdata.get("seed", 0)

    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(args.device)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s on {args.device}")

    for i in range(args.warmup):
        t = time.time()
        _ = pipe(
            prompts[0],
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        print(f"[warmup {i + 1}/{args.warmup}] {time.time() - t:.2f}s")

    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "precision", "prompt_id", "steps", "e2e_s", "load_s", "peak_mem_gb"])
        for k, prompt in enumerate(prompts):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            image = pipe(
                prompt,
                num_inference_steps=args.steps,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            e2e_s = time.time() - t
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else nvsmi_peak_gb()
            )
            png = os.path.join(args.out, f"prompt{k}_steps{args.steps}.png")
            image.save(png)
            w.writerow(
                [args.device_label, args.dtype, k, args.steps,
                 f"{e2e_s:.3f}", f"{load_s:.3f}", f"{peak_gb:.2f}"]
            )
            print(f"[run prompt{k}] {e2e_s:.2f}s peak {peak_gb:.2f}GB -> {png}")

    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()
