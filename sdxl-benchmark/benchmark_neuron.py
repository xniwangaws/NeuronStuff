"""
Load the 5 compiled NEFF-wrapped TorchScript modules, wire them back into a
DiffusionPipeline, and run the 5-prompt benchmark (1024x1024, BF16, seed=0,
warmup=5).

Run (after trace_sdxl.py):
    python benchmark_neuron.py \
        --compile_dir /home/ubuntu/sdxl/compile_dir \
        --model /home/ubuntu/models/sdxl-base \
        --prompts prompts.json \
        --steps 25 \
        --out /home/ubuntu/sdxl_out_25
"""

import argparse
import csv
import json
import os
import time

import torch
import torch_neuronx
from diffusers import DiffusionPipeline

# Reuse wrappers from trace_sdxl.py (same dir)
from trace_sdxl import NeuronUNet, UNetWrap, TextEncoderOutputWrapper, TraceableTextEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compile_dir", required=True)
    p.add_argument("--model", required=True, help="SDXL weights dir")
    p.add_argument("--prompts", required=True)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--unet_cores",
        default=os.environ.get("SDXL_UNET_CORES", "0,1,2,3"),
        help="comma-separated logical core ids for UNet DataParallel",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device_ids = [int(x) for x in args.unet_cores.split(",")]

    with open(args.prompts) as f:
        pdata = json.load(f)
    prompts = pdata["prompts"]
    seed = pdata.get("seed", 0)

    # ---- pipeline load ---------------------------------------------------
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    unet_pt = os.path.join(args.compile_dir, "unet", "model.pt")
    pipe.unet.unetwrap = torch_neuronx.DataParallel(
        torch.jit.load(unet_pt), device_ids, set_dynamic_batching=False
    )

    pipe.vae.decoder = torch.jit.load(
        os.path.join(args.compile_dir, "vae_decoder", "model.pt")
    )
    pipe.vae.post_quant_conv = torch.jit.load(
        os.path.join(args.compile_dir, "vae_post_quant_conv", "model.pt")
    )
    pipe.text_encoder = TextEncoderOutputWrapper(
        torch.jit.load(os.path.join(args.compile_dir, "text_encoder", "model.pt")),
        pipe.text_encoder,
    )
    pipe.text_encoder_2 = TextEncoderOutputWrapper(
        torch.jit.load(os.path.join(args.compile_dir, "text_encoder_2", "model.pt")),
        pipe.text_encoder_2,
    )
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s")

    # ---- warmup ----------------------------------------------------------
    for i in range(args.warmup):
        t = time.time()
        _ = pipe(
            prompts[0],
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        print(f"[warmup {i + 1}/{args.warmup}] {time.time() - t:.2f}s")

    # ---- timed -----------------------------------------------------------
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "precision", "prompt_id", "steps", "e2e_s", "load_s"])
        for k, prompt in enumerate(prompts):
            t = time.time()
            image = pipe(
                prompt,
                num_inference_steps=args.steps,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            e2e_s = time.time() - t
            png = os.path.join(args.out, f"prompt{k}_steps{args.steps}.png")
            image.save(png)
            w.writerow(["trn2", "bf16", k, args.steps, f"{e2e_s:.3f}", f"{load_s:.3f}"])
            print(f"[run prompt{k}] {e2e_s:.2f}s -> {png}")

    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()
