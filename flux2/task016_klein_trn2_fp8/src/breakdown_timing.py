#!/usr/bin/env python3
"""Per-phase timing breakdown for FLUX.2-klein-base-9B on Neuron trn2.

Wraps the text encoder, DiT transformer, and VAE decode with timers, then
runs one warmup image plus measured seeds. Answers: of the ~41.8s per image,
how much is CPU text encoding, how much is DiT (and how many forwards), and
how much is VAE decode.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/mnt/nvme/flux2-klein/src")
from application import NeuronFlux2KleinApplication, create_flux2_klein_config

MODEL = "/mnt/nvme/flux2-klein/weights"
COMPILED = "/mnt/nvme/flux2-klein/compiled_bf16"
OUT_DIR = Path("/mnt/nvme/flux2-klein/outputs_breakdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = 50
SEEDS = [42, 43]
PROMPT = "A cat holding a sign that says hello world"

calls = {"text_encoder": [], "dit": [], "vae_decode": []}


def wrap(fn, bucket):
    def inner(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        calls[bucket].append(time.perf_counter() - t0)
        return result

    return inner


def main():
    config = create_flux2_klein_config(
        model_path=MODEL,
        backbone_tp_degree=4,
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
    )
    app = NeuronFlux2KleinApplication(
        model_path=MODEL,
        backbone_config=config,
        height=1024,
        width=1024,
    )
    app.compile(COMPILED)
    t0 = time.perf_counter()
    app.load(COMPILED)
    load_seconds = time.perf_counter() - t0

    pipe = app.pipe
    pipe.text_encoder.forward = wrap(pipe.text_encoder.forward, "text_encoder")
    pipe.transformer.forward = wrap(pipe.transformer.forward, "dit")
    pipe.vae.decode = wrap(pipe.vae.decode, "vae_decode")

    report = {
        "load_seconds": load_seconds,
        "steps": STEPS,
        "guidance_scale": 4.0,
        "images": [],
    }

    def run_one(seed, label):
        for values in calls.values():
            values.clear()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.perf_counter()
        result = app(
            prompt=PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=STEPS,
            guidance_scale=4.0,
            generator=generator,
        )
        total = time.perf_counter() - t0
        result.images[0].save(OUT_DIR / f"{label}_seed{seed}.png")
        te = sum(calls["text_encoder"])
        dit = sum(calls["dit"])
        vae = sum(calls["vae_decode"])
        entry = {
            "label": label,
            "seed": seed,
            "total_seconds": round(total, 3),
            "text_encoder_seconds": round(te, 3),
            "text_encoder_calls": len(calls["text_encoder"]),
            "dit_seconds": round(dit, 3),
            "dit_calls": len(calls["dit"]),
            "dit_mean_ms_per_call": round(1000 * dit / max(len(calls["dit"]), 1), 1),
            "vae_decode_seconds": round(vae, 3),
            "vae_decode_calls": len(calls["vae_decode"]),
            "other_seconds": round(total - te - dit - vae, 3),
        }
        report["images"].append(entry)
        print(json.dumps(entry), flush=True)

    run_one(52042, "warmup")
    for seed in SEEDS:
        run_one(seed, "measured")

    out_path = OUT_DIR / "breakdown.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE " + str(out_path), flush=True)


if __name__ == "__main__":
    main()
