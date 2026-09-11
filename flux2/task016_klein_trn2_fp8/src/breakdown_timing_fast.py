#!/usr/bin/env python3
"""FLUX.2-klein on trn2 with all three stages on Neuron.

Baseline breakdown (measured 2026-09-10): total 41.4s =
  text encoder (CPU)  ~5.1s
  DiT (Neuron, TP=4) ~26.8s  (100 forwards, classic CFG)
  VAE decode (CPU)    ~9.2s
This runner swaps the CPU text encoder with the traced 27-layer Qwen3 module
(core 1) and the CPU VAE decode with the traced decoder (core 2), keeping the
NxDI DiT on cores 0-3, then re-measures the same per-phase breakdown.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_neuronx

sys.path.insert(0, "/mnt/nvme/flux2-klein/src")
from application import NeuronFlux2KleinApplication, create_flux2_klein_config

MODEL = "/mnt/nvme/flux2-klein/weights"
COMPILED = "/mnt/nvme/flux2-klein/compiled_bf16"
TE_DIR = "/mnt/nvme/flux2-klein/compiled_te"
TE_PLACEMENT = {"seg_a": 1, "seg_b": 2, "seg_c": 3}  # one ~5-7 GB NEFF per core
VAE_PT = "/mnt/nvme/flux2-klein/compiled_vae/vae_decoder_1k.pt"
VAE_CORE = 0
OUT_DIR = Path("/mnt/nvme/flux2-klein/outputs_fast")
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


class NeuronTextEncoderShim(torch.nn.Module):
    """Mimics the slice of Qwen3ForCausalLM API used by _get_qwen3_prompt_embeds.

    Three traced Qwen3 segments (layers 0-8 / 9-17 / 18-26) chained across
    cores 1/2/3 produce hidden states 9/18/27; expose them as a {layer: tensor}
    mapping so diffusers' `output.hidden_states[k]` indexing works unchanged.
    """

    def __init__(self, segs):
        super().__init__()
        self.segs = segs

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h9 = self.segs["seg_a"](input_ids, attention_mask)
        h18 = self.segs["seg_b"](h9, attention_mask)
        h27 = self.segs["seg_c"](h18, attention_mask)
        return SimpleNamespace(hidden_states={9: h9, 18: h18, 27: h27})


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

    print("Loading traced TE segments onto cores 1/2/3, VAE onto core 0...", flush=True)
    t0 = time.perf_counter()
    segs = {}
    for name, core in TE_PLACEMENT.items():
        traced = torch.jit.load(f"{TE_DIR}/text_encoder_{name}.pt")
        torch_neuronx.experimental.set_neuron_cores(traced, start_nc=core, nc_count=1)
        segs[name] = traced
    vae_traced = torch.jit.load(VAE_PT)
    torch_neuronx.experimental.set_neuron_cores(vae_traced, start_nc=VAE_CORE, nc_count=1)
    aux_load_seconds = time.perf_counter() - t0

    pipe = app.pipe
    pipe.text_encoder = NeuronTextEncoderShim(segs)
    pipe.vae.decode = lambda z, return_dict=False, **kw: (
        vae_traced(z.to(torch.bfloat16)),
    )

    # RoPE table is shape-only; the wrapper recomputes it on CPU for every one
    # of the 100 forwards unless this (never-used) cache context is enabled.
    rope_cache = app.backbone_app.image_rotary_emb_cache_context()
    rope_cache.__enter__()

    pipe.text_encoder.forward = wrap(pipe.text_encoder.forward, "text_encoder")
    pipe.transformer.forward = wrap(pipe.transformer.forward, "dit")
    pipe.vae.decode = wrap(pipe.vae.decode, "vae_decode")

    report = {
        "config": "neuron TE 3 segs (cores 1/2/3) + neuron DiT TP=4 + neuron VAE (core 0) + RoPE cache",
        "aux_load_seconds": aux_load_seconds,
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
            "other_seconds": round(total - te - dit - vae, 3),
        }
        report["images"].append(entry)
        print(json.dumps(entry), flush=True)

    run_one(52042, "warmup")
    for seed in SEEDS:
        run_one(seed, "measured")

    out_path = OUT_DIR / "breakdown_fast.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE " + str(out_path), flush=True)


if __name__ == "__main__":
    main()
