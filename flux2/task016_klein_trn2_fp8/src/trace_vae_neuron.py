#!/usr/bin/env python3
"""Trace FLUX.2-klein's VAE decoder (AutoencoderKLFlux2) onto Neuron.

Pipeline call site: image = self.vae.decode(latents, return_dict=False)[0]
with latents [1, 32, 128, 128] BF16 (after unpack + BN denorm + unpatchify,
which stay on CPU - they are cheap reshapes). Output [1, 3, 1024, 1024].

Attempt 1 (generic model type, BF16 GroupNorm) failed with NCC_IXTP002:
10.19M instructions > 10M threshold. This version copies what NxDI's FLUX.1
VAE decoder does (models/diffusers/flux/vae/modeling_vae.py):
  - compiler_args "--model-type=unet-inference -O1"
  - GroupNorm computed in FP32 (PatchedGroupNorm) for BF16 numerics
The FLUX.1 decoder has the same [128,256,512,512] block layout at the same
128x128 latent grid and compiles fine at 1K with these settings.

Validates Neuron vs CPU BF16 on random latents, then times the traced module.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers.models import AutoencoderKLFlux2

WEIGHTS = "/mnt/nvme/flux2-klein/weights"
OUT_DIR = Path("/mnt/nvme/flux2-klein/compiled_vae")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_SHAPE = (1, 32, 128, 128)
COMPILER_ARGS = "--model-type=unet-inference -O1 --auto-cast=none"


class Fp32GroupNorm(nn.Module):
    """GroupNorm evaluated in FP32 (same trick as NxDI FLUX.1 PatchedGroupNorm)."""

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.detach().float(), requires_grad=False)
        self.bias = nn.Parameter(gn.bias.detach().float(), requires_grad=False)

    def forward(self, x):
        return F.group_norm(
            x.float(), self.num_groups, self.weight, self.bias, self.eps
        ).to(x.dtype)


def patch_group_norms(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, Fp32GroupNorm(child))
            count += 1
        else:
            count += patch_group_norms(child)
    return count


class VaeDecode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


def main():
    print("Loading VAE (BF16)...", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        WEIGHTS + "/vae", torch_dtype=torch.bfloat16
    )
    vae.eval()
    module = VaeDecode(vae).eval()

    torch.manual_seed(0)
    latent = torch.randn(LATENT_SHAPE, dtype=torch.bfloat16)

    print("CPU BF16 reference decode (unpatched GroupNorm)...", flush=True)
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = module(latent)
        cpu_seconds = time.perf_counter() - t0
    print(f"  cpu decode: {cpu_seconds:.2f}s, out {tuple(ref.shape)}", flush=True)

    n_patched = patch_group_norms(module)
    print(f"Patched {n_patched} GroupNorm modules to FP32 compute", flush=True)

    print(f"Tracing on Neuron (compile) with: {COMPILER_ARGS}", flush=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        module,
        (latent,),
        compiler_workdir=str(OUT_DIR / "compiler_workdir"),
        compiler_args=COMPILER_ARGS,
    )
    compile_seconds = time.perf_counter() - t0
    print(f"Compile done in {compile_seconds:.1f}s", flush=True)

    report = {
        "compiler_args": COMPILER_ARGS,
        "group_norms_patched_fp32": n_patched,
        "compile_seconds": compile_seconds,
        "cpu_decode_seconds": cpu_seconds,
    }
    with torch.no_grad():
        got = traced(latent)
        ref_f = ref.float()
        got_f = got.float()
        cos = torch.nn.functional.cosine_similarity(
            ref_f.flatten(), got_f.flatten(), dim=0
        ).item()
        report["cosine_vs_cpu_bf16"] = cos
        report["max_abs_diff"] = (ref_f - got_f).abs().max().item()
        report["mean_abs_diff"] = (ref_f - got_f).abs().mean().item()
        print(
            f"  cos={cos:.6f} max_abs={report['max_abs_diff']:.4f} "
            f"mean_abs={report['mean_abs_diff']:.5f}",
            flush=True,
        )

        for _ in range(3):
            traced(latent)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            traced(latent)
            times.append(time.perf_counter() - t0)
    report["neuron_latency_ms"] = {
        "mean": 1000 * sum(times) / len(times),
        "min": 1000 * min(times),
        "max": 1000 * max(times),
    }
    print(f"Neuron VAE latency: {report['neuron_latency_ms']['mean']:.1f} ms mean", flush=True)

    model_path = OUT_DIR / "vae_decoder_1k.pt"
    torch.jit.save(traced, str(model_path))
    report["model_path"] = str(model_path)
    (OUT_DIR / "vae_trace_report.json").write_text(json.dumps(report, indent=2))
    print("WROTE " + str(model_path), flush=True)


if __name__ == "__main__":
    main()
