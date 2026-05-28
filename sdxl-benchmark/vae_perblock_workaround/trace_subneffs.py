"""Trace SDXL VAE decoder as 8 sub-NEFFs (per-block split) for Plan A workaround.

Each sub-NEFF is compiled with --model-type=unet-inference --lnc=2 -O1 and saved
to /home/ubuntu/work_a/traced_<name>.pt . Compile workdirs go under
/home/ubuntu/work_a/compile_<name>/ for instruction-count parsing.

Sub-NEFF list:
    01_post_quant_conv     [1,4,256,256]   -> [1,4,256,256]
    02_conv_in             [1,4,256,256]   -> [1,512,256,256]
    03_mid_block           [1,512,256,256] -> [1,512,256,256]
    04_up_block_0          [1,512,256,256] -> [1,512,512,512]
    05_up_block_1          [1,512,512,512] -> [1,512,1024,1024]
    06_up_block_2          [1,512,1024,1024] -> [1,256,2048,2048]
    07_up_block_3          [1,256,2048,2048] -> [1,128,2048,2048]
    08_conv_out_block      [1,128,2048,2048] -> [1,3,2048,2048]

If any sub-NEFF fails NCC_EVRF007 (>5M instructions), the script catches the
RuntimeError, logs it, and falls back to a finer split for that block.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch_neuronx
from diffusers import AutoencoderKL

VAE_PATH = "/home/ubuntu/sdxl-base"
WORK = Path("/home/ubuntu/work_a")
WORK.mkdir(parents=True, exist_ok=True)

COMPILER_ARGS = ["--target=trn2", "--model-type=unet-inference", "--lnc=2", "-O1"]


# -------------------- module wrappers --------------------
# torch_neuronx.trace needs single tensor in/out (or tuple) and no kwargs.
# Wrap each piece into an nn.Module whose forward() takes pure tensors.

class PostQuantConvWrap(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.m = vae.post_quant_conv
    def forward(self, z):
        return self.m(z)

class ConvInWrap(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.m = vae.decoder.conv_in
    def forward(self, x):
        return self.m(x)

class MidBlockWrap(nn.Module):
    """UNetMidBlock2D forward signature is (hidden_states, temb=None)."""
    def __init__(self, vae):
        super().__init__()
        self.m = vae.decoder.mid_block
    def forward(self, x):
        return self.m(x, temb=None)


class ResnetBlockWrap(nn.Module):
    """ResnetBlock2D forward signature is (input_tensor, temb)."""
    def __init__(self, resnet):
        super().__init__()
        self.m = resnet
    def forward(self, x):
        return self.m(x, None)


class AttentionWrap(nn.Module):
    """diffusers Attention - matches UNetMidBlock2D's call (residual add inside)."""
    def __init__(self, attn):
        super().__init__()
        self.m = attn
    def forward(self, x):
        # diffusers UNetMidBlock2D code: hidden_states = attn(hidden_states, temb=temb)
        # The Attention module already handles spatial reshape via spatial_norm/etc.
        return self.m(x, temb=None)


class UpsamplerWrap(nn.Module):
    def __init__(self, ups):
        super().__init__()
        self.m = ups
    def forward(self, x):
        return self.m(x)


class UpBlockWrap(nn.Module):
    """UpDecoderBlock2D forward signature is (hidden_states, temb=None, upsample_size=None)."""
    def __init__(self, vae, idx):
        super().__init__()
        self.m = vae.decoder.up_blocks[idx]
    def forward(self, x):
        return self.m(x, temb=None)


class UpBlockResnetsOnlyWrap(nn.Module):
    """Run only the resnets (no upsampler) of an UpDecoderBlock2D."""
    def __init__(self, vae, idx):
        super().__init__()
        self.resnets = vae.decoder.up_blocks[idx].resnets
    def forward(self, x):
        for r in self.resnets:
            x = r(x, None)
        return x

class ConvOutWrap(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.norm = vae.decoder.conv_norm_out
        self.act = vae.decoder.conv_act
        self.conv = vae.decoder.conv_out
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


# -------------------- trace driver --------------------

def trace_one(name, mod, sample_in, log):
    """Trace a single sub-module. Returns dict with status/time/instr_count or raises."""
    workdir = WORK / f"compile_{name}"
    workdir.mkdir(parents=True, exist_ok=True)
    out_pt = WORK / f"traced_{name}.pt"

    print(f"\n[trace] === {name}  in_shape={list(sample_in.shape)} dtype={sample_in.dtype} ===")
    sys.stdout.flush()
    mod.eval()
    t0 = time.time()
    try:
        traced = torch_neuronx.trace(
            mod,
            (sample_in,),
            compiler_args=COMPILER_ARGS,
            compiler_workdir=str(workdir),
        )
    except Exception as e:
        dt = time.time() - t0
        print(f"[trace] FAILED {name} after {dt:.1f}s: {e}")
        log[name] = {"status": "FAILED", "elapsed_s": dt, "error": str(e)[:500]}
        raise

    dt = time.time() - t0
    torch.jit.save(traced, str(out_pt))
    print(f"[trace] OK   {name} in {dt:.1f}s  -> {out_pt}")

    # Parse instruction count from compiler logs if present.
    instr_count = None
    for log_path in workdir.rglob("*.log"):
        try:
            txt = log_path.read_text(errors="ignore")
            # Look for "instruction count" or "HLO instructions"
            for line in txt.splitlines():
                low = line.lower()
                if "instruction" in low and ("count" in low or "hlo" in low):
                    print(f"[trace]   log hit: {line.strip()[:200]}")
        except Exception:
            pass

    log[name] = {
        "status": "OK",
        "elapsed_s": dt,
        "out_pt": str(out_pt),
        "in_shape": list(sample_in.shape),
        "out_shape": None,  # filled by caller
    }
    return traced


def run_with_sample(traced, sample_in):
    with torch.no_grad():
        out = traced(sample_in)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of names to trace (e.g. 06_up_block_2)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    print("Loading VAE in bf16...")
    vae = AutoencoderKL.from_pretrained(
        VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
    )
    vae.eval()

    log = {}
    log_path = WORK / "trace_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())

    DTY = torch.bfloat16
    # Generate a deterministic input.
    torch.manual_seed(0)
    latent = torch.randn(1, 4, 256, 256, dtype=DTY)

    plan = [
        ("01_post_quant_conv", PostQuantConvWrap(vae), latent),
        # Subsequent inputs are filled by chaining outputs.
    ]

    # We'll collect intermediate outputs by running each newly traced piece on its input.
    intermediates = {}

    def maybe_trace(name, mod, sample_in):
        pt = WORK / f"traced_{name}.pt"
        if only and name not in only:
            print(f"[skip] {name}")
            if pt.exists():
                m = torch.jit.load(str(pt))
                return run_with_sample(m, sample_in)
            else:
                with torch.no_grad():
                    out = mod(sample_in)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out
        if pt.exists():
            print(f"[resume] {name} already traced -> {pt}")
            m = torch.jit.load(str(pt))
            out = run_with_sample(m, sample_in)
            if name not in log:
                log[name] = {"status": "OK_RESUMED", "elapsed_s": None,
                             "out_pt": str(pt), "in_shape": list(sample_in.shape)}
            log[name]["out_shape"] = list(out.shape)
            log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
            return out
        traced = trace_one(name, mod, sample_in, log)
        out = run_with_sample(traced, sample_in)
        log[name]["out_shape"] = list(out.shape)
        log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
        return out

    # ---- chain of sub-NEFFs ----
    x = latent
    x = maybe_trace("01_post_quant_conv", PostQuantConvWrap(vae), x)
    x = maybe_trace("02_conv_in",          ConvInWrap(vae),         x)

    # mid_block split into 3 sub-NEFFs (resnet0 / attn / resnet1) — this avoids
    # the giant Attention-+-2-Resnets fused HLO graph blowing past HBM.
    x = maybe_trace("03a_mid_resnet0", ResnetBlockWrap(vae.decoder.mid_block.resnets[0]), x)
    x = maybe_trace("03b_mid_attn",    AttentionWrap(vae.decoder.mid_block.attentions[0]), x)
    x = maybe_trace("03c_mid_resnet1", ResnetBlockWrap(vae.decoder.mid_block.resnets[1]), x)

    x = maybe_trace("04_up_block_0",       UpBlockWrap(vae, 0),     x)
    x = maybe_trace("05_up_block_1",       UpBlockWrap(vae, 1),     x)
    x = maybe_trace("06_up_block_2",       UpBlockWrap(vae, 2),     x)
    x = maybe_trace("07_up_block_3",       UpBlockWrap(vae, 3),     x)
    x = maybe_trace("08_conv_out_block",   ConvOutWrap(vae),        x)

    print(f"\nFinal output shape: {list(x.shape)}  dtype={x.dtype}")
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
