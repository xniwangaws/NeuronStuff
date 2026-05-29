"""End-to-end SDXL 1024x1024 image generation on Trn2 with TP=4 UNet.

Composition:
- UNet: TP=4 NEFF compiled via ModelBuilder. Loaded by re-running
  builder.trace + builder.compile (cache hit, fast) then weight sharding.
  Wrapped to split CFG batch=2 -> two batch=1 calls.
- Text encoders: traced TorchScript NEFFs from Jim's compile_dir_1024.
- VAE decoder + post_quant_conv: traced TorchScript NEFFs from same dir.

Outputs:
- E2E_1K_TP4_METRICS.json
- sdxl_1k_tp4_seed{42,43}.png
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch_neuronx

sys.path.insert(0, "/home/ubuntu/work_unet_tp4")

# Must be set BEFORE neuron init.
os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput


COMPILE_DIR_1K = "/home/ubuntu/sdxl/compile_dir_1024"
TP4_COMPILE_DIR = "/home/ubuntu/work_unet_tp4/compile_tp4_1k"
SDXL_BASE = "/home/ubuntu/sdxl-base"
OUT_DIR = "/home/ubuntu/work_unet_tp4"
DTYPE = torch.bfloat16


# ---- attention patch (must match how text encoders / VAE were traced) ----
def custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        probs = scores.softmax(dim=1).permute(0, 2, 1)
    else:
        scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        probs = scores.softmax(dim=-1)
    return probs


Attention.get_attention_scores = get_attention_scores_neuron


# ---- Wrappers for traced text encoders / VAE (Jim's pattern) ----
class NeuronTextEncoder(nn.Module):
    def __init__(self, traced, original, kind):
        super().__init__()
        self.traced = traced
        self.config = original.config
        self.dtype = original.dtype
        self.device = original.device
        self.kind = kind  # "clip_l" or "clip_g"

    def forward(self, ids, output_hidden_states=True, **kwargs):
        out = self.traced(ids)
        if self.kind == "clip_l":
            text_embeds_field = out[1]  # 3D last_hidden_state
        else:
            text_embeds_field = out[0]  # 2D projected (1280-d)
        return CLIPTextModelOutput(
            text_embeds=text_embeds_field,
            last_hidden_state=out[1],
            hidden_states=out[2:],
        )


# ---- TP=4 UNet wrapper ----
# nxd_model is the compiled NxDModel from ModelBuilder. It wants
# (sample, timestep, encoder_hidden_states, text_embeds, time_ids) with sample
# batch=1. We split CFG batch=2 into two calls and concat.

# CFG_TIMING is filled by the wrapper for per-step measurement.
UNET_CALL_LATENCIES = []  # ms


class TP4UNetWrapper(nn.Module):
    def __init__(self, nxd_model, original_unet):
        super().__init__()
        self.nxd_model = nxd_model
        self.config = original_unet.config
        self.in_channels = original_unet.in_channels
        self.add_embedding = original_unet.add_embedding
        self.device = original_unet.device
        self.dtype = original_unet.dtype

    def _call_neff(self, sample, timestep_scalar, ehs, text_embeds, time_ids):
        # sample: [1,4,H,W], timestep_scalar: shape [1] long
        out = self.nxd_model(sample, timestep_scalar, ehs, text_embeds, time_ids)
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, dict):
            return list(out.values())[0]
        return out

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None,
                **kwargs):
        # Diffusers passes timestep typically as scalar tensor; expand to shape [1].
        if torch.is_tensor(timestep):
            ts = timestep.long().reshape(-1)
            if ts.numel() == 0:
                ts = torch.tensor([int(timestep.item())], dtype=torch.long)
        else:
            ts = torch.tensor([int(timestep)], dtype=torch.long)
        # NEFF was traced with timestep shape [1]
        ts1 = ts[:1].clone()

        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]

        # bf16 cast
        sample = sample.to(DTYPE)
        encoder_hidden_states = encoder_hidden_states.to(DTYPE)
        text_embeds = text_embeds.to(DTYPE)
        time_ids = time_ids.to(DTYPE)

        bsz = sample.shape[0]
        t0 = time.time()
        if bsz == 1:
            out = self._call_neff(sample, ts1, encoder_hidden_states, text_embeds, time_ids)
        else:
            # Split batch into individual batch=1 calls (CFG)
            chunks = []
            for i in range(bsz):
                ehs_i = encoder_hidden_states[i:i+1]
                te_i = text_embeds[i:i+1]
                ti_i = time_ids[i:i+1]
                s_i = sample[i:i+1]
                chunks.append(self._call_neff(s_i, ts1, ehs_i, te_i, ti_i))
            out = torch.cat(chunks, dim=0)
        UNET_CALL_LATENCIES.append((time.time() - t0) * 1000.0)

        return UNet2DConditionOutput(sample=out)


def build_tp4_unet(unet_cpu_path, latent_h=128, latent_w=128, tp=4,
                   workdir=TP4_COMPILE_DIR):
    """Re-trace + load TP=4 UNet via ModelBuilder. Cache hit makes this fast.
    Returns the wrapper module (already loaded to neuron)."""
    from neuronx_distributed.trace.parallel_context import NxDParallelState
    from neuronx_distributed.trace.model_builder_v2 import ModelBuilder
    from neuronx_distributed.trace.functions import shard_checkpoint
    from diffusers import UNet2DConditionModel
    from neuron_unet import shard_unet_in_place, UNetTraceWrapper

    print(f"[unet] Loading SDXL UNet from {unet_cpu_path} on CPU bf16...", flush=True)
    t0 = time.time()
    unet_cpu = UNet2DConditionModel.from_pretrained(
        unet_cpu_path, variant="fp16", torch_dtype=DTYPE
    )
    unet_cpu.eval()
    master_sd = {k: v.detach().clone() for k, v in unet_cpu.state_dict().items()}
    original_for_config = unet_cpu  # need config / add_embedding
    print(f"[unet] loaded ({time.time()-t0:.1f}s)", flush=True)

    H, W = latent_h, latent_w
    example_zero = (
        torch.zeros(1, 4, H, W, dtype=DTYPE),
        torch.tensor([999], dtype=torch.long),
        torch.zeros(1, 77, 2048, dtype=DTYPE),
        torch.zeros(1, 1280, dtype=DTYPE),
        torch.zeros(1, 6, dtype=DTYPE),
    )
    with NxDParallelState(world_size=tp, tensor_model_parallel_size=tp):
        unet_sharded = UNet2DConditionModel.from_pretrained(
            unet_cpu_path, variant="fp16", torch_dtype=DTYPE
        )
        unet_sharded.eval()
        stats = shard_unet_in_place(unet_sharded, tp=tp, dtype=DTYPE)
        wrapped = UNetTraceWrapper(unet_sharded)
        wrapped.eval()
        print(f"[unet] sharded {stats['linear_replaced']}L {stats['conv_replaced']}C",
              flush=True)

        builder = ModelBuilder(wrapped)
        print("[unet] trace + compile (cache should hit)...", flush=True)
        t0 = time.time()
        builder.trace(args=example_zero, tag="unet_tp4")
        nxd_model = builder.compile(
            compiler_workdir=workdir,
            compiler_args="--model-type=unet-inference --lnc=2 -O1",
        )
        print(f"[unet] compile/load: {time.time()-t0:.1f}s", flush=True)

        prefixed = {f"unet.{k}": v for k, v in master_sd.items()}
        sharded_sd = shard_checkpoint(prefixed, wrapped)
        nxd_model.set_weights(sharded_sd)
        nxd_model.to_neuron()
        print("[unet] weights loaded to neuron cores", flush=True)

    wrapper = TP4UNetWrapper(nxd_model, original_for_config)
    return wrapper


def build_pipeline():
    print("[pipe] StableDiffusionXLPipeline.from_pretrained...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE, torch_dtype=DTYPE, variant="fp16"
    )

    # Replace text encoders with traced 1K NEFFs
    print("[pipe] loading traced text encoders...", flush=True)
    te1 = torch.jit.load(f"{COMPILE_DIR_1K}/text_encoder/model.pt")
    te2 = torch.jit.load(f"{COMPILE_DIR_1K}/text_encoder_2/model.pt")
    pipe.text_encoder = NeuronTextEncoder(te1, pipe.text_encoder, "clip_l")
    pipe.text_encoder_2 = NeuronTextEncoder(te2, pipe.text_encoder_2, "clip_g")

    # Replace VAE decoder + post_quant_conv with traced 1K NEFFs
    print("[pipe] loading traced VAE decoder...", flush=True)
    vae_dec = torch.jit.load(f"{COMPILE_DIR_1K}/vae_decoder/model.pt")
    vae_pqc = torch.jit.load(f"{COMPILE_DIR_1K}/vae_post_quant_conv/model.pt")
    pipe.vae.decoder = vae_dec
    pipe.vae.post_quant_conv = vae_pqc

    # Replace UNet with TP=4 wrapper
    print("[pipe] building TP=4 UNet (this may take a while on cold cache)...",
          flush=True)
    tp4_unet = build_tp4_unet(f"{SDXL_BASE}/unet", latent_h=128, latent_w=128, tp=4)
    pipe.unet = tp4_unet

    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="An astronaut riding a green horse")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--num_warm", type=int, default=3)
    args = ap.parse_args()

    pipe = build_pipeline()
    pipe.set_progress_bar_config(disable=True)

    metrics = {
        "prompt": args.prompt, "steps": args.steps, "cfg": args.cfg,
        "height": 1024, "width": 1024,
        "tp": 4, "unet_arch": "TP=4 NEFF, batch=1, split CFG -> 2x batch=1",
        "runs": [],
    }

    seeds_for_imgs = [42, 43]
    saved = {}
    runs = []
    # 1 cold + N warm using seed 42
    n_total = 1 + args.num_warm
    for run_idx in range(n_total):
        UNET_CALL_LATENCIES.clear()
        gen = torch.Generator(device="cpu").manual_seed(42)

        t0 = time.time()
        result = pipe(
            prompt=args.prompt, height=1024, width=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=gen,
        )
        total = time.time() - t0
        img = result.images[0]
        unet_calls = list(UNET_CALL_LATENCIES)
        run = {
            "idx": run_idx,
            "kind": "cold" if run_idx == 0 else "warm",
            "total_s": total,
            "num_unet_calls": len(unet_calls),
            "unet_call_ms_mean": sum(unet_calls) / max(1, len(unet_calls)),
            "unet_call_ms_min": min(unet_calls) if unet_calls else 0,
            "unet_call_ms_max": max(unet_calls) if unet_calls else 0,
            "unet_total_s": sum(unet_calls) / 1000.0,
        }
        print(f"[run {run_idx}] {run['kind']} total={run['total_s']:.2f}s "
              f"unet_calls={run['num_unet_calls']} "
              f"unet_mean={run['unet_call_ms_mean']:.1f}ms "
              f"unet_total={run['unet_total_s']:.2f}s", flush=True)
        runs.append(run)
        if run_idx == 0:
            img.save(f"{OUT_DIR}/sdxl_1k_tp4_seed42.png")
            saved["seed42"] = f"{OUT_DIR}/sdxl_1k_tp4_seed42.png"

    # bonus seed 43 (warm)
    UNET_CALL_LATENCIES.clear()
    gen = torch.Generator(device="cpu").manual_seed(43)
    t0 = time.time()
    img43 = pipe(prompt=args.prompt, height=1024, width=1024,
                 num_inference_steps=args.steps, guidance_scale=args.cfg,
                 generator=gen).images[0]
    t43 = time.time() - t0
    img43.save(f"{OUT_DIR}/sdxl_1k_tp4_seed43.png")
    saved["seed43"] = f"{OUT_DIR}/sdxl_1k_tp4_seed43.png"
    seed43_run = {"idx": "seed43", "kind": "warm",
                  "total_s": t43,
                  "num_unet_calls": len(UNET_CALL_LATENCIES),
                  "unet_call_ms_mean": sum(UNET_CALL_LATENCIES)/max(1, len(UNET_CALL_LATENCIES)),
                  "unet_total_s": sum(UNET_CALL_LATENCIES)/1000.0}
    runs.append(seed43_run)
    print(f"[seed43] total={t43:.2f}s", flush=True)

    metrics["runs"] = runs
    metrics["saved_pngs"] = saved
    cold = runs[0]["total_s"]
    warms = [r["total_s"] for r in runs if r["kind"] == "warm" and r["idx"] != "seed43"]
    metrics["cold_s"] = cold
    metrics["warm_s_mean"] = sum(warms) / len(warms) if warms else 0
    metrics["warm_s_list"] = warms

    with open(f"{OUT_DIR}/E2E_1K_TP4_METRICS.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] metrics -> {OUT_DIR}/E2E_1K_TP4_METRICS.json", flush=True)


if __name__ == "__main__":
    main()
