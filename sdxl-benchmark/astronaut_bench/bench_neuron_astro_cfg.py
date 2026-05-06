"""Track A bench: SDXL with CFG=7.5 on trn2, using batch=2 BF16 UNet NEFF.

Based on bench_final.py but:
  - Load UNet from ~/sdxl_compile_b2_1024/unet/model.pt (batch=2)
  - guidance_scale=7.5 — pipeline duplicates inputs to batch=2
  - Still uses no-DataParallel single jit.load
  - CLIP-L wrapper returns text_embeds=None so pipeline picks CLIP-G pooled
  - timestep wrapper uses .expand((B,)) matching trace_unet_batch2.py
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import diffusers
from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

DTYPE = torch.bfloat16
B2_COMPILE = "/home/ubuntu/sdxl_compile_b2_1024"
# Text encoders + VAE decoder + post_quant_conv can reuse batch=1 NEFFs
# (they're called once per step, not on the CFG-doubled batch).
REUSE_COMPILE = "/home/ubuntu/sdxl_compile_aws_nb_1024"


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        return self.unet(sample, timestep, encoder_hidden_states,
                         added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                         return_dict=False)


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.config.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None,
                timestep_cond=None, attention_mask=None, encoder_attention_mask=None,
                mid_block_additional_residual=None, down_block_additional_residuals=None,
                down_intrablock_additional_residuals=None, **kwargs):
        # NEFF was traced with FP32 sample/encoder/text_embeds/time_ids and
        # 0-dim FP32 timestep. Bench loaded pipeline in BF16, so cast here.
        sample = self.unetwrap(
            sample.float(),
            timestep.float().reshape(()),
            encoder_hidden_states.float(),
            added_cond_kwargs["text_embeds"].float(),
            added_cond_kwargs["time_ids"].float(),
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder, is_clip_g):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device
        self.is_clip_g = is_clip_g

    def forward(self, text_input_ids, output_hidden_states=True):
        out = self.traceable_text_encoder(text_input_ids)
        # Traced output layouts:
        #   CLIP-L (CLIPTextModel):          (last_hidden_state, pooler_output, hidden_states_tuple)
        #   CLIP-G (CLIPTextModelWithProjection): (text_embeds, last_hidden_state, hidden_states_tuple)
        # NB: hidden_states is out[2] (already a tuple), not out[2:].
        if self.is_clip_g:
            return CLIPTextModelOutput(
                text_embeds=out[0],
                last_hidden_state=out[1],
                hidden_states=out[2],
            )
        else:
            # CLIP-L: text_embeds=None forces diffusers 0.38 to use CLIP-G's pooled.
            return CLIPTextModelOutput(
                text_embeds=None,
                last_hidden_state=out[0],
                hidden_states=out[2],
            )


def main():
    model_id = "/home/ubuntu/models/sdxl-base"
    out_dir = os.path.expanduser("~/sdxl_b2_cfg_out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cfg] B2_COMPILE={B2_COMPILE}  REUSE={REUSE_COMPILE}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
    Attention.get_attention_scores = get_attention_scores_neuron

    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    t0 = time.time()
    print("[load] unet batch=2 ...", flush=True)
    pipe.unet.unetwrap = torch.jit.load(os.path.join(B2_COMPILE, "unet/model.pt"))
    print(f"[load] unet loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    pipe.vae.decoder = torch.jit.load(os.path.join(REUSE_COMPILE, "vae_decoder/model.pt"))
    pipe.vae.post_quant_conv = torch.jit.load(os.path.join(REUSE_COMPILE, "vae_post_quant_conv/model.pt"))
    pipe.text_encoder = TextEncoderOutputWrapper(
        torch.jit.load(os.path.join(REUSE_COMPILE, "text_encoder/model.pt")), pipe.text_encoder, is_clip_g=False)
    pipe.text_encoder_2 = TextEncoderOutputWrapper(
        torch.jit.load(os.path.join(REUSE_COMPILE, "text_encoder_2/model.pt")), pipe.text_encoder_2, is_clip_g=True)
    print(f"[load] rest loaded in {time.time()-t0:.1f}s", flush=True)

    prompt = "An astronaut riding a green horse"
    print(f"[bench] prompt={prompt!r}  steps=50  1024x1024  CFG=7.5", flush=True)

    print("[bench] warmup ...", flush=True)
    t_wu = time.time()
    _ = pipe(prompt, num_inference_steps=50, height=1024, width=1024, guidance_scale=7.5,
             generator=torch.Generator().manual_seed(42)).images[0]
    print(f"[bench] warmup done in {time.time()-t_wu:.2f}s", flush=True)

    latencies = []
    seeds = list(range(42, 52))
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        t0 = time.time()
        image = pipe(prompt, num_inference_steps=50, height=1024, width=1024,
                     guidance_scale=7.5, generator=gen).images[0]
        dt = time.time() - t0
        latencies.append(dt)
        path = os.path.join(out_dir, f"seed{seed}.png")
        image.save(path)
        print(f"[bench] seed={seed}  latency={dt:.3f}s  -> {path}", flush=True)

    arr = np.array(latencies)
    summary = {
        "config": "trn2.3xl BF16 CFG=7.5 batch=2 1024x1024 50 steps",
        "prompt": prompt,
        "seeds": seeds,
        "latencies_s": [round(x, 3) for x in latencies],
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60, flush=True)
    print("RESULTS  trn2.3xl CFG=7.5 batch=2 1024^2 50 steps", flush=True)
    for k, v in summary.items():
        print(f"  {k:15s} : {v}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
