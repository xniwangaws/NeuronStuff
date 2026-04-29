"""
Migrate aws-neuron-samples hf_pretrained_sdxl_base_1024_inference.ipynb
(commit f532a05) to trn2.3xlarge + SDK 2.29 + BF16 + LNC=2.

Six deltas vs. the notebook (see tasks/active/004-notebook-migration.md):
  1. torch_dtype float32 -> bfloat16
  2. drop diffusers==0.29.2 / transformers==4.42.3 pins; use venv defaults
  3. DataParallel device_ids [0,1] -> [0,1,2,3] (4 logical cores on LNC=2)
  4. absolute compile workdir
  5. add --auto-cast matmult (no-op on BF16 but keeps AGENTS.md rule)
  6. caller runs benchmark_neuron.py after this; this script only compiles

The four wrapper classes and the attention patch are preserved verbatim from
cell 8 of the notebook.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python trace_sdxl.py 2>&1 | tee compile.log
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

COMPILER_WORKDIR_ROOT = "/home/ubuntu/sdxl/compile_dir"
MODEL_ID = os.environ.get("SDXL_MODEL_ID", "/home/ubuntu/models/sdxl-base")
DTYPE = torch.bfloat16
# LNC=2 on trn2.3xlarge -> 4 logical cores. Drop to [0, 1] if UNet compile OOMs
# the host (neuronx-cc has been seen to peak >80 GB on 124 GB trn2.3xlarge).
UNET_DEVICE_IDS = [int(x) for x in os.environ.get("SDXL_UNET_CORES", "0,1,2,3").split(",")]
COMPILER_ARGS = ["--auto-cast", "matmult"]


# --- attention patch from notebook cell 8 (unchanged) ---------------------

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


Attention.get_attention_scores = get_attention_scores_neuron


# --- wrappers from notebook cell 8 (unchanged) ----------------------------

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TraceableTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input_ids):
        return self.text_encoder(text_input_ids, output_hidden_states=True)


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out_tuple = self.traceable_text_encoder(text_input_ids)
        return CLIPTextModelOutput(
            text_embeds=out_tuple[0],
            last_hidden_state=out_tuple[1],
            hidden_states=out_tuple[2],
        )


# --- compile ---------------------------------------------------------------

def trace_and_save(module, example_inputs, save_path, label):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[trace] {label} -> {save_path}")
    t0 = time.time()
    traced = torch_neuronx.trace(module, example_inputs, compiler_args=COMPILER_ARGS)
    elapsed = time.time() - t0
    torch.jit.save(traced, save_path)
    size_gb = os.path.getsize(save_path) / 1e9
    print(f"[trace] {label} done in {elapsed:.1f}s, size {size_gb:.2f} GB")
    return elapsed


def main():
    os.makedirs(COMPILER_WORKDIR_ROOT, exist_ok=True)

    # ---- text encoders (shapes mirror notebook cell 10) -------------------
    print("[load] pipeline for text-encoder tracing (BF16)")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    te1 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder))
    te2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2))

    # 77-token sequence (SDXL CLIP max length)
    input_ids = torch.zeros((1, 77), dtype=torch.long)
    trace_and_save(te1, input_ids, f"{COMPILER_WORKDIR_ROOT}/text_encoder/model.pt", "text_encoder (CLIP-L)")
    trace_and_save(te2, input_ids, f"{COMPILER_WORKDIR_ROOT}/text_encoder_2/model.pt", "text_encoder_2 (CLIP-G)")

    del te1, te2, pipe

    # ---- UNet -------------------------------------------------------------
    print("[load] pipeline for UNet tracing (BF16)")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    unet = copy.deepcopy(UNetWrap(pipe.unet))
    del pipe

    # UNet at 1024x1024: latent 128x128, 4 channels, SDXL uses 2xbatch for CFG
    sample = torch.randn([2, 4, 128, 128], dtype=DTYPE)
    timestep = torch.tensor(999, dtype=torch.float32)
    encoder_hidden_states = torch.randn([2, 77, 2048], dtype=DTYPE)
    text_embeds = torch.randn([2, 1280], dtype=DTYPE)
    time_ids = torch.randn([2, 6], dtype=DTYPE)

    trace_and_save(
        unet,
        (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
        f"{COMPILER_WORKDIR_ROOT}/unet/model.pt",
        "UNet",
    )
    del unet

    # ---- VAE decoder + post_quant_conv ------------------------------------
    print("[load] pipeline for VAE tracing (BF16)")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    # Decoder takes [1, 4, 128, 128] latents at 1024x1024
    latent = torch.randn([1, 4, 128, 128], dtype=DTYPE)
    decoder = copy.deepcopy(pipe.vae.decoder)
    trace_and_save(
        decoder,
        latent,
        f"{COMPILER_WORKDIR_ROOT}/vae_decoder/model.pt",
        "VAE decoder",
    )

    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    trace_and_save(
        post_quant_conv,
        latent,
        f"{COMPILER_WORKDIR_ROOT}/vae_post_quant_conv/model.pt",
        "VAE post_quant_conv",
    )

    print("[done] all 5 NEFF-wrapped modules saved under", COMPILER_WORKDIR_ROOT)
    print("[info] UNet will run on cores", UNET_DEVICE_IDS, "via torch_neuronx.DataParallel at inference time")


if __name__ == "__main__":
    main()
