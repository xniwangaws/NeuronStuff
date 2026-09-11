# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Neuron text encoder and VAE decoder for FLUX.2-klein (the two stages that
previously ran on the host CPU).

Measured on trn2.3xlarge at 1024x1024 / 50 steps / CFG (2026-09-11):

    stage                    CPU (before)   Neuron (after)
    Qwen3-8B text encoder x2      5.1 s          0.12 s
    VAE decode                    9.2 s          0.29 s
    DiT 100 forwards (TP=4)      26.8 s         25.3 s   (RoPE table cached)
    total                        41.4 s         26.0 s

Text encoder
------------
diffusers' Flux2KleinPipeline._get_qwen3_prompt_embeds only consumes hidden
states 9/18/27 of Qwen3-8B for a fixed [1, 512] input, so layers 27..35 and
lm_head never run. The 27 remaining layers are traced with torch_neuronx as
three segments cut at those boundaries:

    seg_a  embed_tokens + layers  0..8   input_ids -> h9
    seg_b                 layers  9..17  h9        -> h18
    seg_c                 layers 18..26  h18       -> h27

A single 27-layer NEFF (18.9 GB) does not fit next to the TP=4 DiT shard in
one logical core's ~22 GiB HBM; the segments (7.2 / 5.2 / 5.2 GB) are placed
one per logical core 1/2/3. Each segment is an ordinary transformers
Qwen3Model over a layer slice with the final RMSNorm replaced by Identity
(so its output is the raw layer output, exactly hidden_states[k] of the full
model); HF's own mask/RoPE code is reused rather than re-implemented.
Segments b/c take `inputs_embeds`. Validation against the pipeline-exact CPU
BF16 reference: cosine 1.000, mean |diff| 0.067 on a tensor with RMS ~16.

VAE decoder
-----------
AutoencoderKLFlux2.decode at latent [1, 32, 128, 128] -> image [1, 3, 1024,
1024]. Uses the same recipe as NxDI's FLUX.1 VAE decoder:
`--model-type=unet-inference` (the generic model type exceeded the 10M
instruction limit, NCC_IXTP002) and GroupNorm evaluated in FP32. Placed on
logical core 0. Validation vs CPU BF16: cosine 0.9998.
"""

import copy
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TE_HIDDEN_LAYERS = (9, 18, 27)
TE_SEGMENTS = (("seg_a", 0, 9), ("seg_b", 9, 18), ("seg_c", 18, 27))
TE_DEFAULT_PLACEMENT = {"seg_a": 1, "seg_b": 2, "seg_c": 3}
TE_COMPILER_ARGS = "--model-type=transformer -O1 --auto-cast=none"
TE_MAX_LEN = 512

VAE_DEFAULT_CORE = 0
VAE_COMPILER_ARGS = "--model-type=unet-inference -O1 --auto-cast=none"


def _set_cores(traced, start_nc: int, nc_count: int = 1):
    import torch_neuronx

    setter = getattr(torch_neuronx, "set_neuron_cores", None)
    if setter is None:  # older torch_neuronx
        setter = torch_neuronx.experimental.set_neuron_cores
    setter(traced, start_nc=start_nc, nc_count=nc_count)


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------


def build_qwen3_segment(full, start: int, end: int, with_embed: bool):
    """Qwen3Model over full.layers[start:end], sharing parameter storage."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    cfg = copy.deepcopy(full.config)
    cfg.num_hidden_layers = end - start
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types[start:end])
    with torch.device("meta"):
        seg = Qwen3Model(cfg)
    state = {}
    for i in range(start, end):
        for k, v in full.layers[i].state_dict().items():
            state[f"layers.{i - start}.{k}"] = v
    state["embed_tokens.weight"] = full.embed_tokens.weight
    state["norm.weight"] = full.norm.weight
    seg.load_state_dict(state, strict=True, assign=True)
    seg.rotary_emb = copy.deepcopy(full.rotary_emb)  # non-persistent buffer
    seg.norm = nn.Identity()  # raw layer output == hidden_states[end]
    if not with_embed:
        seg.embed_tokens = nn.Embedding(2, cfg.hidden_size)  # unused
    return seg.eval()


class _IdsSegment(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, input_ids, attention_mask):
        return self.seg(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
        ).last_hidden_state


class _HiddenSegment(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, hidden, attention_mask):
        return self.seg(
            inputs_embeds=hidden, attention_mask=attention_mask, use_cache=False, return_dict=True
        ).last_hidden_state


def te_segment_path(compiled_dir: str, name: str) -> str:
    return os.path.join(compiled_dir, "text_encoder", f"{name}.pt")


def compile_text_encoder(text_encoder, tokenizer, compiled_dir: str, compiler_args: str = TE_COMPILER_ARGS):
    """Trace the three Qwen3 segments. `text_encoder` is the CPU Qwen3ForCausalLM."""
    import torch_neuronx

    out_dir = os.path.join(compiled_dir, "text_encoder")
    os.makedirs(out_dir, exist_ok=True)
    if all(os.path.isfile(te_segment_path(compiled_dir, n)) for n, _, _ in TE_SEGMENTS):
        logger.info(f"Text encoder already compiled at {out_dir}, skipping.")
        return

    ids, mask = tokenize_for_klein(tokenizer, "A cat holding a sign that says hello world")
    full = text_encoder.model
    segs = {}
    for name, start, end in TE_SEGMENTS:
        seg = build_qwen3_segment(full, start, end, with_embed=(start == 0))
        segs[name] = (_IdsSegment(seg) if start == 0 else _HiddenSegment(seg)).eval()

    with torch.no_grad():
        h9 = segs["seg_a"](ids, mask)
        h18 = segs["seg_b"](h9, mask)
    examples = {"seg_a": (ids, mask), "seg_b": (h9, mask), "seg_c": (h18, mask)}

    report = {"compiler_args": compiler_args, "segments": {}}
    for name, _, _ in TE_SEGMENTS:
        started = time.perf_counter()
        traced = torch_neuronx.trace(
            segs[name],
            examples[name],
            compiler_workdir=os.path.join(out_dir, f"compiler_workdir_{name}"),
            compiler_args=compiler_args,
        )
        torch.jit.save(traced, te_segment_path(compiled_dir, name))
        report["segments"][name] = {"compile_seconds": time.perf_counter() - started}
        logger.info(f"Compiled text encoder {name} in {report['segments'][name]['compile_seconds']:.1f}s")
        del traced
    with open(os.path.join(out_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def tokenize_for_klein(tokenizer, prompt: str, max_length: int = TE_MAX_LEN):
    """Same tokenization as Flux2KleinPipeline._get_qwen3_prompt_embeds."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
    )
    return inputs["input_ids"], inputs["attention_mask"]


class NeuronKleinTextEncoder(nn.Module):
    """Drop-in for the pipeline's `text_encoder` (Qwen3ForCausalLM).

    Implements the slice of the HF API used by _get_qwen3_prompt_embeds:
    `dtype`, `device`, and `forward(input_ids, attention_mask, ...)` returning
    an object whose `.hidden_states[k]` is defined for k in 9/18/27.
    """

    def __init__(self, compiled_dir: str, placement: Optional[Dict[str, int]] = None):
        super().__init__()
        placement = placement or TE_DEFAULT_PLACEMENT
        self.segs = {}
        for name, _, _ in TE_SEGMENTS:
            traced = torch.jit.load(te_segment_path(compiled_dir, name))
            _set_cores(traced, start_nc=placement[name], nc_count=1)
            self.segs[name] = traced
        self.placement = dict(placement)

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids.shape[-1] != TE_MAX_LEN:
            raise ValueError(f"Neuron text encoder was traced for seq_len={TE_MAX_LEN}, got {input_ids.shape}")
        h9 = self.segs["seg_a"](input_ids, attention_mask)
        h18 = self.segs["seg_b"](h9, attention_mask)
        h27 = self.segs["seg_c"](h18, attention_mask)
        return SimpleNamespace(hidden_states={9: h9, 18: h18, 27: h27}, last_hidden_state=h27)


# ---------------------------------------------------------------------------
# VAE decoder
# ---------------------------------------------------------------------------


class Fp32GroupNorm(nn.Module):
    """GroupNorm evaluated in FP32 (same fix as NxDI FLUX.1 PatchedGroupNorm)."""

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.detach().float(), requires_grad=False)
        self.bias = nn.Parameter(gn.bias.detach().float(), requires_grad=False)

    def forward(self, x):
        return F.group_norm(x.float(), self.num_groups, self.weight, self.bias, self.eps).to(x.dtype)


def patch_group_norms(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, Fp32GroupNorm(child))
            count += 1
        else:
            count += patch_group_norms(child)
    return count


class _VaeDecode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


def vae_path(compiled_dir: str) -> str:
    return os.path.join(compiled_dir, "vae_decoder", "model.pt")


def compile_vae_decoder(vae, compiled_dir: str, height: int, width: int, compiler_args: str = VAE_COMPILER_ARGS):
    """Trace vae.decode for a fixed resolution. `vae` is the CPU AutoencoderKLFlux2."""
    import torch_neuronx

    path = vae_path(compiled_dir)
    if os.path.isfile(path):
        logger.info(f"VAE decoder already compiled at {path}, skipping.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    vae = copy.deepcopy(vae).eval()
    n_patched = patch_group_norms(vae)
    spatial = 2 ** (len(vae.config.block_out_channels) - 1)  # 8
    latent = torch.randn(
        1, vae.config.latent_channels, height // spatial, width // spatial, dtype=torch.bfloat16
    )
    started = time.perf_counter()
    traced = torch_neuronx.trace(
        _VaeDecode(vae),
        (latent,),
        compiler_workdir=os.path.join(compiled_dir, "vae_decoder", "compiler_workdir"),
        compiler_args=compiler_args,
    )
    torch.jit.save(traced, path)
    with open(os.path.join(compiled_dir, "vae_decoder", "compile_report.json"), "w") as f:
        json.dump(
            {
                "compiler_args": compiler_args,
                "group_norms_fp32": n_patched,
                "latent_shape": list(latent.shape),
                "compile_seconds": time.perf_counter() - started,
            },
            f,
            indent=2,
        )
    logger.info(f"Compiled VAE decoder in {time.perf_counter() - started:.1f}s")


class NeuronVaeDecode:
    """Callable replacing `pipe.vae.decode`; the rest of the VAE object (bn stats,
    config) stays on CPU because the pipeline reads them for latent denorm."""

    def __init__(self, compiled_dir: str, core: int = VAE_DEFAULT_CORE):
        self.traced = torch.jit.load(vae_path(compiled_dir))
        _set_cores(self.traced, start_nc=core, nc_count=1)
        self.core = core

    def __call__(self, z, return_dict=False, **kwargs):
        image = self.traced(z.to(torch.bfloat16))
        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=image)
        return (image,)
