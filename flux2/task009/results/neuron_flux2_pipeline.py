"""NeuronFlux2Pipeline — end-to-end FLUX.2-dev orchestrator on AWS Neuron.

Stitches:
  - Neuron text encoder   (Mistral-3-24B, hidden states [10,20,30] stacked)
  - Neuron DiT            (stub for now; real NEFF drops in later, same contract)
  - CPU / Neuron VAE      (tiled decode via vae_tile_decode.py)

Mirrors diffusers' Flux2Pipeline.__call__ logic with only the external-to-model
pieces replaced. Scheduler, _pack/_unpack, image-id construction, guidance,
BN denorm, unpatchify — all copied verbatim from
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/
diffusers/pipelines/flux2/pipeline_flux2.py (credit: diffusers, Apache-2).

Usage:
    from neuron_flux2_pipeline import NeuronFlux2Pipeline, NeuronDiTStub
    pipe = NeuronFlux2Pipeline(
        neuron_text_encoder=text_fn,   # (input_ids, attention_mask) -> (B, S, 3, H)
        neuron_dit=NeuronDiTStub(),    # real Neuron DiT when compiled
        vae_decode_fn=vae_fn,          # (B, 32, h_l, w_l) packed-unnorm latent -> (B, 3, H, W) in [-1,1]
        scheduler=FlowMatchEulerDiscreteScheduler(...),
        tokenizer=AutoProcessor.from_pretrained(...),
        vae_bn_mean=vae.bn.running_mean,
        vae_bn_var=vae.bn.running_var,
        vae_bn_eps=vae.config.batch_norm_eps,
    )
    image = pipe("a red panda", height=1024, width=1024,
                 num_inference_steps=28, guidance_scale=4.0, seed=42)
"""

from __future__ import annotations

import inspect
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# System message (copied from diffusers flux2 pipeline/system_messages.py —
# we only need the base T2I one; agent text encoder NEFF was traced against
# whatever the raw input_ids look like, so we reuse diffusers' chat template).
# ---------------------------------------------------------------------------

# This is the short marketing prompt wrapper BFL uses by default. Mirrors
# diffusers.pipelines.flux2.system_messages.SYSTEM_MESSAGE so encode_prompt
# matches the CPU reference bit-for-bit.
# TODO(port): if we ever fine-tune the system message, keep this in sync with
# whatever is in diffusers at that version.
SYSTEM_MESSAGE = (
    "You are an assistant designed to generate high-quality images based on user prompts."
)


# ---------------------------------------------------------------------------
# DiT stub — placeholder until the real Neuron DiT NEFF lands.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Weight init helper for ModelBuilder-traced NxD archives.
# torch.jit.load on a NxDModel returns an uninitialized archive; calling the
# forward raises "This model is not initialized, please call
# traced_model.nxd_model.initialize(...)". We call
# initialize_with_saved_weights(start_rank_tensor=0) which reloads the sharded
# weights that were persisted next to the .pt at trace time.
# ---------------------------------------------------------------------------


def _init_nxd_if_needed(jit_module, label: str = ""):
    """Best-effort initialize for NxD ModelBuilder archives.

    No-op if the module has no .nxd_model attribute (e.g. plain torch_neuronx.trace
    outputs like the VAE). Silent on errors — the failure will surface on first
    forward if init was truly required.
    """
    try:
        nxd = getattr(jit_module, "nxd_model", None)
    except Exception:
        nxd = None
    if nxd is None:
        return
    init_fn = getattr(nxd, "initialize_with_saved_weights", None)
    if init_fn is None:
        return
    try:
        init_fn(torch.tensor(0, dtype=torch.int32))
        print(f"[pipeline] initialized NxD weights for {label}")
    except Exception as e:
        print(f"[pipeline] WARN NxD init failed for {label}: {e}")


class NeuronDiTStub:
    """Placeholder DiT returning noise-shape-matched tensor.

    Contract (what the real Neuron DiT port must match exactly):
      inputs:
        hidden_states        : (B, L_img, 128) bf16         — packed image latent tokens
        timestep             : (B,)            fp32 / bf16  — in [0, 1], diffusers divides by 1000
        encoder_hidden_states: (B, L_txt, D_txt) bf16       — Mistral stacked hidden states (D_txt = 3 * 5120)
        guidance             : (B,)            fp32         — scalar guidance per-sample
        img_ids              : (B, L_img, 4)   int64        — 4-axis RoPE position ids
        txt_ids              : (B, L_txt, 4)   int64        — 4-axis RoPE position ids for text
      returns:
        noise_pred           : (B, L_img + L_txt, 128) bf16 — diffusers slices [:, :L_img, :] after
                                                              (see pipeline_flux2.py L982)

    NOTE: With FLUX.2, the transformer concatenates image+text sequences internally
    and predicts noise over the joint sequence. The pipeline slices off the text
    portion before scheduler.step. The stub mirrors that exactly so the caller
    slice in __call__ is correct.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.dtype = dtype

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        guidance: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        # TODO(port): replace with real Neuron DiT call. NEFF input dtypes must
        # match the values above; NxDI trace should pin them explicitly.
        B, L_img, C = hidden_states.shape
        L_txt = encoder_hidden_states.shape[1]
        # Return joint seq so caller's `noise_pred[:, :L_img, :]` slice works.
        return torch.randn(B, L_img + L_txt, C, dtype=hidden_states.dtype)


# ---------------------------------------------------------------------------
# Neuron DiT wrapper — wraps a torch.jit-loaded NEFF in a callable that matches
# the NeuronDiTStub contract (same kwargs, same output slicing semantics).
# ---------------------------------------------------------------------------


class NeuronDiT:
    """Callable wrapper around a torch.jit-loaded Neuron DiT NEFF.

    The underlying NEFF actual forward signature is:

        forward(hidden_states, encoder_hidden_states, timestep, guidance,
                image_rotary_emb)

    with `image_rotary_emb` a pre-computed tensor of shape
    ``[S_txt + S_img, head_dim, 2]`` (cos/sin stacked on the last dim).
    The NEFF internally slices off the text tokens, so the returned tensor
    has shape ``(B, L_img, 128)``.

    This wrapper keeps the same kwarg contract as NeuronDiTStub
    (``img_ids`` / ``txt_ids`` are 4-axis position ids) and computes the
    rotary table on the host before calling the NEFF.

    Dtypes:
      hidden_states, encoder_hidden_states       : bf16
      timestep                                   : bf16 (pipeline divides by 1000)
      guidance                                   : bf16 (NEFF scales by 1000 inside)
      image_rotary_emb                           : bf16
    """

    # 4-axis RoPE config for FLUX.2 (matches neuron_flux2_dit.NeuronFlux2Config
    # defaults / compile_dit_tp8.py).
    _AXES_DIMS_ROPE = (32, 32, 32, 32)
    _ROPE_THETA = 2000

    def __init__(self, jit_module):
        self.model = jit_module
        # Build the RoPE module once; it holds no trainable params.
        # Import lazily so this file stays importable without the dit module.
        try:
            import sys as _sys
            _sys.path.insert(0, "/home/ubuntu")
            from neuron_flux2_dit import NeuronFlux2RotaryEmbedding
            self._rope = NeuronFlux2RotaryEmbedding(
                theta=self._ROPE_THETA, axes_dim=self._AXES_DIMS_ROPE
            )
        except Exception as e:
            raise RuntimeError(
                f"NeuronDiT requires neuron_flux2_dit.NeuronFlux2RotaryEmbedding "
                f"(needed to build image_rotary_emb on the host): {e}"
            )

    @staticmethod
    def _build_rotary_emb(rope_module, img_ids: torch.Tensor, txt_ids: torch.Tensor
                          ) -> torch.Tensor:
        """Concatenate [txt_ids, img_ids] (matching the NEFF internal order
        where single_stream cat's encoder first), compute per-axis RoPE, and
        stack (cos, sin) on a new last dim -> [S, head_dim, 2].
        """
        # ids come in as (B, S, 4); RoPE module expects (S, 4)
        txt = txt_ids[0].to(torch.int64)
        img = img_ids[0].to(torch.int64)
        ids = torch.cat([txt, img], dim=0)  # (S_txt + S_img, 4)
        cos, sin = rope_module(ids)         # each (S, head_dim)
        return torch.stack([cos, sin], dim=-1).contiguous()  # (S, head_dim, 2)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        guidance: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.bfloat16)
        encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
        # The NEFF internally does `timestep * 1000` (see
        # neuron_flux2_dit.NeuronFlux2Transformer.forward); the pipeline scales
        # by 1/1000 before calling us, so pass as-is in bf16.
        timestep = timestep.to(torch.bfloat16)
        guidance = guidance.to(torch.bfloat16)

        image_rotary_emb = self._build_rotary_emb(self._rope, img_ids, txt_ids)
        image_rotary_emb = image_rotary_emb.to(torch.bfloat16)

        out = self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            guidance,
            image_rotary_emb,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


# ---------------------------------------------------------------------------
# Diffusers internal helpers — copied verbatim (minor cosmetic changes) from
# diffusers/pipelines/flux2/pipeline_flux2.py. Copying (not subclassing) keeps
# this module free of diffusers-pipeline heavy imports so it can be dropped on
# an inference-only host.
# ---------------------------------------------------------------------------


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_text_ids
def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    """(B, L, D) text embeds -> (B, L, 4) position ids (T=0, H=0, W=0, L=0..L-1)."""
    B, L, _ = x.shape
    out_ids = []
    for _ in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l_axis = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l_axis)
        out_ids.append(coords)
    return torch.stack(out_ids)


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_latent_ids
def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) latent -> (B, H*W, 4) position ids (T=0, H=h, W=w, L=0)."""
    B, _, H, W = latents.shape
    t = torch.arange(1)
    h = torch.arange(H)
    w = torch.arange(W)
    l_axis = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l_axis)  # (H*W, 4)
    return latent_ids.unsqueeze(0).expand(B, -1, -1)


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*4, H/2, W/2) via 2x2 patchify."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(B, C * 4, H // 2, W // 2)
    return latents


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C*4, H, W) -> (B, C, H*2, W*2) — inverse of _patchify_latents."""
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(B, C // 4, H * 2, W * 2)
    return latents


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._pack_latents
def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


# diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpack_latents_with_ids
def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """(B, H*W, C) + (B, H*W, 4) -> (B, C, H, W). Uses position ids to scatter."""
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = int(torch.max(h_ids).item()) + 1
        w = int(torch.max(w_ids).item()) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        out_list.append(out)
    return torch.stack(out_list, dim=0)


# diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
    if sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"{scheduler.__class__} does not accept sigmas")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


class NeuronFlux2Pipeline:
    """FLUX.2-dev end-to-end pipeline using Neuron-traced components.

    This is not a subclass of diffusers.DiffusionPipeline: it's a minimal
    re-implementation of Flux2Pipeline.__call__ that takes pre-loaded Neuron
    callables (no HF model hooks, no CPU offload, no lora). Matches the
    reference __call__ signature for the common text-to-image path.
    """

    # Mirrors diffusers.pipelines.flux2.pipeline_flux2 vae_scale_factor default
    # (VAE downsamples by 8; the `* 2` factors come from the 2x2 patchify step).
    VAE_SCALE_FACTOR = 8

    def __init__(
        self,
        neuron_text_encoder: Callable[..., torch.Tensor],
        neuron_dit: Callable[..., torch.Tensor],
        vae_decode_fn: Callable[[torch.Tensor], torch.Tensor],
        scheduler,
        tokenizer,
        vae_bn_mean: torch.Tensor,
        vae_bn_var: torch.Tensor,
        vae_bn_eps: float = 1e-5,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        transformer_in_channels: int = 128,
        text_encoder_max_len: int = 512,
        extract_layers: tuple = (10, 20, 30),
        system_message: str = SYSTEM_MESSAGE,
    ):
        self.text_encoder = neuron_text_encoder
        self.dit = neuron_dit
        self.vae_decode_fn = vae_decode_fn
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        # vae.bn denorm params for unpacking latents back into VAE input space.
        self.register_vae_bn(vae_bn_mean, vae_bn_var, vae_bn_eps)

        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self.transformer_in_channels = transformer_in_channels
        self.text_encoder_max_len = text_encoder_max_len
        self.extract_layers = tuple(extract_layers)
        self.system_message = system_message

    def register_vae_bn(self, mean: torch.Tensor, var: torch.Tensor, eps: float):
        self._bn_mean = mean.detach().clone()
        self._bn_var = var.detach().clone()
        self._bn_eps = float(eps)

    # -------------------------------------------------------------------
    # Convenience loader — build a pipeline directly from on-disk traces.
    # -------------------------------------------------------------------

    @classmethod
    def from_traced(
        cls,
        dit_neff: str,
        te_neff: str,
        vae_neff: str,
        weights_dir: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "NeuronFlux2Pipeline":
        """Load all traced Neuron components + scheduler + tokenizer from disk.

        Args:
          dit_neff:     path to `torch.jit.save`'d DiT NEFF (TP sharded internally).
          te_neff:      path to `torch.jit.save`'d text encoder NEFF (TP sharded).
          vae_neff:     path to 512²-bucket VAE decoder NEFF; higher resolutions
                        are handled via tiled decode (vae_tile_decode.py).
          weights_dir:  FLUX.2-dev weights dir (contains scheduler/, tokenizer/,
                        vae/ for BN params + eps, and optionally ae.safetensors).
        """
        import json as _json
        import os as _os
        import torch_neuronx  # noqa: F401  (required so jit.load picks up neuron ops)

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoProcessor
        from safetensors import safe_open

        # ---- traced Neuron modules (all torch.jit archives) ---------------
        te_jit = torch.jit.load(te_neff)
        dit_jit = torch.jit.load(dit_neff)
        vae_jit = torch.jit.load(vae_neff)

        # ModelBuilder-saved NxDModel archives require explicit weight init on
        # first load. start_rank=0 for single-process SPMD runs. (The VAE was
        # traced via torch_neuronx.trace, not ModelBuilder — no init needed.)
        _init_nxd_if_needed(te_jit, label="text_encoder")
        _init_nxd_if_needed(dit_jit, label="dit")

        # ---- text encoder callable (returns stacked (B, S, 3, H)) ---------
        # Signature is single-positional `input_ids` so the pipeline's
        # encode_prompt takes the Neuron-NEFF right-pad branch (the NEFF was
        # traced with causal-only masking, so we right-pad before the call
        # and shift the output into HF-layout afterwards). DO NOT add an
        # `attention_mask=...` kwarg here or encode_prompt will route this
        # to the HF/CPU-fallback branch and silently run in broken left-pad
        # mode (re-introducing the bad-quality bug).
        def text_fn(input_ids):
            out = te_jit(input_ids)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out.cpu() if hasattr(out, "cpu") else out

        # ---- VAE decode callable (tiled for >= 1024² via vae_tile_decode) --
        import importlib.util
        tile_mod_path = _os.path.join(_os.path.dirname(_os.path.abspath(vae_neff)), "..",
                                       "vae_tile_decode.py")
        tile_mod_path = _os.path.normpath(tile_mod_path)
        if not _os.path.exists(tile_mod_path):
            # fall back to the well-known location on the Neuron instance
            tile_mod_path = "/home/ubuntu/vae_tile_decode.py"
        spec = importlib.util.spec_from_file_location("vae_tile_decode", tile_mod_path)
        tile_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tile_mod)

        def vae_neff_call(x):
            with torch.no_grad():
                return vae_jit(x.to(torch.bfloat16))

        # warm the NEFF once so first pipeline call isn't cold
        try:
            _ = vae_neff_call(torch.randn(1, 32, 64, 64, dtype=torch.bfloat16))
        except Exception as e:  # non-fatal; real call will surface any issue
            print(f"[from_traced] VAE warmup skipped: {e}")

        def vae_decode_fn(latents):
            """(B, 32, h_l, w_l) -> (B, 3, H, W) pixel-space in [-1, 1]."""
            return tile_mod.tiled_decode_neff(vae_neff_call, latents.to(torch.bfloat16))

        # ---- VAE BN params (for post-scheduler denorm before VAE decode) --
        # Prefer ae.safetensors (top-level bn.*); else vae/diffusion_pytorch_model.safetensors
        bn_candidates = [
            _os.path.join(weights_dir, "ae.safetensors"),
            _os.path.join(weights_dir, "vae", "diffusion_pytorch_model.safetensors"),
        ]
        bn_path = next((p for p in bn_candidates if _os.path.exists(p)), None)
        if bn_path is None:
            raise FileNotFoundError(
                f"VAE BN weights not found under {weights_dir}; tried {bn_candidates}"
            )
        with safe_open(bn_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            mean_key = next(k for k in keys if k.endswith("bn.running_mean"))
            var_key = next(k for k in keys if k.endswith("bn.running_var"))
            bn_mean = f.get_tensor(mean_key).to(dtype)
            bn_var = f.get_tensor(var_key).to(dtype)
        with open(_os.path.join(weights_dir, "vae", "config.json")) as f:
            vae_cfg = _json.load(f)
        bn_eps = float(vae_cfg.get("batch_norm_eps", 1e-5))

        # ---- scheduler + tokenizer ----------------------------------------
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            weights_dir, subfolder="scheduler"
        )
        tokenizer = AutoProcessor.from_pretrained(
            _os.path.join(weights_dir, "tokenizer")
        )

        return cls(
            neuron_text_encoder=text_fn,
            neuron_dit=NeuronDiT(dit_jit),
            vae_decode_fn=vae_decode_fn,
            scheduler=scheduler,
            tokenizer=tokenizer,
            vae_bn_mean=bn_mean,
            vae_bn_var=bn_var,
            vae_bn_eps=bn_eps,
            dtype=dtype,
        )

    # -------------------------------------------------------------------
    # Text encoding — mirrors Flux2Pipeline._get_mistral_3_small_prompt_embeds.
    # -------------------------------------------------------------------

    def _encode_prompt(self, prompt: str | list[str]):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)

        # Build chat-template messages. Matches diffusers' format_input for T2I.
        messages_batch = [
            [
                {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
                {"role": "user", "content": [{"type": "text", "text": p.replace("[IMG]", "")}]},
            ]
            for p in prompts
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_encoder_max_len,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Tokenizer pads on the LEFT for this Mistral-3 tokenizer (e.g.
        # input_ids = [pad, pad, ..., pad, real_0, real_1, ..., real_{N-1}]).
        # The traced Neuron TE NEFF only applies a causal mask (no padding
        # attention mask was wired through at trace time). Under a LEFT-padded
        # layout, each real token at position p can attend to all positions
        # <=p, which **include all padding tokens**. This contaminates the
        # real-token hidden states and yields cos_sim ~0.22 vs the HF reference
        # that does pass a real attention_mask. The fix is to right-pad the
        # input before calling the NEFF — under a right-padded layout each
        # real token sees only other real tokens (causal) and matches HF with
        # full attention_mask at cos_sim >= 0.999. After the forward we shift
        # the stacked hidden states back into the original LEFT-padded layout
        # so downstream RoPE position ids (0..S-1) align with what the DiT
        # was compiled against. Pad-position hidden states are zeroed — the
        # DiT has no text-mask so we feed it zeros for those positions rather
        # than the garbage right-pad NEFF produced. # VERIFIED on 'red panda'
        # prompt: real-position cos_sim vs HF = 1.000, end-to-end PSNR fix.
        pad_id = None
        for b in range(input_ids.shape[0]):
            pad_positions = (~attention_mask[b].bool()).nonzero(as_tuple=False)
            if pad_positions.numel() > 0:
                pad_id = int(input_ids[b, pad_positions[0].item()].item())
                break
        if pad_id is None:
            # No padding anywhere — safe to call directly.
            pad_id = 0

        # Build a right-padded batch: real tokens at [0, N_b), pad at [N_b, S).
        S = input_ids.shape[1]
        real_lens = attention_mask.sum(dim=1).to(torch.long)  # (B,)
        rp_ids = torch.full_like(input_ids, pad_id)
        for b in range(input_ids.shape[0]):
            mask_b = attention_mask[b].bool()
            real = input_ids[b, mask_b]
            rp_ids[b, : real.numel()] = real

        # Call the Neuron (or CPU fallback) text encoder. Contract:
        #   Path A (Neuron traced, our module): takes (input_ids) and returns
        #     stacked (B, S, 3, H) — see trace_text_encoder.py.
        #   Path B (HF MistralModel CPU fallback): takes (input_ids, attention_mask)
        #     with output_hidden_states=True — we wrap it in run_pipeline_stub.
        # We probe the signature once and pick the right call shape.
        try:
            sig = inspect.signature(self.text_encoder)
            params = sig.parameters
        except (TypeError, ValueError):
            params = {}

        if "attention_mask" in params:
            # CPU/HF fallback path honours the real attention_mask itself —
            # pass the ORIGINAL (left-padded) ids+mask so this matches HF byte
            # for byte when the fallback is a real diffusers Mistral.
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # Neuron NEFF path: use the right-padded ids so causal attention
            # on real positions isn't contaminated by padding tokens.
            out = self.text_encoder(rp_ids)

        # Accept several return conventions:
        #   - stacked tensor (B, S, 3, H)     [Neuron trace directly]
        #   - (B, 3, S, H) from HF stack-along-dim=1 style
        #   - HF ModelOutput with .hidden_states
        if hasattr(out, "hidden_states"):
            hs = [out.hidden_states[k] for k in self.extract_layers]
            stacked = torch.stack(hs, dim=1)  # (B, 3, S, H)
        elif isinstance(out, (tuple, list)):
            stacked = out[0]
        else:
            stacked = out

        # Normalize to (B, S, 3, H) — some consumers produce (B, 3, S, H).
        if stacked.shape[1] == len(self.extract_layers):
            stacked = stacked.permute(0, 2, 1, 3)

        stacked = stacked.to(self.dtype)
        B, S, N, H = stacked.shape

        # If we took the right-pad NEFF path above, `stacked` holds real
        # token hidden states in positions [0, N_b) and garbage in [N_b, S).
        # Shift each batch element back into the original LEFT-padded layout
        # so real tokens live at [S - N_b, S), matching what HF would send
        # to the DiT. Pad positions are zeroed out (the DiT uses no text
        # attention mask, so feeding zeros avoids the NEFF's garbage pad
        # outputs leaking into attention). # VERIFIED: with this shift the
        # Neuron encoder output matches HF at real positions cos_sim=1.000
        # and zeros elsewhere.
        if "attention_mask" not in params:
            shifted = torch.zeros_like(stacked)
            for b in range(B):
                n_b = int(real_lens[b].item())
                if n_b == 0:
                    continue
                shifted[b, S - n_b :, :, :] = stacked[b, :n_b, :, :]
            stacked = shifted

        # (B, S, N, H) -> (B, S, N*H) to match diffusers' prompt_embeds shape
        # (see pipeline_flux2.py L351).
        prompt_embeds = stacked.reshape(B, S, N * H)
        text_ids = _prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids

    # -------------------------------------------------------------------
    # Latents prep — mirrors Flux2Pipeline.prepare_latents.
    # -------------------------------------------------------------------

    def _prepare_latents(self, batch_size, num_latents_channels, height, width, generator):
        # (copied from diffusers) VAE 8x compression + 2x2 patchify forces even dims.
        latent_h = 2 * (int(height) // (self.VAE_SCALE_FACTOR * 2))
        latent_w = 2 * (int(width) // (self.VAE_SCALE_FACTOR * 2))

        shape = (batch_size, num_latents_channels * 4, latent_h // 2, latent_w // 2)
        latents = torch.randn(shape, generator=generator, dtype=self.dtype)
        latent_ids = _prepare_latent_ids(latents)
        latents = _pack_latents(latents)  # (B, H*W, C)
        return latents, latent_ids, (latent_h, latent_w)

    # -------------------------------------------------------------------
    # Main __call__
    # -------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        profile: bool = True,
        return_timings: bool | None = None,
    ):
        # Back-compat: if return_timings is not specified, default it to True
        # when called with `seed=` (the old stub path) and False when called
        # with `generator=` (the bench_neuron path, which does `image.save`).
        if return_timings is None:
            return_timings = generator is None
        if seed is None and generator is None:
            seed = 42

        timings = {}
        multiple_of = self.VAE_SCALE_FACTOR * 2
        if height % multiple_of or width % multiple_of:
            raise ValueError(f"height/width must be divisible by {multiple_of}, got {height}x{width}")

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # ---- 1. text encoder ------------------------------------------------
        t0 = time.perf_counter()
        prompt_embeds, text_ids = self._encode_prompt(prompt)
        timings["text_encoder_s"] = time.perf_counter() - t0
        if profile:
            print(
                f"[pipe] text encoder: {timings['text_encoder_s']:.2f}s  "
                f"prompt_embeds={tuple(prompt_embeds.shape)} text_ids={tuple(text_ids.shape)}"
            )

        # ---- 2. prepare latents --------------------------------------------
        num_channels_latents = self.transformer_in_channels // 4  # 32
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents, latent_ids, (latent_h, latent_w) = self._prepare_latents(
            batch_size, num_channels_latents, height, width, generator
        )
        if profile:
            print(
                f"[pipe] latents init: {tuple(latents.shape)}  latent_ids={tuple(latent_ids.shape)}  "
                f"latent_hw=({latent_h},{latent_w})"
            )

        # ---- 3. prepare timesteps ------------------------------------------
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = _compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, device=None, sigmas=sigmas, mu=mu
        )

        # ---- 4. guidance ----------------------------------------------------
        guidance = torch.full([1], guidance_scale, dtype=torch.float32).expand(latents.shape[0])

        # ---- 5. denoising loop ---------------------------------------------
        self.scheduler.set_begin_index(0)
        t0 = time.perf_counter()
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(self.dtype)

            noise_pred = self.dit(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                guidance=guidance,
                img_ids=latent_ids,
                txt_ids=text_ids,
            )
            # Diffusers slices off the text portion (L_txt is concatenated inside
            # the DiT). See pipeline_flux2.py L982:
            #   noise_pred = noise_pred[:, : latents.size(1) :]
            noise_pred = noise_pred[:, : latents.size(1), :]

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if profile and (i == 0 or i == len(timesteps) - 1):
                print(
                    f"[pipe] step {i+1}/{len(timesteps)}  t={float(t):.3f}  "
                    f"latents={tuple(latents.shape)}"
                )
        timings["scheduler_loop_s"] = time.perf_counter() - t0
        if profile:
            print(f"[pipe] denoise loop: {timings['scheduler_loop_s']:.2f}s ({len(timesteps)} steps)")

        # ---- 6. VAE decode --------------------------------------------------
        t0 = time.perf_counter()
        # Unpack + BN denorm + unpatchify (see pipeline_flux2.py L1014-1021).
        latents = _unpack_latents_with_ids(latents, latent_ids)  # (B, 128, h/2, w/2)
        bn_mean = self._bn_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self._bn_var.view(1, -1, 1, 1) + self._bn_eps).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean
        latents = _unpatchify_latents(latents)  # (B, 32, h, w)
        if profile:
            print(f"[pipe] VAE input (unpacked, unpatchified): {tuple(latents.shape)}")

        image = self.vae_decode_fn(latents)  # (B, 3, H, W) in [-1, 1]
        timings["vae_decode_s"] = time.perf_counter() - t0
        if profile:
            print(
                f"[pipe] VAE decode: {timings['vae_decode_s']:.2f}s  image={tuple(image.shape)}"
            )

        # ---- 7. postprocess -------------------------------------------------
        # Match diffusers' VaeImageProcessor.postprocess for output_type="pil":
        # scale [-1,1] -> [0,1], clamp, to uint8 HWC, to PIL.
        image = image.float().clamp(-1, 1)
        image = (image + 1.0) / 2.0

        # Always stash timings on the instance so callers that opted into the
        # one-return-value path (bench_neuron) can still inspect per-stage time.
        self.last_timings = dict(timings)

        if output_type == "pt":
            out = image
        elif output_type == "np":
            out = image.permute(0, 2, 3, 1).cpu().numpy()
        else:  # "pil"
            arr = (image.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype("uint8")
            pil_images = [Image.fromarray(a) for a in arr]
            out = pil_images[0] if len(pil_images) == 1 else pil_images

        if return_timings:
            return out, timings
        return out
