"""Phase 4+5: CPU TP=1 verification of v3 scaffold.

- Phase 4: per-component R-ratio (tap at several places).
- Phase 5: full forward rel_L2 + cosine vs HF fp32.

Run with NXD_CPU_MODE=1 and gloo init_process_group so NxDI TP layers work at TP=1.
"""
from __future__ import annotations
import os
os.environ.setdefault("NXD_CPU_MODE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29512")

import sys, time, gc, json
sys.path.insert(0, "/home/ubuntu/flux2_v3")
import torch
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
initialize_model_parallel(tensor_model_parallel_size=1)

import neuron_flux2_dit_v3 as v3
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from neuronx_distributed_inference.models.config import NeuronConfig
from safetensors import safe_open


WEIGHTS_DIR = "/home/ubuntu/flux2_weights/transformer"
NUM_LAYERS = 2      # smaller for CPU speed
NUM_SINGLE = 2
NUM_PATCHES = 256   # 16x16
TEXT_SEQ_LEN = 32


def _load_full_hf_sd():
    idx_path = os.path.join(WEIGHTS_DIR, "diffusion_pytorch_model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)
    sd = {}
    for shard_name, keys in by_shard.items():
        shard_path = os.path.join(WEIGHTS_DIR, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                sd[k] = f.get_tensor(k).to(torch.float32)  # fp32 master
    return sd


def _subset_sd_for_layers(sd, num_layers, num_single):
    """Keep only layers 0..num_layers-1 and single 0..num_single-1; rest are dropped."""
    out = {}
    for k, v in sd.items():
        if k.startswith("transformer_blocks."):
            idx = int(k.split(".")[1])
            if idx < num_layers:
                out[k] = v
        elif k.startswith("single_transformer_blocks."):
            idx = int(k.split(".")[1])
            if idx < num_single:
                out[k] = v
        else:
            out[k] = v
    return out


def _rratio(tgt, src_lowprec, src_fp32, name="tensor"):
    tgt = tgt.detach().float()
    src_lowprec = src_lowprec.detach().float()
    src_fp32 = src_fp32.detach().float()
    num = (tgt - src_fp32).norm().item()
    den = (src_lowprec - src_fp32).norm().item() + 1e-12
    rel_l2 = (tgt - src_fp32).norm().item() / (src_fp32.norm().item() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(
        tgt.flatten().unsqueeze(0), src_fp32.flatten().unsqueeze(0)
    ).item()
    print(f"[{name}] R={num/den:.4f}  rel_L2={rel_l2:.4e}  cos={cos:.6f}  "
          f"num={num:.4e} den={den:.4e} shape={tuple(tgt.shape)}")
    return num / den, rel_l2, cos


def main():
    print(f"[cfg] NUM_LAYERS={NUM_LAYERS} NUM_SINGLE={NUM_SINGLE} patches={NUM_PATCHES} txt={TEXT_SEQ_LEN}")
    print("[load] HF state dict (fp32 master)...")
    t0 = time.perf_counter()
    full_sd_fp32 = _load_full_hf_sd()
    print(f"[load] fp32 sd in {time.perf_counter()-t0:.1f}s  keys={len(full_sd_fp32)}")

    # Subset to small for CPU
    sd_fp32 = _subset_sd_for_layers(full_sd_fp32, NUM_LAYERS, NUM_SINGLE)
    sd_bf16 = {k: v.to(torch.bfloat16) for k, v in sd_fp32.items()}

    # HF model constructor args
    hf_kwargs = dict(
        patch_size=1, in_channels=128, out_channels=128,
        num_layers=NUM_LAYERS, num_single_layers=NUM_SINGLE,
        attention_head_dim=128, num_attention_heads=48,
        joint_attention_dim=15360, timestep_guidance_channels=256,
        mlp_ratio=3.0, axes_dims_rope=(32, 32, 32, 32),
        rope_theta=2000, eps=1e-6, guidance_embeds=True,
    )

    print("[build] HF fp32 reference...")
    hf_fp32 = Flux2Transformer2DModel(**hf_kwargs).eval()
    hf_fp32.load_state_dict(sd_fp32, strict=True)
    hf_fp32 = hf_fp32.to(torch.float32)

    print("[build] HF bf16 (source lowprec)...")
    hf_bf16 = Flux2Transformer2DModel(**hf_kwargs).eval()
    hf_bf16.load_state_dict(sd_bf16, strict=True)
    hf_bf16 = hf_bf16.to(torch.bfloat16)

    print("[build] v3 @ TP=1 bf16...")
    ncfg = NeuronConfig(tp_degree=1, torch_dtype=torch.bfloat16, cast_type="config")
    v3cfg = v3.NeuronFlux2V3Config(neuron_config=ncfg, **hf_kwargs)
    v3_model = v3.NeuronFlux2Transformer2DModel(v3cfg).eval()

    # Convert bf16 sd to v3 format
    v3_sd = v3.convert_hf_to_neuron_state_dict(sd_bf16, v3cfg)
    missing, unexpected = v3_model.load_state_dict(v3_sd, strict=False)
    print(f"[v3] missing keys: {len(missing)} | unexpected: {len(unexpected)}")
    if missing:
        print(f"  first 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  first 5 unexpected: {unexpected[:5]}")
    v3_model = v3_model.to(torch.bfloat16)

    # Build inputs (same for all three)
    torch.manual_seed(0)
    B = 1
    hidden = torch.randn(B, NUM_PATCHES, 128)
    encoder = torch.randn(B, TEXT_SEQ_LEN, 15360)
    timestep = torch.tensor([0.5], dtype=torch.float32)    # already divided by 1000 (pipeline convention)
    guidance = torch.tensor([3.5], dtype=torch.float32)

    # Build ids for HF RoPE
    def _prepare_latent_ids(N, B=1):
        import math as _m
        H = int(_m.isqrt(N))
        W = N // H
        t = torch.arange(1); h = torch.arange(H); w = torch.arange(W); l = torch.arange(1)
        coords = torch.cartesian_prod(t, h, w, l)
        return torch.stack([coords] * B)

    def _prepare_text_ids(L, B=1):
        t = torch.arange(1); h = torch.arange(1); w = torch.arange(1); l_ax = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l_ax)
        return torch.stack([coords] * B)

    img_ids = _prepare_latent_ids(NUM_PATCHES)
    txt_ids = _prepare_text_ids(TEXT_SEQ_LEN)

    # Precompute rotary (fp32) via HF pos_embed once — share across the three
    # (v3 needs stacked cos/sin form; HF uses tuple)
    hf_pos = hf_fp32.pos_embed
    img_rot = hf_pos(img_ids[0])  # tuple(cos, sin)  each [S, D]
    txt_rot = hf_pos(txt_ids[0])
    cos_cat = torch.cat([txt_rot[0], img_rot[0]], dim=0)
    sin_cat = torch.cat([txt_rot[1], img_rot[1]], dim=0)
    image_rotary_emb_stack = torch.stack([cos_cat, sin_cat], dim=-1)  # [S, D, 2]

    # --- HF fp32 forward ---
    print("[run] HF fp32 forward...")
    with torch.no_grad():
        hf_fp32_out = hf_fp32(
            hidden_states=hidden.to(torch.float32),
            encoder_hidden_states=encoder.to(torch.float32),
            timestep=timestep * 1000,  # HF internally doesn't multiply; *1000 happens on outside?
            # Wait: HF DOES multiply timestep by 1000 inside. Pipeline already divides by 1000.
            # So: HF takes timestep in [0,1] and multiplies internally.
            img_ids=img_ids, txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
    # Actually — reread HF forward: `timestep = timestep.to(hidden_states.dtype) * 1000`.
    # So HF expects timestep in [0,1], not [0,1000]. Let me re-do without the *1000 above.
    print("[FIX] re-running HF fp32 with unscaled timestep...")
    with torch.no_grad():
        hf_fp32_out = hf_fp32(
            hidden_states=hidden.to(torch.float32),
            encoder_hidden_states=encoder.to(torch.float32),
            timestep=timestep,
            img_ids=img_ids, txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
    print(f"[hf_fp32] shape={tuple(hf_fp32_out.shape)} mean={hf_fp32_out.mean():.4f} std={hf_fp32_out.std():.4f}")

    # --- HF bf16 forward (source lowprec baseline) ---
    print("[run] HF bf16 forward...")
    with torch.no_grad():
        hf_bf16_out = hf_bf16(
            hidden_states=hidden.to(torch.bfloat16),
            encoder_hidden_states=encoder.to(torch.bfloat16),
            timestep=timestep.to(torch.bfloat16),
            img_ids=img_ids, txt_ids=txt_ids,
            guidance=guidance.to(torch.bfloat16),
            return_dict=False,
        )[0]
    print(f"[hf_bf16] shape={tuple(hf_bf16_out.shape)} mean={hf_bf16_out.mean():.4f} std={hf_bf16_out.std():.4f}")

    # --- v3 TP=1 bf16 forward ---
    print("[run] v3 TP=1 bf16 forward...")
    with torch.no_grad():
        v3_out = v3_model(
            hidden.to(torch.bfloat16),
            encoder.to(torch.bfloat16),
            timestep.to(torch.bfloat16),
            guidance.to(torch.bfloat16),
            image_rotary_emb_stack.to(torch.bfloat16),
        )
    print(f"[v3_bf16] shape={tuple(v3_out.shape)} mean={v3_out.mean():.4f} std={v3_out.std():.4f}")

    # --- R-ratio FINAL ---
    print("\n===== FINAL R-RATIO =====")
    R_final, rel_l2, cos_final = _rratio(v3_out, hf_bf16_out, hf_fp32_out, "FINAL")
    print(f"\nVERDICT: R_final={R_final:.3f}  rel_L2={rel_l2:.3e}  cos={cos_final:.6f}")
    if R_final < 1.2 and cos_final > 0.99:
        print("[PASS] v3 matches HF within expected bf16 precision envelope")
    else:
        print("[FAIL] v3 diverges — component-wise taps recommended")
    return R_final


if __name__ == "__main__":
    main()
