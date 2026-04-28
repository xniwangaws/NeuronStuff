"""Compile FLUX.2-dev DiT (Flux2Transformer2DModel) for Neuron at TP=8.

Uses neuronx_distributed.trace.model_builder.ModelBuilder — the same API the
text-encoder compile script uses. The DiT scaffold is imported from
neuron_flux2_dit.py (same directory). Weights come from
/home/ubuntu/flux2_weights/transformer/.

Output: /home/ubuntu/dit_traced/dit_tp8_1024.pt (torch.jit.save)

Usage:
    python compile_dit_tp8.py                    # 1024x1024 full compile
    python compile_dit_tp8.py --dry-run          # shape-check inputs + build
    python compile_dit_tp8.py --tp 4 --seq-img 1024   # smaller scale for smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch


# ---- trn2 / LNC=2 env (must be set BEFORE torch_neuronx touches anything) ----
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
# Physical cores 0-15 (16 cores) = 8 logical cores at LNC=2 = TP=8.
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-15")


DEFAULT_WEIGHTS_DIR = "/home/ubuntu/flux2_weights/transformer"
DEFAULT_OUT_DIR = "/home/ubuntu/dit_traced"
DEFAULT_COMPILE_WORKDIR = "/home/ubuntu/dit_compile_workdir"


# =============================================================================
# Config helpers.
# =============================================================================

def _load_hf_config(weights_dir: str) -> dict:
    with open(os.path.join(weights_dir, "config.json")) as f:
        return json.load(f)


# =============================================================================
# State-dict loader.
# =============================================================================

def _build_sd_loader(weights_dir: str, tp_degree: int):
    """
    Returns a checkpoint_loader callable that ModelBuilder invokes per-rank
    (inside an XLA subprocess) to materialize a flat HF-layout state_dict
    converted + (when tp>1) row-interleaved for fused SwiGLU projections.

    Loads all 7 shards into CPU bf16 and applies the scaffold's
    `convert_hf_to_neuron_state_dict` which does the interleave internally
    when `config.neuron_config.tp_degree > 1` (the interleave is a strict
    no-op at tp=1).
    """
    # We import here (not module scope) because this function must be pickled
    # and executed per-rank by ModelBuilder. Imports happen under the
    # subprocess's Python interpreter with our env vars honored.
    def _load() -> Dict[str, torch.Tensor]:
        import os
        import json
        from safetensors import safe_open
        import torch

        sys.path.insert(0, "/home/ubuntu")
        import neuron_flux2_dit as scaf  # scaffold

        idx_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                weight_map = json.load(f)["weight_map"]
            # Group keys by shard for batched open.
            by_shard: Dict[str, list] = {}
            for k, shard in weight_map.items():
                by_shard.setdefault(shard, []).append(k)
            sd: Dict[str, torch.Tensor] = {}
            for shard_name, keys in by_shard.items():
                shard_path = os.path.join(weights_dir, shard_name)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for k in keys:
                        sd[k] = f.get_tensor(k).to(torch.bfloat16)
        else:
            raise FileNotFoundError(f"No safetensors index at {idx_path}")

        # Build the minimal config object expected by the converter.
        hf_cfg = _load_hf_config(weights_dir)
        from neuronx_distributed_inference.models.config import NeuronConfig
        ncfg = NeuronConfig(
            tp_degree=tp_degree,
            torch_dtype=torch.bfloat16,
            cast_type="config",
        )
        cfg = scaf.NeuronFlux2Config(
            neuron_config=ncfg,
            patch_size=hf_cfg.get("patch_size", 1),
            in_channels=hf_cfg.get("in_channels", 128),
            out_channels=hf_cfg.get("out_channels", None),
            num_layers=hf_cfg.get("num_layers", 8),
            num_single_layers=hf_cfg.get("num_single_layers", 48),
            num_attention_heads=hf_cfg.get("num_attention_heads", 48),
            attention_head_dim=hf_cfg.get("attention_head_dim", 128),
            joint_attention_dim=hf_cfg.get("joint_attention_dim", 15360),
            mlp_ratio=hf_cfg.get("mlp_ratio", 3.0),
            axes_dims_rope=tuple(hf_cfg.get("axes_dims_rope", (32, 32, 32, 32))),
            rope_theta=hf_cfg.get("rope_theta", 2000),
            timestep_guidance_channels=hf_cfg.get("timestep_guidance_channels", 256),
            guidance_embeds=hf_cfg.get("guidance_embeds", True),
            eps=hf_cfg.get("eps", 1e-6),
        )
        converted = scaf.convert_hf_to_neuron_state_dict(sd, cfg)
        return converted

    return _load


# =============================================================================
# Model factory (picklable, called per-rank by ModelBuilder).
# =============================================================================
#
# ModelBuilder pickles the module_cls and invokes it inside each rank's XLA
# subprocess. So the factory can't close over anything not picklable — use
# module-level state.


_WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR
_TP_DEGREE = 8
_NUM_PATCHES = 4096     # L_img
_TEXT_SEQ_LEN = 512     # L_txt
_NUM_LAYERS_OVERRIDE = None
_NUM_SINGLE_LAYERS_OVERRIDE = None


def _factory_build_model():
    """Return an nn.Module instance configured for tracing.

    The module's forward signature must accept positional tensors matching
    what we pass as `example_inputs`. We use NeuronFlux2Transformer directly
    (NOT the ModelWrapper) because ModelWrapper expects .forward with host-side
    dispatch; ModelBuilder traces the raw nn.Module's forward.
    """
    import torch
    sys.path.insert(0, "/home/ubuntu")
    import neuron_flux2_dit as scaf
    from neuronx_distributed_inference.models.config import NeuronConfig

    hf_cfg = _load_hf_config(_WEIGHTS_DIR)
    ncfg = NeuronConfig(
        tp_degree=_TP_DEGREE,
        torch_dtype=torch.bfloat16,
        cast_type="config",
    )
    num_layers = _NUM_LAYERS_OVERRIDE if _NUM_LAYERS_OVERRIDE is not None else hf_cfg.get("num_layers", 8)
    num_single = _NUM_SINGLE_LAYERS_OVERRIDE if _NUM_SINGLE_LAYERS_OVERRIDE is not None else hf_cfg.get("num_single_layers", 48)
    cfg = scaf.NeuronFlux2Config(
        neuron_config=ncfg,
        patch_size=hf_cfg.get("patch_size", 1),
        in_channels=hf_cfg.get("in_channels", 128),
        out_channels=hf_cfg.get("out_channels", None),
        num_layers=num_layers,
        num_single_layers=num_single,
        num_attention_heads=hf_cfg.get("num_attention_heads", 48),
        attention_head_dim=hf_cfg.get("attention_head_dim", 128),
        joint_attention_dim=hf_cfg.get("joint_attention_dim", 15360),
        mlp_ratio=hf_cfg.get("mlp_ratio", 3.0),
        axes_dims_rope=tuple(hf_cfg.get("axes_dims_rope", (32, 32, 32, 32))),
        rope_theta=hf_cfg.get("rope_theta", 2000),
        timestep_guidance_channels=hf_cfg.get("timestep_guidance_channels", 256),
        guidance_embeds=hf_cfg.get("guidance_embeds", True),
        eps=hf_cfg.get("eps", 1e-6),
    )
    model = scaf.NeuronFlux2Transformer(cfg)
    model = model.to(torch.bfloat16)
    model.eval()
    return model


# =============================================================================
# Example inputs for tracing.
# =============================================================================

def build_example_inputs(num_patches: int, text_seq_len: int, hf_cfg: dict,
                        dtype=torch.bfloat16, batch: int = 1):
    """Match NeuronFlux2Transformer.forward signature exactly:

        forward(hidden_states, encoder_hidden_states, timestep, guidance,
                image_rotary_emb)
    """
    head_dim = hf_cfg.get("attention_head_dim", 128)
    in_ch = hf_cfg.get("in_channels", 128)
    jad = hf_cfg.get("joint_attention_dim", 15360)

    hidden = torch.randn(batch, num_patches, in_ch, dtype=dtype)
    encoder = torch.randn(batch, text_seq_len, jad, dtype=dtype)
    timestep = torch.tensor([500.0] * batch, dtype=dtype)
    if hf_cfg.get("guidance_embeds", True):
        guidance = torch.tensor([3.5] * batch, dtype=dtype)
    else:
        guidance = torch.tensor([], dtype=dtype)
    image_rotary_emb = torch.randn(num_patches + text_seq_len, head_dim, 2, dtype=dtype)

    return (hidden, encoder, timestep, guidance, image_rotary_emb)


# =============================================================================
# Main.
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    # patchify: 1024 / (2*vae_scale=16) = 1024/16 = 64 per-axis, 64*64=4096 tokens
    # at L_img.
    ap.add_argument("--num-patches", type=int, default=4096, help="L_img tokens (post-patchify)")
    ap.add_argument("--text-seq-len", type=int, default=512, help="L_txt tokens")
    ap.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--compile-workdir", default=DEFAULT_COMPILE_WORKDIR)
    ap.add_argument("--name", default="dit_tp8_1024", help="output NEFF filename (no extension)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="override num_layers (for proof-of-compile smoke)")
    ap.add_argument("--num-single-layers", type=int, default=None,
                    help="override num_single_layers")
    ap.add_argument("--dry-run", action="store_true",
                    help="build model factory + inputs once on CPU, no compile")
    ap.add_argument("--no-inline-weights", action="store_true",
                    help="disable inline_weights_to_neff (ship sharded tp_N.pt companions)")
    args = ap.parse_args()

    global _WEIGHTS_DIR, _TP_DEGREE, _NUM_PATCHES, _TEXT_SEQ_LEN
    global _NUM_LAYERS_OVERRIDE, _NUM_SINGLE_LAYERS_OVERRIDE
    _WEIGHTS_DIR = args.weights_dir
    _TP_DEGREE = args.tp
    _NUM_PATCHES = args.num_patches
    _TEXT_SEQ_LEN = args.text_seq_len
    _NUM_LAYERS_OVERRIDE = args.num_layers
    _NUM_SINGLE_LAYERS_OVERRIDE = args.num_single_layers

    os.makedirs(args.out_dir, exist_ok=True)

    hf_cfg = _load_hf_config(args.weights_dir)
    example_inputs = build_example_inputs(args.num_patches, args.text_seq_len, hf_cfg)
    print(f"[main] example_inputs shapes:")
    for i, t in enumerate(example_inputs):
        print(f"    [{i}] shape={tuple(t.shape)} dtype={t.dtype}")

    if args.dry_run:
        print("[dry-run] building model factory on CPU...")
        m = _factory_build_model()
        print(f"[dry-run] model has {sum(p.numel() for p in m.parameters()) / 1e9:.2f}B params")
        print(f"[dry-run] done, exiting")
        return

    # ---- ModelBuilder trace ----
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

    # clean workdir each run
    import shutil
    if os.path.exists(args.compile_workdir):
        shutil.rmtree(args.compile_workdir)
    os.makedirs(args.compile_workdir, exist_ok=True)

    ckpt_loader = _build_sd_loader(args.weights_dir, args.tp)

    model_instance = BaseModelInstance(
        module_cls=_factory_build_model,
        input_output_aliases={},
    )

    builder = ModelBuilder(
        router=None,
        tp_degree=args.tp,
        checkpoint_loader=ckpt_loader,
        compiler_workdir=args.compile_workdir,
        logical_nc_config=2,
    )

    compiler_args = (
        "--model-type=transformer "
        "-O1 "
        "--auto-cast=none "
        "--tensorizer-options='--enable-ccop-compute-overlap' "
        "--internal-hlo2tensorizer-options='--verify-hlo=true'"
    )
    # Setting LOCAL_WORLD_SIZE and NEURON_RT_VIRTUAL_CORE_SIZE is the scaffold's
    # convention for LNC=2 + TP sharding.
    os.environ["LOCAL_WORLD_SIZE"] = str(args.tp)

    builder.add(
        key="dit",
        model_instance=model_instance,
        example_inputs=[example_inputs],
        compiler_args=compiler_args,
    )

    print(f"[main] builder.trace tp={args.tp} L_img={args.num_patches} L_txt={args.text_seq_len}")
    print(f"[main] working in {args.compile_workdir}")
    t0 = time.perf_counter()
    traced = builder.trace(initialize_model_weights=True)
    compile_time = time.perf_counter() - t0
    print(f"[main] trace + compile + weight-init done in {compile_time:.1f}s")

    out_path = os.path.join(args.out_dir, f"{args.name}.pt")
    torch.jit.save(traced, out_path)
    neff_size = os.path.getsize(out_path)
    print(f"[main] saved to {out_path}  ({neff_size / (1024**2):.1f} MB)")

    # Dummy forward sanity check.
    print(f"[main] dummy forward with example inputs")
    gc.collect()
    t0 = time.perf_counter()
    out = traced(*example_inputs)
    first_lat = time.perf_counter() - t0
    if isinstance(out, (tuple, list)):
        out = out[0]
    print(f"[main] first forward latency: {first_lat * 1000:.0f} ms")
    print(f"[main] output: shape={tuple(out.shape)} dtype={out.dtype}")

    # 10-iter warm latency
    t0 = time.perf_counter()
    for _ in range(5):
        _ = traced(*example_inputs)
    warm = (time.perf_counter() - t0) / 5
    print(f"[main] warm per-iter latency: {warm * 1000:.0f} ms")

    print(f"\n=== SUMMARY ===")
    print(f"tp_degree       : {args.tp}")
    print(f"L_img           : {args.num_patches}")
    print(f"L_txt           : {args.text_seq_len}")
    print(f"num_layers      : {args.num_layers or hf_cfg.get('num_layers', 8)} double")
    print(f"num_single      : {args.num_single_layers or hf_cfg.get('num_single_layers', 48)} single")
    print(f"compile_time    : {compile_time:.1f}s")
    print(f"neff_path       : {out_path}")
    print(f"neff_size       : {neff_size / (1024**2):.1f} MB")
    print(f"first_forward   : {first_lat * 1000:.0f} ms")
    print(f"warm_forward    : {warm * 1000:.0f} ms")


if __name__ == "__main__":
    main()
