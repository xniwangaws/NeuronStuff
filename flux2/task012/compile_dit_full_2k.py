"""Compile full FLUX.2 DiT (8 double + 48 single) at TP=8 for end-to-end use.

No taps; this is the production-shaped NEFF. Intended for pipeline benchmarks.
"""
from __future__ import annotations

import argparse, gc, json, os, shutil, sys, time
from typing import Dict
import torch

os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-15")

WEIGHTS_DIR = "/home/ubuntu/flux2_weights/transformer"
TP_DEGREE = 8
NUM_PATCHES = 16384          # 1024^2 latent at 8x downsample
TEXT_SEQ_LEN = 512
NUM_LAYERS = 8              # full double blocks
NUM_SINGLE_LAYERS = 48      # full single blocks

OUT_DIR = "/home/ubuntu/flux2_tap_debug/compiled_full_2k"
WORKDIR = "/home/ubuntu/flux2_tap_debug/workdir_full_2k"


def _load_hf_config(weights_dir):
    with open(os.path.join(weights_dir, "config.json")) as f:
        return json.load(f)


def _build_sd_loader(weights_dir, tp_degree):
    def _load() -> Dict[str, torch.Tensor]:
        import os, json, sys
        from safetensors import safe_open
        import torch

        sys.path.insert(0, "/home/ubuntu/flux2/task008")
        import neuron_flux2_dit as scaf

        idx_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        by_shard: Dict[str, list] = {}
        for k, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(k)
        sd: Dict[str, torch.Tensor] = {}
        for shard_name, keys in by_shard.items():
            shard_path = os.path.join(weights_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in keys:
                    sd[k] = f.get_tensor(k).to(torch.bfloat16)

        hf_cfg = _load_hf_config(weights_dir)
        from neuronx_distributed_inference.models.config import NeuronConfig
        ncfg = NeuronConfig(tp_degree=tp_degree, torch_dtype=torch.bfloat16, cast_type="config")
        cfg = scaf.NeuronFlux2Config(
            neuron_config=ncfg,
            patch_size=hf_cfg.get("patch_size", 1),
            in_channels=hf_cfg.get("in_channels", 128),
            out_channels=hf_cfg.get("out_channels", None),
            num_layers=NUM_LAYERS,
            num_single_layers=NUM_SINGLE_LAYERS,
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
        return scaf.convert_hf_to_neuron_state_dict(sd, cfg)
    return _load


def _factory_build_model():
    import torch
    sys.path.insert(0, "/home/ubuntu/flux2/task008")
    import neuron_flux2_dit as scaf
    from neuronx_distributed_inference.models.config import NeuronConfig

    hf_cfg = _load_hf_config(WEIGHTS_DIR)
    ncfg = NeuronConfig(tp_degree=TP_DEGREE, torch_dtype=torch.bfloat16, cast_type="config")
    cfg = scaf.NeuronFlux2Config(
        neuron_config=ncfg,
        patch_size=hf_cfg.get("patch_size", 1),
        in_channels=hf_cfg.get("in_channels", 128),
        out_channels=hf_cfg.get("out_channels", None),
        num_layers=NUM_LAYERS,
        num_single_layers=NUM_SINGLE_LAYERS,
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
    # Plain (no-tap) model; call forward directly.
    model = scaf.NeuronFlux2Transformer(cfg)
    model = model.to(torch.bfloat16)
    model.eval()
    return model


def build_example_inputs(num_patches, text_seq_len, hf_cfg, dtype=torch.bfloat16, batch=1):
    head_dim = hf_cfg.get("attention_head_dim", 128)
    in_ch = hf_cfg.get("in_channels", 128)
    jad = hf_cfg.get("joint_attention_dim", 15360)
    hidden = torch.randn(batch, num_patches, in_ch, dtype=dtype)
    encoder = torch.randn(batch, text_seq_len, jad, dtype=dtype)
    timestep = torch.tensor([500.0] * batch, dtype=dtype)
    guidance = torch.tensor([3.5] * batch, dtype=dtype) if hf_cfg.get("guidance_embeds", True) else torch.tensor([], dtype=dtype)
    image_rotary_emb = torch.randn(num_patches + text_seq_len, head_dim, 2, dtype=dtype)
    return (hidden, encoder, timestep, guidance, image_rotary_emb)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hf_cfg = _load_hf_config(WEIGHTS_DIR)
    example_inputs = build_example_inputs(NUM_PATCHES, TEXT_SEQ_LEN, hf_cfg)

    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR, exist_ok=True)

    ckpt_loader = _build_sd_loader(WEIGHTS_DIR, TP_DEGREE)
    model_instance = BaseModelInstance(module_cls=_factory_build_model, input_output_aliases={})
    builder = ModelBuilder(
        router=None, tp_degree=TP_DEGREE,
        checkpoint_loader=ckpt_loader,
        compiler_workdir=WORKDIR, logical_nc_config=2,
    )
    compiler_args = (
        "--model-type=transformer -O1 --auto-cast=none "
        "--tensorizer-options='--enable-ccop-compute-overlap' "
        "--internal-hlo2tensorizer-options='--verify-hlo=true'"
    )
    os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)
    builder.add(
        key="dit_full",
        model_instance=model_instance,
        example_inputs=[example_inputs],
        compiler_args=compiler_args,
    )
    print(f"[full] tracing num_layers={NUM_LAYERS} num_single={NUM_SINGLE_LAYERS}")
    t0 = time.perf_counter()
    traced = builder.trace(initialize_model_weights=True)
    compile_time = time.perf_counter() - t0
    print(f"[full] compile_time={compile_time:.1f}s")

    out_path = os.path.join(OUT_DIR, "dit_full.pt")
    torch.jit.save(traced, out_path)
    print(f"[full] saved [2K] to {out_path} ({os.path.getsize(out_path)/(1024**2):.1f}MB)")

    print(f"[full] dummy forward...")
    t0 = time.perf_counter()
    out = traced(*example_inputs)
    print(f"[full] forward took {(time.perf_counter()-t0)*1000:.0f} ms")
    if isinstance(out, (tuple, list)):
        print(f"  output is tuple of {len(out)}; first: {tuple(out[0].shape)}")
    else:
        print(f"  output: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
