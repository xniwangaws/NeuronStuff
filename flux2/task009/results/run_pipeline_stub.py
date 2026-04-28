"""End-to-end NeuronFlux2Pipeline integration test with stub DiT.

Validates orchestration only — the DiT is a random stub, so the output image
is garbage by design. Success == pipeline runs without crashing, produces an
image of the requested size.

Usage:
    python run_pipeline_stub.py
    python run_pipeline_stub.py --prompt "a red panda" --height 1024 --width 1024
    python run_pipeline_stub.py --text-encoder-mode cpu   # force HF CPU fallback
    python run_pipeline_stub.py --text-encoder-mode neuron # force Neuron NEFF (fails if not saved)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make the pipeline module importable when we scp this alongside it.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from neuron_flux2_pipeline import NeuronFlux2Pipeline, NeuronDiTStub, NeuronDiT, _init_nxd_if_needed


# -------------------- paths on the Neuron instance --------------------------
FLUX2_WEIGHTS = "/home/ubuntu/flux2_weights"
TOKENIZER_DIR = f"{FLUX2_WEIGHTS}/tokenizer"
VAE_NEFF      = "/home/ubuntu/vae_traced/vae_decoder_512.pt"
TEXT_NEFF_DIR = "/home/ubuntu/text_encoder_traced"       # parallel_model_save target
TEXT_NEFF_PT  = "/home/ubuntu/text_encoder_traced/text_encoder.pt"  # torch.jit.save target
DIT_NEFF      = "/home/ubuntu/dit_traced/dit_tp8_1024.pt"
VAE_TILE_MOD  = "/home/ubuntu/vae_tile_decode.py"        # source of tiled_decode_neff

DEFAULT_PROMPT = "a red panda in a misty forest, photorealistic"


# -----------------------------------------------------------------------------
# Text encoder loaders.
# -----------------------------------------------------------------------------

def load_text_encoder_neuron():
    """Load the torch.jit.save'd Neuron text encoder.

    The trace script (trace_text_encoder.py) uses ModelBuilder + torch.jit.save,
    yielding a single .pt archive that torch.jit.load restores directly.
    """
    import torch_neuronx  # noqa: F401  (registers neuron ops for jit.load)

    if not os.path.exists(TEXT_NEFF_PT):
        raise RuntimeError(
            f"Neuron text encoder NEFF {TEXT_NEFF_PT} not found "
            f"— trace_text_encoder.py has not finished saving yet"
        )
    print(f"[load] torch.jit.load({TEXT_NEFF_PT})")
    traced = torch.jit.load(TEXT_NEFF_PT)
    _init_nxd_if_needed(traced, label="text_encoder")

    def encode_fn(input_ids, attention_mask=None):
        # Neuron trace ignores attention_mask (causal-only, fixed 512 padding).
        out = traced(input_ids)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.cpu() if hasattr(out, "cpu") else out

    return encode_fn


def load_text_encoder_cpu():
    """Load HF MistralModel on CPU, return a callable matching our interface.

    This is the fallback / ground truth path (same as cpu_reference.py).
    The model is ~48GB in bf16 so host RAM is the real bottleneck; but we
    don't load lm_head or vision towers, so MistralModel alone is ~45GB.
    """
    import json as _json

    from transformers import MistralConfig, MistralModel

    cfg_path = f"{FLUX2_WEIGHTS}/text_encoder/config.json"
    with open(cfg_path) as f:
        full_cfg = _json.load(f)
    text_cfg = full_cfg["text_config"]
    cfg = MistralConfig(**text_cfg)
    print(f"[load-cpu] instantiating MistralModel (bf16, CPU) num_hidden_layers={cfg.num_hidden_layers}")
    model = MistralModel(cfg).to(torch.bfloat16).eval()

    # Load weights from the language_model.model.* prefix in the HF safetensors.
    from safetensors import safe_open

    idx_path = f"{FLUX2_WEIGHTS}/text_encoder/model.safetensors.index.json"
    with open(idx_path) as f:
        weight_map = _json.load(f)["weight_map"]
    prefix = "language_model.model."
    by_shard = {}
    for k in weight_map:
        if k.startswith(prefix):
            by_shard.setdefault(weight_map[k], []).append(k)

    sd = {}
    for shard_name, keys in by_shard.items():
        shard_path = f"{FLUX2_WEIGHTS}/text_encoder/{shard_name}"
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_k in keys:
                sd[hf_k[len(prefix):]] = f.get_tensor(hf_k)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_real = [m for m in missing if "rotary_emb" not in m]
    if miss_real:
        print(f"[load-cpu] WARN missing: {miss_real[:5]}")
    if unexpected:
        print(f"[load-cpu] WARN unexpected: {unexpected[:5]}")
    print(f"[load-cpu] MistralModel ready")

    @torch.no_grad()
    def encode_fn(input_ids, attention_mask=None):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # Return the ModelOutput; NeuronFlux2Pipeline._encode_prompt extracts
        # hidden_states[extract_layers] and stacks.
        return out

    return encode_fn


# -----------------------------------------------------------------------------
# VAE decode wrapper — reuses /home/ubuntu/vae_tile_decode.py
# -----------------------------------------------------------------------------

def load_vae_decode_fn():
    """Wrap the traced 512² VAE NEFF with tiled_decode for arbitrary HxW."""
    # import vae_tile_decode by file path
    import importlib.util

    spec = importlib.util.spec_from_file_location("vae_tile_decode", VAE_TILE_MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import torch_neuronx  # noqa: F401  (required so jit.load picks up neuron ops)

    print(f"[load] VAE NEFF: {VAE_NEFF}")
    vae_neff = torch.jit.load(VAE_NEFF)

    def neff_call(x):
        with torch.no_grad():
            return vae_neff(x.to(torch.bfloat16))

    # Warmup the NEFF so we don't time cold-start.
    print("[load] VAE warmup")
    _ = neff_call(torch.randn(1, 32, 64, 64, dtype=torch.bfloat16))

    def decode_fn(latents):
        """(B, 32, h_l, w_l) -> (B, 3, H, W) pixel-space in [-1, 1]."""
        return mod.tiled_decode_neff(neff_call, latents.to(torch.bfloat16))

    return decode_fn


# -----------------------------------------------------------------------------
# VAE BN params — only need the running_mean/var + eps from the VAE config.
# Avoids loading the whole decoder/encoder state_dict into host RAM.
# -----------------------------------------------------------------------------

def load_vae_bn_params():
    from safetensors import safe_open
    # ae.safetensors has the BN buffers at top-level (AutoencoderKLFlux2.bn.*).
    bn_path = f"{FLUX2_WEIGHTS}/ae.safetensors"
    if not os.path.exists(bn_path):
        bn_path = f"{FLUX2_WEIGHTS}/vae/diffusion_pytorch_model.safetensors"
    print(f"[load] VAE BN params from {bn_path}")
    with safe_open(bn_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        mean_key = next(k for k in keys if k.endswith("bn.running_mean"))
        var_key = next(k for k in keys if k.endswith("bn.running_var"))
        mean = f.get_tensor(mean_key).to(torch.bfloat16)
        var = f.get_tensor(var_key).to(torch.bfloat16)

    # eps comes from vae config.json
    with open(f"{FLUX2_WEIGHTS}/vae/config.json") as f:
        cfg = json.load(f)
    eps = float(cfg.get("batch_norm_eps", 1e-5))
    print(f"[load] BN mean={tuple(mean.shape)} var={tuple(var.shape)} eps={eps}")
    return mean, var, eps


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-encoder-mode", choices=["auto", "neuron", "cpu"], default="auto")
    ap.add_argument("--dit-real", action="store_true",
                    help=f"Load real Neuron DiT NEFF from {DIT_NEFF} instead of the stub.")
    ap.add_argument("--dit-neff", default=DIT_NEFF, help="path to DiT NEFF (when --dit-real)")
    ap.add_argument("--output", default=None,
                    help="default: /home/ubuntu/stub_pipeline_out.png (stub) or "
                         "/home/ubuntu/real_dit_pipeline_out.png (--dit-real)")
    args = ap.parse_args()
    if args.output is None:
        args.output = ("/home/ubuntu/real_dit_pipeline_out.png"
                       if args.dit_real else "/home/ubuntu/stub_pipeline_out.png")

    print(f"=== run_pipeline_stub.py ===")
    print(f"prompt    : {args.prompt!r}")
    print(f"resolution: {args.width}x{args.height}  steps={args.steps}  cfg={args.guidance}")

    # --- text encoder ---------------------------------------------------
    t0 = time.perf_counter()
    if args.text_encoder_mode == "neuron":
        text_fn = load_text_encoder_neuron()
    elif args.text_encoder_mode == "cpu":
        text_fn = load_text_encoder_cpu()
    else:  # auto — try Neuron first, fall back to CPU
        try:
            text_fn = load_text_encoder_neuron()
            print("[main] using Neuron text encoder")
        except Exception as e:
            print(f"[main] Neuron text encoder unavailable ({e}); falling back to HF CPU MistralModel")
            text_fn = load_text_encoder_cpu()
    print(f"[main] text encoder loaded in {time.perf_counter()-t0:.1f}s")

    # --- VAE ------------------------------------------------------------
    t0 = time.perf_counter()
    vae_decode_fn = load_vae_decode_fn()
    print(f"[main] VAE loaded+warmed in {time.perf_counter()-t0:.1f}s")

    # --- VAE BN params + scheduler + tokenizer --------------------------
    bn_mean, bn_var, bn_eps = load_vae_bn_params()

    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        FLUX2_WEIGHTS, subfolder="scheduler"
    )

    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(TOKENIZER_DIR)

    # --- DiT ------------------------------------------------------------
    if args.dit_real:
        if not os.path.exists(args.dit_neff):
            raise RuntimeError(f"--dit-real requested but {args.dit_neff} not found")
        import torch_neuronx  # noqa: F401
        print(f"[load] torch.jit.load({args.dit_neff})")
        t0 = time.perf_counter()
        dit_jit = torch.jit.load(args.dit_neff)
        _init_nxd_if_needed(dit_jit, label="dit")
        dit_mod = NeuronDiT(dit_jit)
        print(f"[load] DiT NEFF loaded in {time.perf_counter()-t0:.1f}s")
    else:
        dit_mod = NeuronDiTStub(dtype=torch.bfloat16)
        print(f"[load] using DiT stub (random output)")

    # --- pipeline -------------------------------------------------------
    pipe = NeuronFlux2Pipeline(
        neuron_text_encoder=text_fn,
        neuron_dit=dit_mod,
        vae_decode_fn=vae_decode_fn,
        scheduler=sched,
        tokenizer=tokenizer,
        vae_bn_mean=bn_mean,
        vae_bn_var=bn_var,
        vae_bn_eps=bn_eps,
        dtype=torch.bfloat16,
        transformer_in_channels=128,
        text_encoder_max_len=512,
        extract_layers=(10, 20, 30),
    )

    # --- run ------------------------------------------------------------
    print(f"\n[main] running pipeline...")
    t0 = time.perf_counter()
    image, timings = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        output_type="pil",
        profile=True,
    )
    total = time.perf_counter() - t0

    # --- validate -------------------------------------------------------
    print(f"\n=== orchestration validation ===")
    print(f"image type   : {type(image).__name__}")
    print(f"image size   : {image.size}  (expected {args.width}x{args.height})")
    assert image.size == (args.width, args.height), "image size mismatch"
    image.save(args.output)
    print(f"[done] saved garbage-but-well-shaped image to {args.output}")

    print(f"\n=== timings ===")
    print(f"text_encoder : {timings['text_encoder_s']:.2f} s")
    print(f"scheduler    : {timings['scheduler_loop_s']:.2f} s  ({args.steps} steps, DiT stub)")
    print(f"vae_decode   : {timings['vae_decode_s']:.2f} s")
    print(f"total (pipe) : {total:.2f} s")

    # Also dump machine-readable timing.
    with open("/home/ubuntu/stub_pipeline_timings.json", "w") as f:
        json.dump({"total_s": total, **timings,
                   "resolution": [args.width, args.height],
                   "steps": args.steps,
                   "text_encoder_mode": args.text_encoder_mode}, f, indent=2)


if __name__ == "__main__":
    main()
