"""Trace Mistral-Small-3.2-24B text encoder on Neuron for FLUX.2-dev.

Feature-extractor: single forward pass, no KV cache, no sampling. Extracts
hidden states from layers [10, 20, 30] stacked as (B, S, 3, H).

Uses neuronx_distributed.trace.model_builder.ModelBuilder directly (newer API
than parallel_model_trace; handles LNC=2 / TP mapping cleanly).

Usage:
    python trace_text_encoder.py --ref-only    # just build CPU reference
    python trace_text_encoder.py --trace-only  # compile + load + validate
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# trn2 / LNC=2 required env vars (match what NxDI's Flux T5 encoder sets)
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("LOCAL_WORLD_SIZE", "8")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
# Use physical cores 16-31 = 16 cores = 8 logical cores at LNC=2.
# Avoids cores 8-9 currently held by a concurrent VAE validation job.
# Can be overridden externally if other allocation is needed.
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "16-31")


EXTRACT_LAYERS = (10, 20, 30)
MODEL_DIR = "/home/ubuntu/flux2_weights/text_encoder"
TOKENIZER_DIR = "/home/ubuntu/flux2_weights/tokenizer"
COMPILE_WORKDIR = "/home/ubuntu/text_encoder_compile"
COMPILED_MODEL_PATH = "/home/ubuntu/text_encoder_traced"
REF_PATH = "/home/ubuntu/text_encoder_ref_hidden.pt"


def load_text_config():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        full = json.load(f)
    return full["text_config"]


# -----------------------------------------------------------------------------
# Minimal parallel Mistral text encoder using neuronx_distributed primitives.
# -----------------------------------------------------------------------------


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x32).to(dtype_in)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, dtype: torch.dtype):
    pos = torch.arange(seq_len, dtype=torch.float32)
    freq_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freq_idx / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class ParallelMistralAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, tp_degree, dtype):
        from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp = tp_degree
        self.num_heads_per_rank = num_heads // tp_degree
        self.num_kv_heads_per_rank = num_kv_heads // tp_degree
        self.scale = head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, bias=False, gather_output=False, dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False, dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False, dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False, input_is_parallel=True, dtype=dtype,
        )

    def forward(self, hidden, cos, sin, causal_mask):
        B, S, _ = hidden.shape
        q = self.q_proj(hidden).view(B, S, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads_per_rank, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        group = self.num_heads_per_rank // self.num_kv_heads_per_rank
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + causal_mask
        attn = F.softmax(scores.to(torch.float32), dim=-1).to(v.dtype)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(ctx)


class ParallelMistralMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype):
        from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False, dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True, dtype=dtype,
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ParallelMistralDecoderLayer(nn.Module):
    def __init__(self, cfg, tp_degree, dtype):
        super().__init__()
        self.input_layernorm = MistralRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = ParallelMistralAttention(
            cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"],
            cfg["head_dim"], tp_degree, dtype,
        )
        self.post_attention_layernorm = MistralRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = ParallelMistralMLP(cfg["hidden_size"], cfg["intermediate_size"], dtype)

    def forward(self, x, cos, sin, causal_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ParallelMistralTextEncoder(nn.Module):
    """Mistral text model returning hidden_states[10,20,30] stacked along dim=2."""

    def __init__(self, cfg, tp_degree, seq_len, extract_layers, dtype):
        from neuronx_distributed.parallel_layers.layers import ParallelEmbedding
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.extract_layers = tuple(extract_layers)

        self.embed_tokens = ParallelEmbedding(
            cfg["vocab_size"], cfg["hidden_size"], padding_idx=None, dtype=dtype,
            shard_across_embedding=True,
        )

        # Build only the layers we need (up to max_needed) to save compute + memory.
        # hidden_states[k] in HF convention = output of layer k-1.
        # So we need decoder layers 0..max(extract_layers)-1 inclusive.
        self.num_needed_layers = max(self.extract_layers)
        self.layers = nn.ModuleList([
            ParallelMistralDecoderLayer(cfg, tp_degree, dtype)
            for _ in range(self.num_needed_layers)
        ])

        cos, sin = build_rope_cache(seq_len, cfg["head_dim"], cfg["rope_theta"], dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        neg_inf = torch.finfo(dtype).min
        mask = torch.triu(torch.full((seq_len, seq_len), neg_inf, dtype=dtype), diagonal=1)
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        wanted = {lid - 1 for lid in self.extract_layers}  # 0-indexed decoder layer positions
        outs = []
        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope_cos, self.rope_sin, self.causal_mask)
            if i in wanted:
                outs.append(h)
        # Stack along new axis: (B, S, num_layers, H). Order matches EXTRACT_LAYERS.
        return torch.stack(outs, dim=2)


# -----------------------------------------------------------------------------
# Weight loading: HF Mistral3 safetensors -> our flat state_dict keys.
# -----------------------------------------------------------------------------


def load_hf_state_dict(max_needed_layer):
    """Load language_model weights for layers 0..max_needed_layer (inclusive).
    Skips vision_tower, multi_modal_projector, lm_head.
    """
    from safetensors import safe_open
    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    keep_layers = set(range(max_needed_layer + 1))
    prefix = "language_model.model."
    wanted_hf_keys = []
    for k in weight_map:
        if not k.startswith(prefix):
            continue
        sub = k[len(prefix):]
        if sub.startswith("layers."):
            lid = int(sub.split(".")[1])
            if lid not in keep_layers:
                continue
        # We don't use `model.norm.weight` (final norm) since we extract mid-layers.
        if sub == "norm.weight":
            continue
        wanted_hf_keys.append(k)

    by_shard = {}
    for k in wanted_hf_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    sd = {}
    for shard_name, keys in by_shard.items():
        shard_path = os.path.join(MODEL_DIR, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_k in keys:
                our_k = hf_k[len(prefix):]
                sd[our_k] = f.get_tensor(hf_k).to(torch.bfloat16)
    return sd


# -----------------------------------------------------------------------------
# CPU reference.
# -----------------------------------------------------------------------------


def compute_cpu_reference(prompt: str, seq_len: int):
    from transformers import AutoTokenizer, MistralConfig, MistralModel

    print(f"[ref] tokenizing: {prompt!r}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enc = tok(prompt, return_tensors="pt", padding="max_length",
              truncation=True, max_length=seq_len)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    print(f"[ref] building HF MistralModel (CPU, bf16)")
    cfg_dict = load_text_config()
    cfg = MistralConfig(**cfg_dict)
    model = MistralModel(cfg).to(torch.bfloat16)
    model.eval()

    print(f"[ref] loading weights from safetensors")
    sd = load_hf_state_dict(max_needed_layer=cfg_dict["num_hidden_layers"] - 1)
    # Also get norm.weight for HF (it uses it even though we skip).
    from safetensors import safe_open
    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map.get("language_model.model.norm.weight")
    if shard:
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt", device="cpu") as f:
            sd["norm.weight"] = f.get_tensor("language_model.model.norm.weight").to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[ref] WARN unexpected keys: {unexpected[:5]}")
    if missing:
        miss_names = [m for m in missing if "rotary_emb" not in m]
        if miss_names:
            print(f"[ref] WARN missing keys: {miss_names[:5]}")

    print(f"[ref] forward pass (CPU, bf16)")
    t0 = time.perf_counter()
    with torch.no_grad():
        # IMPORTANT: pass attention_mask=None so HF only applies the causal mask,
        # matching the traced Neuron graph (no padding mask support yet). For
        # FLUX.2-dev, the DiT consumes ALL token positions of the text encoder
        # output including pads, so this is the right behavior.
        out = model(input_ids=input_ids, attention_mask=None,
                    output_hidden_states=True, return_dict=True, use_cache=False)
    dt = time.perf_counter() - t0
    print(f"[ref] done in {dt:.1f}s")
    hs = [out.hidden_states[i] for i in EXTRACT_LAYERS]
    stacked = torch.stack(hs, dim=2)
    return input_ids, attention_mask, stacked, dt


# -----------------------------------------------------------------------------
# ModelBuilder integration.
# -----------------------------------------------------------------------------


# Module-level config for cross-process pickling.
TRACE_TP_DEGREE = 8
TRACE_SEQ_LEN = 512


def _build_traceable_module():
    """Factory used by BaseModelInstance. Must be picklable / restartable."""
    cfg = load_text_config()
    model = ParallelMistralTextEncoder(
        cfg=cfg,
        tp_degree=TRACE_TP_DEGREE,
        seq_len=TRACE_SEQ_LEN,
        extract_layers=EXTRACT_LAYERS,
        dtype=torch.bfloat16,
    )
    model = model.to(torch.bfloat16)
    model.eval()
    return model


def _checkpoint_loader():
    """Called once per rank by ModelBuilder to load + shard weights."""
    max_needed = max(EXTRACT_LAYERS) - 1
    return load_hf_state_dict(max_needed_layer=max_needed)


def trace_on_neuron(tp_degree: int, seq_len: int, input_ids_example: torch.Tensor):
    """Compile the model via ModelBuilder. Returns the compiled NxDModel ready to .forward()."""
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

    global TRACE_TP_DEGREE, TRACE_SEQ_LEN
    TRACE_TP_DEGREE = tp_degree
    TRACE_SEQ_LEN = seq_len

    # Clean workdir.
    if os.path.exists(COMPILE_WORKDIR):
        import shutil
        shutil.rmtree(COMPILE_WORKDIR)
    os.makedirs(COMPILE_WORKDIR, exist_ok=True)
    os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)

    model_instance = BaseModelInstance(
        module_cls=_build_traceable_module,
        input_output_aliases={},
    )

    example_inputs = [(input_ids_example,)]  # list of tuples (one per bucket)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=_checkpoint_loader,
        compiler_workdir=COMPILE_WORKDIR,
        logical_nc_config=2,  # LNC=2 on this instance
    )

    compiler_args = (
        "--model-type=transformer "
        "--enable-saturate-infinity "
        "--enable-mixed-precision-accumulation "
        "-O1 "
        "--auto-cast=none"
    )

    builder.add(
        key="text_encoder",
        model_instance=model_instance,
        example_inputs=example_inputs,
        compiler_args=compiler_args,
    )

    print(f"[trace] ModelBuilder.trace tp_degree={tp_degree} seq_len={seq_len}")
    t0 = time.perf_counter()
    traced_model = builder.trace(initialize_model_weights=True)
    compile_time = time.perf_counter() - t0
    print(f"[trace] compile+weight-init done in {compile_time:.1f}s")

    # Save NEFF.
    torch.jit.save(traced_model, os.path.join(COMPILED_MODEL_PATH, "text_encoder.pt"))
    print(f"[trace] saved compiled model to {COMPILED_MODEL_PATH}")
    return traced_model, compile_time


# -----------------------------------------------------------------------------
# Accuracy.
# -----------------------------------------------------------------------------


def neuron_allclose(ref, got, name=""):
    ref = ref.to(torch.float32)
    got = got.to(torch.float32)
    diff = (ref - got).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = F.cosine_similarity(ref.flatten().unsqueeze(0), got.flatten().unsqueeze(0)).item()
    rel = (diff / (ref.abs().clamp(min=1e-6))).mean().item()
    print(f"  [{name}] shape={tuple(ref.shape)} "
          f"cos_sim={cos:.6f} max_abs={max_abs:.4f} mean_abs={mean_abs:.4f} mean_rel={rel:.4f}")
    return cos


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--prompt", default="a red panda in a misty forest")
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--use-saved-ref", action="store_true")
    args = ap.parse_args()

    if args.trace_only or args.use_saved_ref:
        print(f"[main] loading saved CPU reference from {REF_PATH}")
        data = torch.load(REF_PATH, map_location="cpu")
        input_ids = data["input_ids"]
        ref_stacked = data["hidden_stacked"]
        ref_time = data.get("ref_time", float("nan"))
    else:
        input_ids, attention_mask, ref_stacked, ref_time = compute_cpu_reference(args.prompt, args.seq_len)
        torch.save({
            "input_ids": input_ids, "attention_mask": attention_mask,
            "hidden_stacked": ref_stacked, "ref_time": ref_time,
        }, REF_PATH)
        print(f"[main] saved CPU reference to {REF_PATH}")

    if args.ref_only:
        print("[main] --ref-only set, exiting")
        return

    gc.collect()

    traced, compile_time = trace_on_neuron(args.tp, args.seq_len, input_ids)

    print(f"[main] neuron warm-up forward")
    _ = traced(input_ids)

    print(f"[main] neuron timed forward (10 iter)")
    t0 = time.perf_counter()
    for _ in range(10):
        got = traced(input_ids)
    elapsed = (time.perf_counter() - t0) / 10
    print(f"[main] per-forward latency: {elapsed*1000:.1f} ms")

    if isinstance(got, (tuple, list)):
        got = got[0]
    got = got.cpu()

    print(f"[main] accuracy vs CPU reference:")
    for i, lid in enumerate(EXTRACT_LAYERS):
        neuron_allclose(ref_stacked[:, :, i, :], got[:, :, i, :], name=f"layer {lid}")
    neuron_allclose(ref_stacked, got, name="stack")

    print(f"\n=== summary ===")
    print(f"tp_degree       : {args.tp}")
    print(f"seq_len         : {args.seq_len}")
    print(f"extract_layers  : {EXTRACT_LAYERS}")
    print(f"cpu_ref_time    : {ref_time:.1f}s")
    print(f"compile_time    : {compile_time:.1f}s")
    print(f"neuron_latency  : {elapsed*1000:.1f} ms / prompt")
    print(f"neff_dir        : {COMPILED_MODEL_PATH}")


if __name__ == "__main__":
    main()
