"""
CPU parity test for NeuronFlux2DoubleStreamBlock and NeuronFlux2SingleStreamBlock.

Approach:
  1. Monkey-patch parallel-layer imports BEFORE importing the scaffold so that
     ColumnParallelLinear/RowParallelLinear/LayerNorm/CustomRMSNorm/SPMDRank/etc.
     all degrade to stock `nn.Module` equivalents usable on CPU with TP=1.
  2. Pre-seed `torch.distributed` to look like a 1-rank world.
  3. Load HF Flux2Transformer2DModel, snapshot block-0 + shared modulation.
  4. Instantiate scaffold modules, load HF weights via
     convert_hf_to_neuron_state_dict (module-scoped).
  5. Run forward, compare cos-sim per block type.

No Neuron devices touched.
"""

from __future__ import annotations

import os
import sys
import math
import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. Monkey-patch parallel infra BEFORE importing the scaffold.
# --------------------------------------------------------------------------

# a) torch.distributed single-rank init
if not torch.distributed.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    try:
        torch.distributed.init_process_group(
            backend="gloo", rank=0, world_size=1
        )
    except Exception as e:
        print(f"[warn] dist init failed: {e}; continuing (probably not needed)")


# b) stub parallel_layers.layers module: ColumnParallelLinear/RowParallelLinear
#    just become plain nn.Linear that ignores gather/reduce kwargs, plus SPMDRank
import neuronx_distributed.parallel_layers.layers as _pl_layers


class _CPLinear(nn.Linear):
    """Stand-in for ColumnParallelLinear on CPU TP=1. Accepts all extra kwargs."""

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        # pop every non-standard kwarg
        for k in (
            "gather_output",
            "reduce_dtype",
            "input_is_parallel",
            "reduce_output",
            "sequence_parallel_enabled",
            "sequence_dimension",
            "dtype",
            "device",
            "stride",
            "keep_master_weight_for_test",
            "init_method",
            "skip_bias_add",
            "pad",
            "tensor_model_parallel_group",
            "use_spmd_rank",
        ):
            kwargs.pop(k, None)
        super().__init__(in_features, out_features, bias=bool(bias))
        # mock tensor parallel group for reduce_from_tensor_model_parallel_region
        self.tensor_parallel_group = None


class _RPLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        for k in (
            "gather_output",
            "reduce_dtype",
            "input_is_parallel",
            "reduce_output",
            "sequence_parallel_enabled",
            "sequence_dimension",
            "dtype",
            "device",
            "stride",
            "keep_master_weight_for_test",
            "init_method",
            "skip_bias_add",
            "pad",
            "tensor_model_parallel_group",
            "use_spmd_rank",
        ):
            kwargs.pop(k, None)
        super().__init__(in_features, out_features, bias=bool(bias))
        self.tensor_parallel_group = None


class _SPMDRank(nn.Module):
    def __init__(self, world_size=1, tensor_model_parallel_size=None):
        super().__init__()
        self.register_buffer("rank", torch.arange(0, world_size, dtype=torch.int32))

    def forward(self):
        return self.rank


_pl_layers.ColumnParallelLinear = _CPLinear
_pl_layers.RowParallelLinear = _RPLinear
_pl_layers.SPMDRank = _SPMDRank


# c) stub parallel_state so get_tensor_model_parallel_size()==1 and world group exists
import neuronx_distributed.parallel_layers.parallel_state as _pstate

def _get_tp_size():
    return 1

class _Group:
    def size(self):
        return 1

def _get_world_group():
    return _Group()

def _get_dp_group(*a, **k):
    return _Group()

_pstate.get_tensor_model_parallel_size = _get_tp_size
_pstate.get_world_group = _get_world_group
_pstate.get_data_parallel_group = _get_dp_group


# d) stub reduce_from_tensor_model_parallel_region / gather helpers
import neuronx_distributed.parallel_layers.mappings as _mappings

def _reduce_noop(x, *a, **k):
    return x

def _gather_noop(x, *a, **k):
    return x

_mappings.reduce_from_tensor_model_parallel_region = _reduce_noop
_mappings.gather_from_tensor_model_parallel_region_with_dim = _gather_noop


# e) stub LayerNorm from neuronx_distributed.parallel_layers.layer_norm
import neuronx_distributed.parallel_layers.layer_norm as _ln_mod

class _LN(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, **kwargs):
        # drop any extra kwargs (dtype, device, sequence_parallel_enabled, ...)
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=(bias and elementwise_affine),
        )

_ln_mod.LayerNorm = _LN


# f) stub CustomRMSNorm (used for QK-norm on head_dim)
import neuronx_distributed_inference.modules.custom_calls as _cc

class _RMS(nn.RMSNorm):
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=True)

_cc.CustomRMSNorm = _RMS


# g) neutralise attention_wrapper_sharded_without_swap -> fallback to SDPA
import neuronx_distributed_inference.models.diffusers.flux.modeling_flux as _flux_nx

def _attn_sdpa(q, k, v):
    # q,k,v: [B, H, S, D]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

_flux_nx.attention_wrapper_sharded_without_swap = _attn_sdpa


# h) force hardware check to not be TRN1 (so we use our fallback attn).
# The scaffold calls `_HARDWARE = hardware(get_platform_target())` and then
# branches on `_HARDWARE == hardware.TRN1` — so `hardware` must be a callable
# AND expose TRN1/TRN2 attributes. Make it a class.
import neuronx_distributed.utils.utils as _ndu

class _hardware:
    TRN1 = "trn1_mock"
    TRN2 = "trn2_mock"
    _value = "cpu_mock"

    def __init__(self, target=None):
        self.target = target

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return id(self)

_ndu.hardware = _hardware

import torch_neuronx.utils as _tnxu
_tnxu.get_platform_target = lambda: "cpu_mock"


# --------------------------------------------------------------------------
# 2. Now import the scaffold (it will pick up the patches above).
# --------------------------------------------------------------------------
sys.path.insert(0, "/home/ubuntu")
import neuron_flux2_dit as scaf

# double-check the patches stuck in the scaffold module's namespace
scaf.ColumnParallelLinear = _CPLinear
scaf.RowParallelLinear = _RPLinear
scaf.LayerNorm = _LN
scaf.CustomRMSNorm = _RMS
scaf.reduce_from_tensor_model_parallel_region = _reduce_noop
scaf.attention_wrapper_sharded_without_swap = _attn_sdpa


# patch the module-level _HARDWARE so both blocks route to SDPA fallback.
# Also patch `hardware` symbol inside scaffold so `hardware.TRN1` access works.
scaf._HARDWARE = "cpu_mock_instance"
scaf.hardware = _hardware


# --------------------------------------------------------------------------
# 3. Load HF model and grab reference block-0 + shared modulation.
# --------------------------------------------------------------------------
print("[info] loading HF Flux2Transformer2DModel (bf16) on CPU ...", flush=True)

from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel,
    Flux2Modulation,
)

DTYPE = torch.float32  # run parity in fp32 for tight cos-sim

hf = Flux2Transformer2DModel.from_pretrained(
    "/home/ubuntu/flux2_weights/transformer",
    torch_dtype=DTYPE,
)
hf.eval()
print("[info] HF model loaded.", flush=True)


# --------------------------------------------------------------------------
# 4. Build inputs: tiny seq lens for CPU speed.
# --------------------------------------------------------------------------
torch.manual_seed(0)

B = 1
S_img = 64            # image tokens
S_txt = 16            # text tokens
D = 6144
HEADS = 48
HEAD_DIM = 128
MLP_HIDDEN = 18432    # 6144 * 3

img = torch.randn(B, S_img, D, dtype=DTYPE) * 0.5
txt = torch.randn(B, S_txt, D, dtype=DTYPE) * 0.5

# temb: (B, D) — feed through HF's own shared modulation to get temb_mod_img/txt
timestep = torch.tensor([500.0], dtype=DTYPE)
guidance = torch.tensor([3.5], dtype=DTYPE)
temb = hf.time_guidance_embed(timestep, guidance)
mod_img = hf.double_stream_modulation_img(temb)  # [B, 6*D]
mod_txt = hf.double_stream_modulation_txt(temb)
mod_single = hf.single_stream_modulation(temb)   # [B, 3*D]

# rotary emb for S=S_txt+S_img tokens; ids zeros for txt, grid for img.
ids = torch.zeros(S_txt + S_img, 4, dtype=torch.float32)
# give the img tokens a simple 2D grid on axes (2,3)
grid_h = int(math.sqrt(S_img))
assert grid_h * grid_h == S_img
for i in range(S_img):
    h = i // grid_h
    w = i % grid_h
    ids[S_txt + i, 2] = h
    ids[S_txt + i, 3] = w

# HF version (tuple cos,sin)
hf_cos, hf_sin = hf.pos_embed(ids)
hf_rotary = (hf_cos, hf_sin)

# Scaffold version: stack to [S, D, 2]
scaf_rotary = torch.stack([hf_cos, hf_sin], dim=-1).to(DTYPE)


# --------------------------------------------------------------------------
# 5. HF reference: run block-0 double + block-0 single.
# --------------------------------------------------------------------------
print("[info] running HF reference (block-0 double)...", flush=True)
hf_db = hf.transformer_blocks[0]
with torch.no_grad():
    hf_enc_out, hf_img_out = hf_db(
        hidden_states=img,
        encoder_hidden_states=txt,
        temb_mod_img=mod_img,
        temb_mod_txt=mod_txt,
        image_rotary_emb=hf_rotary,
    )
print(f"[info] HF double-block done; img_out={tuple(hf_img_out.shape)}  enc_out={tuple(hf_enc_out.shape)}", flush=True)

# HF single block: concat(enc, img), run
print("[info] running HF reference (block-0 single)...", flush=True)
hf_single_in = torch.cat([hf_enc_out, hf_img_out], dim=1)  # [B, S_txt+S_img, D]
hf_sb = hf.single_transformer_blocks[0]
with torch.no_grad():
    hf_single_out = hf_sb(
        hidden_states=hf_single_in,
        encoder_hidden_states=None,
        temb_mod=mod_single,
        image_rotary_emb=hf_rotary,
    )
print(f"[info] HF single-block done; out={tuple(hf_single_out.shape)}", flush=True)


# --------------------------------------------------------------------------
# 6. Build scaffold blocks (no config object — direct ctor).
# --------------------------------------------------------------------------
print("[info] instantiating scaffold blocks on CPU ...", flush=True)

nx_db = scaf.NeuronFlux2DoubleStreamBlock(
    dim=D,
    num_attention_heads=HEADS,
    attention_head_dim=HEAD_DIM,
    mlp_ratio=3.0,
    eps=1e-6,
    reduce_dtype=DTYPE,
).to(DTYPE)
nx_db.eval()

nx_sb = scaf.NeuronFlux2SingleStreamBlock(
    dim=D,
    num_attention_heads=HEADS,
    attention_head_dim=HEAD_DIM,
    mlp_ratio=3.0,
    eps=1e-6,
    reduce_dtype=DTYPE,
).to(DTYPE)
nx_sb.eval()


# --------------------------------------------------------------------------
# 7. Load HF weights into scaffold blocks.
#
# We only need the keys relevant to block-0 (double) / block-0 (single);
# build a minimal state_dict that exercises the converter for both.
# --------------------------------------------------------------------------

hf_sd = hf.state_dict()


# Rename HF keys to scaffold-block-local keys.
# ---- Double block state dict ----
db_sd_scaf = {}
prefix = "transformer_blocks.0."
for k, v in hf_sd.items():
    if not k.startswith(prefix):
        continue
    local = k[len(prefix):]  # e.g. "attn.to_q.weight"
    if local.endswith(".bias"):
        continue
    # Map `attn.to_out.0.weight` -> `attn.to_out_0.weight`
    if local == "attn.to_out.0.weight":
        local = "attn.to_out_0.weight"
    db_sd_scaf[local] = v.clone().detach().contiguous()

missing, unexpected = nx_db.load_state_dict(db_sd_scaf, strict=False)
print(f"[info] double-block load: missing={missing} unexpected={unexpected}")


# ---- Single block state dict ----
sb_sd_scaf = {}
prefix = "single_transformer_blocks.0."
for k, v in hf_sd.items():
    if not k.startswith(prefix):
        continue
    local = k[len(prefix):]  # e.g. "attn.norm_q.weight", "attn.to_qkv_mlp_proj.weight", "attn.to_out.weight"
    if local.endswith(".bias"):
        continue
    if local == "attn.to_out.weight":
        # Split fused to_out into to_out_attn / to_out_mlp
        w = v  # [D, D + mlp_hidden]
        sb_sd_scaf["to_out_attn.weight"] = w[:, :D].clone().detach().contiguous()
        sb_sd_scaf["to_out_mlp.weight"] = w[:, D:].clone().detach().contiguous()
        continue
    # strip leading `attn.`
    if local.startswith("attn."):
        local = local[len("attn."):]
    sb_sd_scaf[local] = v.clone().detach().contiguous()

missing, unexpected = nx_sb.load_state_dict(sb_sd_scaf, strict=False)
print(f"[info] single-block load: missing={missing} unexpected={unexpected}")


# --------------------------------------------------------------------------
# 8. Run scaffold blocks.
# --------------------------------------------------------------------------
print("[info] running scaffold double-block ...", flush=True)
with torch.no_grad():
    nx_enc_out, nx_img_out = nx_db(
        hidden_states=img,
        encoder_hidden_states=txt,
        temb_mod_img=mod_img,
        temb_mod_txt=mod_txt,
        rotary_emb=scaf_rotary,
    )

print("[info] running scaffold single-block ...", flush=True)
with torch.no_grad():
    nx_single_out = nx_sb(
        hidden_states=hf_single_in,
        temb_mod=mod_single,
        rotary_emb=scaf_rotary,
    )


# --------------------------------------------------------------------------
# 9. Compare.
# --------------------------------------------------------------------------
def report(name, a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    # use double precision dot-product for cos_sim to avoid fp32 rounding >1
    ad = a.double()
    bd = b.double()
    cos = (ad @ bd / (ad.norm() * bd.norm() + 1e-30)).item()
    mx = (a - b).abs().max().item()
    mean = (a - b).abs().mean().item()
    rel = ((a - b).abs() / (b.abs().clamp_min(1e-8))).mean().item()
    a_norm = a.norm().item()
    b_norm = b.norm().item()
    print(f"[PARITY] {name:30s}  cos_sim={cos:.8f}  max_abs={mx:.4e}  mean_abs={mean:.4e}  rel={rel:.4e}  |a|={a_norm:.3e} |b|={b_norm:.3e}")
    return cos

print("=" * 80)
cos_img = report("double-block img_out", nx_img_out, hf_img_out)
cos_txt = report("double-block enc_out", nx_enc_out, hf_enc_out)
cos_sng = report("single-block out",    nx_single_out, hf_single_out)
print("=" * 80)

ok = all(c > 0.999 for c in (cos_img, cos_txt, cos_sng))
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
