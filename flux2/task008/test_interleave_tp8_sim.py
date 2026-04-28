"""
Pure-CPU simulation to validate the `interleave_fused` weight rearrangement
in neuron_flux2_dit.py at TP=8 without touching Neuron cores.

The bug this verifies:

Real Neuron TP flow for the FF `linear_in` (SwiGLU):
  1. ColumnParallelLinear shards weight along output rows: rank r gets
     w[r*per:(r+1)*per, :]  where per = (2*mlp_hidden) / tp.
  2. Each rank computes y_r = linear(x, w_r)  -> shape [..., per].
  3. Each rank does `gate_r, up_r = y_r.chunk(2, dim=-1)` locally.
  4. Each rank computes silu(gate_r) * up_r   -> shape [..., per/2 = mlp_hidden/tp].
  5. RowParallelLinear consumes that.

For this to equal the unsharded reference, each rank's per-row-slab of `w`
MUST contain [gate_rows_for_r, up_rows_for_r] (first half gate, second up).
The fused HF weight stores [gate_rows_all, up_rows_all], so a naive
chunk(tp, dim=0) gives rank 0 only gate rows — broken.

`interleave_fused` permutes rows so chunk(tp) yields the right layout.

Same logic for single-block to_qkv_mlp_proj: forward does
`torch.split(..., [3*inner//tp, 2*mlp//tp], dim=-1)` then
`qkv.chunk(3, dim=-1)` and `mlp.chunk(2, dim=-1)`. Each rank needs
[Q_r, K_r, V_r, gate_r, up_r] contiguous.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.distributed single-rank init
if not torch.distributed.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    try:
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception:
        pass


# ---- Monkey-patch parallel infra (same as test_block_parity.py) ----
import neuronx_distributed.parallel_layers.layers as _pl_layers

class _CPLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        for k in ("gather_output", "reduce_dtype", "input_is_parallel",
                  "reduce_output", "sequence_parallel_enabled", "sequence_dimension",
                  "dtype", "device", "stride", "keep_master_weight_for_test",
                  "init_method", "skip_bias_add", "pad",
                  "tensor_model_parallel_group", "use_spmd_rank"):
            kwargs.pop(k, None)
        super().__init__(in_features, out_features, bias=bool(bias))
        self.tensor_parallel_group = None

_pl_layers.ColumnParallelLinear = _CPLinear
_pl_layers.RowParallelLinear = _CPLinear

class _SPMDRank(nn.Module):
    def __init__(self, world_size=1, **k):
        super().__init__()
        self.register_buffer("rank", torch.arange(0, world_size, dtype=torch.int32))
    def forward(self):
        return self.rank
_pl_layers.SPMDRank = _SPMDRank

import neuronx_distributed.parallel_layers.parallel_state as _pstate
_pstate.get_tensor_model_parallel_size = lambda: 1
class _G:
    def size(self): return 1
_pstate.get_world_group = lambda: _G()
_pstate.get_data_parallel_group = lambda *a, **k: _G()

import neuronx_distributed.parallel_layers.mappings as _mappings
_mappings.reduce_from_tensor_model_parallel_region = lambda x, *a, **k: x
_mappings.gather_from_tensor_model_parallel_region_with_dim = lambda x, *a, **k: x

import neuronx_distributed.parallel_layers.layer_norm as _ln
class _LN(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, **kw):
        super().__init__(normalized_shape=normalized_shape, eps=eps,
                         elementwise_affine=elementwise_affine,
                         bias=(bias and elementwise_affine))
_ln.LayerNorm = _LN

import neuronx_distributed_inference.modules.custom_calls as _cc
class _RMS(nn.RMSNorm):
    def __init__(self, normalized_shape, eps=1e-6, **k):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=True)
_cc.CustomRMSNorm = _RMS

import neuronx_distributed_inference.models.diffusers.flux.modeling_flux as _flux_nx
_flux_nx.attention_wrapper_sharded_without_swap = lambda q, k, v: \
    F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

import neuronx_distributed.utils.utils as _ndu
class _hw:
    TRN1 = "trn1_mock"; TRN2 = "trn2_mock"
    def __init__(self, t=None): self.t = t
    def __eq__(self, o): return False
    def __ne__(self, o): return True
    def __hash__(self): return id(self)
_ndu.hardware = _hw

import torch_neuronx.utils as _tnxu
_tnxu.get_platform_target = lambda: "cpu_mock"

sys.path.insert(0, "/home/ubuntu")
import neuron_flux2_dit as scaf


# -------------------------------------------------------------------
# Helpers simulating ColumnParallel forward + post-processing.
# -------------------------------------------------------------------

def swiglu_per_rank_output_no_interleave(w_full: torch.Tensor, x: torch.Tensor, tp: int):
    """
    Simulate what happens WITHOUT interleave_fused:
      - shard w by chunk(tp, dim=0)
      - each rank: linear -> chunk(2, dim=-1) -> silu*up
      - gather along -1 to get [B, S, mlp_hidden_total]
    Returns the full gathered activation.
    """
    shards = list(w_full.chunk(tp, dim=0))
    outs = []
    for s in shards:
        y = F.linear(x, s)                       # [..., per_out]
        a, b = y.chunk(2, dim=-1)
        outs.append(F.silu(a) * b)
    return torch.cat(outs, dim=-1)


def swiglu_reference(w_full: torch.Tensor, x: torch.Tensor):
    """Unsharded SwiGLU: linear -> chunk(2, dim=-1) -> silu*up."""
    y = F.linear(x, w_full)
    a, b = y.chunk(2, dim=-1)
    return F.silu(a) * b


def qkv_mlp_per_rank_no_interleave(w_full: torch.Tensor, x: torch.Tensor, tp: int,
                                   inner: int, mlp: int):
    """
    Simulate single-block fused projection WITHOUT interleave:
      - shard w by chunk(tp, dim=0)
      - each rank: linear -> split [3*inner/tp, 2*mlp/tp] -> qkv.chunk(3), mlp.chunk(2)
      - produce (Q_r, K_r, V_r, silu(gate_r)*up_r); gather Q,K,V,mlp_out along -1
    Returns tuple (Q_full, K_full, V_full, mlp_out_full).
    """
    shards = list(w_full.chunk(tp, dim=0))
    q_parts, k_parts, v_parts, mlp_parts = [], [], [], []
    for s in shards:
        y = F.linear(x, s)  # [..., (3*inner + 2*mlp)/tp]
        qkv_local = (3 * inner) // tp
        mlp_local = (2 * mlp) // tp
        qkv, mlp_gate_up = torch.split(y, [qkv_local, mlp_local], dim=-1)
        q_r, k_r, v_r = qkv.chunk(3, dim=-1)
        g_r, u_r = mlp_gate_up.chunk(2, dim=-1)
        mlp_out_r = F.silu(g_r) * u_r
        q_parts.append(q_r); k_parts.append(k_r); v_parts.append(v_r)
        mlp_parts.append(mlp_out_r)
    return (torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1),
            torch.cat(v_parts, dim=-1), torch.cat(mlp_parts, dim=-1))


def qkv_mlp_reference(w_full: torch.Tensor, x: torch.Tensor, inner: int, mlp: int):
    y = F.linear(x, w_full)
    qkv, mlp_gate_up = torch.split(y, [3 * inner, 2 * mlp], dim=-1)
    q, k, v = qkv.chunk(3, dim=-1)
    g, u = mlp_gate_up.chunk(2, dim=-1)
    return q, k, v, F.silu(g) * u


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def assert_close(got, ref, name, tol=0.05):  # loose: CPU fp32 BLAS matmul
                                              # reorders accumulation at shard
                                              # sizes, and SiLU amplifies. On
                                              # Neuron bf16 this noise is below
                                              # the native bf16 resolution.
    diff = (got - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ok = max_abs < tol
    flag = "PASS" if ok else "FAIL"
    print(f"  [{name}] max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} -> {flag}")
    return ok


def test_ff_linear_in(mlp_hidden, hidden, tp):
    print(f"=== test ff.linear_in  tp={tp}  sizes=[{mlp_hidden},{mlp_hidden}] in={hidden} ===")
    torch.manual_seed(0)
    total = 2 * mlp_hidden
    w = torch.randn(total, hidden, dtype=torch.float32)
    x = torch.randn(2, 8, hidden, dtype=torch.float32)

    ref = swiglu_reference(w, x)  # [2, 8, mlp_hidden]

    # First: validate that `interleave_fused` produces a row-permutation whose
    # per-rank slabs contain exactly the expected per-rank [gate_r, up_r] rows.
    if tp > 1:
        w_int_check = scaf.interleave_fused(w, [mlp_hidden, mlp_hidden], tp)
        per = total // tp
        per_gate = mlp_hidden // tp
        for r in range(tp):
            shard = w_int_check[r * per : (r + 1) * per]
            exp_gate = w[r * per_gate : (r + 1) * per_gate]
            exp_up = w[mlp_hidden + r * per_gate : mlp_hidden + (r + 1) * per_gate]
            assert torch.equal(shard[:per_gate], exp_gate), \
                f"ff.linear_in tp={tp} rank {r}: gate row mismatch"
            assert torch.equal(shard[per_gate:], exp_up), \
                f"ff.linear_in tp={tp} rank {r}: up row mismatch"
        print(f"  [STRUCT tp={tp}] per-rank [gate_r | up_r] row layout VERIFIED")

    if tp > 1:
        # Naive (no interleave) — should be wrong except at tp=1.
        naive = swiglu_per_rank_output_no_interleave(w, x, tp)
        naive_ok = assert_close(naive, ref, f"NAIVE tp={tp} (expected FAIL)", tol=1e-4)
        if naive_ok:
            raise RuntimeError(f"At tp={tp}, naive unexpectedly matched the reference — "
                               f"the sharding bug is not being provoked.")

    # With interleave.
    w_int = scaf.interleave_fused(w, [mlp_hidden, mlp_hidden], tp)
    assert w_int.shape == w.shape
    if tp == 1:
        assert torch.equal(w_int, w), "interleave_fused must be a strict no-op at tp=1"
    fixed = swiglu_per_rank_output_no_interleave(w_int, x, tp)
    return assert_close(fixed, ref, f"FIXED tp={tp}")


def test_to_qkv_mlp(inner, mlp, hidden, tp):
    print(f"=== test to_qkv_mlp_proj  tp={tp}  inner={inner} mlp={mlp} in={hidden} ===")
    torch.manual_seed(0)
    total = 3 * inner + 2 * mlp
    w = torch.randn(total, hidden, dtype=torch.float32)
    x = torch.randn(2, 8, hidden, dtype=torch.float32)

    ref_q, ref_k, ref_v, ref_mlp = qkv_mlp_reference(w, x, inner, mlp)

    # Validate per-rank [Q_r, K_r, V_r, gate_r, up_r] row layout.
    if tp > 1:
        sizes = [inner, inner, inner, mlp, mlp]
        w_int_check = scaf.interleave_fused(w, sizes, tp)
        per = total // tp
        offsets_in_w = [0, inner, 2 * inner, 3 * inner, 3 * inner + mlp]
        per_rank = [s // tp for s in sizes]
        for r in range(tp):
            shard = w_int_check[r * per : (r + 1) * per]
            srow = 0
            for b, s in enumerate(sizes):
                expected = w[offsets_in_w[b] + r * per_rank[b]:
                             offsets_in_w[b] + (r + 1) * per_rank[b]]
                got = shard[srow : srow + per_rank[b]]
                assert torch.equal(expected, got), \
                    f"to_qkv_mlp_proj tp={tp} rank {r} block {b} mismatch"
                srow += per_rank[b]
        print(f"  [STRUCT tp={tp}] per-rank [Q_r|K_r|V_r|gate_r|up_r] VERIFIED")

    if tp > 1:
        nq, nk, nv, nm = qkv_mlp_per_rank_no_interleave(w, x, tp, inner, mlp)
        # Naive path is mis-ordered but sometimes some heads accidentally align.
        # Check at least one component is wrong.
        mismatches = 0
        for name, got, ref in [("Q", nq, ref_q), ("K", nk, ref_k), ("V", nv, ref_v),
                               ("mlp", nm, ref_mlp)]:
            if not torch.allclose(got, ref, atol=1e-4):
                mismatches += 1
        if mismatches == 0:
            raise RuntimeError(f"At tp={tp}, naive unexpectedly matched the reference.")
        print(f"  [NAIVE tp={tp}] {mismatches}/4 components mismatch (expected >=1)")

    w_int = scaf.interleave_fused(w, [inner, inner, inner, mlp, mlp], tp)
    if tp == 1:
        assert torch.equal(w_int, w), "interleave_fused must be a strict no-op at tp=1"
    q, k, v, m = qkv_mlp_per_rank_no_interleave(w_int, x, tp, inner, mlp)
    ok_q = assert_close(q, ref_q, f"FIXED Q tp={tp}")
    ok_k = assert_close(k, ref_k, f"FIXED K tp={tp}")
    ok_v = assert_close(v, ref_v, f"FIXED V tp={tp}")
    ok_m = assert_close(m, ref_mlp, f"FIXED mlp tp={tp}")
    return ok_q and ok_k and ok_v and ok_m


def main():
    results = []
    # Realistic FLUX.2 sizes.
    MLP_HIDDEN = 18432
    HIDDEN = 6144
    INNER = 6144

    # FF linear_in
    for tp in (1, 2, 4, 8, 16):
        results.append((f"ff.linear_in tp={tp}", test_ff_linear_in(MLP_HIDDEN, HIDDEN, tp)))

    # to_qkv_mlp_proj
    for tp in (1, 2, 4, 8, 16):
        results.append((f"to_qkv_mlp_proj tp={tp}", test_to_qkv_mlp(INNER, MLP_HIDDEN, HIDDEN, tp)))

    print("\n=== summary ===")
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
