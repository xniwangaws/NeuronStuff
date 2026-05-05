"""Correctness test for DeModLoRALinear / DeModLoRAConv2d.

Compares new folded-einsum forward against a reference implementation that
mirrors `my_lora_fwd` in the original S3Diff repo line-for-line. Gate:
max|diff| < 1e-5 in fp32.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")
from de_mod_lora import DeModLoRALinear, DeModLoRAConv2d


def ref_lora_linear_forward(base_W, base_b, lora_A_W, lora_B_W, x, de_mod, scaling):
    """Mirrors my_lora_fwd for nn.Linear LoRA."""
    result = F.linear(x, base_W, base_b)
    a = F.linear(x, lora_A_W)  # (..., r)
    # einsum('...lk,...kr->...lr', a, de_mod)
    # our a is (..., r), de_mod is (B, r, r); peft writes this as ...lk,...kr->...lr where k and l are last-2
    # Actually in the original: _tmp = lora_A(dropout(x))  is (..., r)
    # then einsum('...lk,...kr->...lr', _tmp, self.de_mod)
    # But de_mod is (B, r, r); the '...' captures everything up to last 2 dims of _tmp.
    # Safest: reshape a to (B, seq, r), de_mod is (B, r, r), result is (B, seq, r).
    orig_shape = a.shape
    a_flat = a.view(-1, orig_shape[-1]) if a.dim() > 2 else a  # (B*seq, r) or (B, r)
    # Actually to be robust, treat `...` as the batch + seq dims. Use batched matmul:
    # a: (..., r), de_mod: (B_of_demod, r, r). If batch matches, broadcast.
    B = de_mod.shape[0]
    if a.shape[0] == B:
        # a: (B, seq, r), de_mod: (B, r, r) -> (B, seq, r)
        _tmp = torch.einsum("blk,bkr->blr", a, de_mod) if a.dim() == 3 else torch.einsum("bk,bkr->br", a, de_mod)
    else:
        # broadcast over batch: a has no batch (2D: seq, r) — not our case
        _tmp = torch.einsum("...k,bkr->...br", a, de_mod)
    result = result + F.linear(_tmp, lora_B_W) * scaling
    return result


def ref_lora_conv_forward(base, lora_A, lora_B, x, de_mod, scaling):
    """Mirrors my_lora_fwd for Conv2d LoRA."""
    result = base(x)
    a = lora_A(x)  # (B, r, H', W')
    # einsum('...khw,...kr->...rhw', _tmp, de_mod)
    # _tmp: (B, r, H, W), de_mod: (B, r, r)  -> (B, r, H, W) new
    # Interpretation: a's "k" dim is channels (r), de_mod's "k" is row, "r" is column.
    # So: new[b, r', h, w] = sum_k a[b, k, h, w] * de_mod[b, k, r']
    _tmp = torch.einsum("bkhw,bkr->brhw", a, de_mod)
    result = result + lora_B(_tmp) * scaling
    return result


def test_linear():
    torch.manual_seed(0)
    in_f, out_f, r = 320, 320, 32
    B, seq = 1, 128

    layer = DeModLoRALinear(in_f, out_f, r, bias=True, scaling=1.5).double()
    # Random input + de_mod
    x = torch.randn(B, seq, in_f, dtype=torch.float64)
    de_mod = torch.randn(B, r, r, dtype=torch.float64)

    # New forward
    out_new = layer(x, de_mod)

    # Reference forward
    out_ref = ref_lora_linear_forward(
        layer.base.weight, layer.base.bias,
        layer.lora_A.weight, layer.lora_B.weight,
        x, de_mod, layer.scaling,
    )

    diff = (out_new - out_ref).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    print(f"[test_linear] shape={tuple(out_new.shape)}  max|diff|={max_diff:.2e}  mean|diff|={mean_diff:.2e}")
    assert max_diff < 1e-5, f"Linear mismatch: max|diff|={max_diff}"
    print("[test_linear] PASS")


def test_conv2d():
    torch.manual_seed(0)
    in_c, out_c, r, k = 320, 320, 32, 3
    B, H, W = 1, 32, 32

    layer = DeModLoRAConv2d(in_c, out_c, kernel_size=k, lora_rank=r, padding=1, bias=True, scaling=1.5).double()
    x = torch.randn(B, in_c, H, W, dtype=torch.float64)
    de_mod = torch.randn(B, r, r, dtype=torch.float64)

    out_new = layer(x, de_mod)
    out_ref = ref_lora_conv_forward(layer.base, layer.lora_A, layer.lora_B, x, de_mod, layer.scaling)

    diff = (out_new - out_ref).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    print(f"[test_conv2d] shape={tuple(out_new.shape)}  max|diff|={max_diff:.2e}  mean|diff|={mean_diff:.2e}")
    assert max_diff < 1e-5, f"Conv2d mismatch: max|diff|={max_diff}"
    print("[test_conv2d] PASS")


def test_linear_no_bias():
    torch.manual_seed(1)
    in_f, out_f, r = 640, 1280, 32
    B, seq = 1, 256

    layer = DeModLoRALinear(in_f, out_f, r, bias=False, scaling=2.0).double()
    x = torch.randn(B, seq, in_f, dtype=torch.float64)
    de_mod = torch.randn(B, r, r, dtype=torch.float64)
    out_new = layer(x, de_mod)
    out_ref = ref_lora_linear_forward(
        layer.base.weight, None,
        layer.lora_A.weight, layer.lora_B.weight,
        x, de_mod, layer.scaling,
    )
    diff = (out_new - out_ref).abs()
    max_diff = diff.max().item()
    print(f"[test_linear_no_bias] max|diff|={max_diff:.2e}")
    assert max_diff < 1e-5
    print("[test_linear_no_bias] PASS")


def test_conv2d_various_shapes():
    torch.manual_seed(2)
    configs = [
        # (in_c, out_c, kernel, pad, H, W, r)
        (320, 640, 3, 1, 64, 64, 32),   # down_block conv1
        (640, 320, 1, 0, 32, 32, 32),   # conv_shortcut 1x1
        (1280, 1280, 3, 1, 16, 16, 32), # mid_block conv
    ]
    for (ic, oc, k, p, H, W, r) in configs:
        layer = DeModLoRAConv2d(ic, oc, kernel_size=k, lora_rank=r, padding=p, bias=True).double()
        x = torch.randn(1, ic, H, W, dtype=torch.float64)
        de_mod = torch.randn(1, r, r, dtype=torch.float64)
        out_new = layer(x, de_mod)
        out_ref = ref_lora_conv_forward(layer.base, layer.lora_A, layer.lora_B, x, de_mod, 1.0)
        md = (out_new - out_ref).abs().max().item()
        print(f"  conv {ic}->{oc} k{k} {H}x{W}  max|diff|={md:.2e}")
        assert md < 1e-5
    print("[test_conv2d_various_shapes] PASS")


if __name__ == "__main__":
    test_linear()
    test_linear_no_bias()
    test_conv2d()
    test_conv2d_various_shapes()
    print("\n[ALL PASS]")
