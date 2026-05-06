"""Phase R1 step 2: convert HF-shaped S3Diff UNet state_dict to NeuronS3Diff* layout.

Layout transform:
  {name}.base_layer.{weight,bias}            -> {name}.base.{weight,bias}
  {name}.lora_A.default.weight                -> {name}.lora_A.weight
  {name}.lora_B.default.weight                -> {name}.lora_B.weight

Validation:
  - preserve total parameter count
  - every original key maps to exactly one new key

Usage:
  python convert_state_dict.py --in_sd /path/to/input.pt --out_sd /path/to/output.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


KEY_MAP = [
    (".base_layer.weight", ".base.weight"),
    (".base_layer.bias", ".base.bias"),
    (".lora_A.default.weight", ".lora_A.weight"),
    (".lora_B.default.weight", ".lora_B.weight"),
]


def convert_key(k: str) -> str:
    for old, new in KEY_MAP:
        if k.endswith(old):
            return k[: -len(old)] + new
    return k


def convert_state_dict(sd: dict) -> dict:
    out = {}
    dup_check = Counter()
    for k, v in sd.items():
        nk = convert_key(k)
        dup_check[nk] += 1
        out[nk] = v
    dups = {k: c for k, c in dup_check.items() if c > 1}
    if dups:
        raise ValueError(f"key collision in convert_state_dict: {dups}")
    return out


def verify(old_sd: dict, new_sd: dict) -> None:
    """Param count preserved + ordered keys map 1:1."""
    assert len(old_sd) == len(new_sd), f"key count changed: {len(old_sd)} -> {len(new_sd)}"
    total_old = sum(v.numel() for v in old_sd.values())
    total_new = sum(v.numel() for v in new_sd.values())
    assert total_old == total_new, f"param count changed: {total_old} -> {total_new}"
    shapes_old = sorted((tuple(v.shape), v.dtype) for v in old_sd.values())
    shapes_new = sorted((tuple(v.shape), v.dtype) for v in new_sd.values())
    assert shapes_old == shapes_new, "shape/dtype multiset changed"
    print(f"[verify] ok — {len(old_sd)} keys, {total_old:,} params preserved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_sd", required=True, help="input state_dict .pt path (HF-shaped)")
    p.add_argument("--out_sd", required=True, help="output state_dict .pt path (Neuron-shaped)")
    args = p.parse_args()

    print(f"[load] {args.in_sd}", flush=True)
    old = torch.load(args.in_sd, map_location="cpu", weights_only=True)
    print(f"[load] {len(old)} keys, {sum(v.numel() for v in old.values()):,} params", flush=True)

    new = convert_state_dict(old)
    verify(old, new)

    Path(args.out_sd).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, args.out_sd)
    print(f"[save] {args.out_sd}", flush=True)

    # Print sample of rewritten keys
    sample_in = list(old.keys())[:5]
    print("[sample] before/after:")
    for k in sample_in:
        print(f"  {k}\n  -> {convert_key(k)}")


if __name__ == "__main__":
    main()
