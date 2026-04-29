"""Inspect the S3 snapshot files to understand structure.

Outputs key list, shapes, dtypes for both files. Safe read-only.
"""
import pickle
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent / "s3_snapshots"
INPUTS = HERE / "task011-neff_step0_v2.pt"
COMPARE = HERE / "task011-step0_compare_v2.pt"


def describe(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}dict with {len(obj)} keys:")
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                print(f"{pad}  {k!r}: Tensor shape={tuple(v.shape)} dtype={v.dtype} "
                      f"device={v.device} "
                      f"min={v.min().item() if v.numel() else 'n/a':.4g} "
                      f"max={v.max().item() if v.numel() else 'n/a':.4g} "
                      f"norm={v.float().norm().item() if v.numel() else 'n/a':.4g}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"{pad}  {k!r}: {type(v).__name__} = {v!r}")
            elif isinstance(v, dict):
                print(f"{pad}  {k!r}:")
                describe(v, indent + 2)
            elif isinstance(v, (list, tuple)):
                print(f"{pad}  {k!r}: {type(v).__name__} len={len(v)}")
                for i, x in enumerate(v[:3]):
                    if isinstance(x, torch.Tensor):
                        print(f"{pad}    [{i}] Tensor shape={tuple(x.shape)} dtype={x.dtype}")
                    else:
                        print(f"{pad}    [{i}] {type(x).__name__} = {x!r}")
            else:
                print(f"{pad}  {k!r}: {type(v).__name__}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj}")


def main():
    for name, path in [("INPUTS (neff_step0_v2)", INPUTS), ("COMPARE (step0_compare_v2)", COMPARE)]:
        print("=" * 80)
        print(f"FILE: {name}  ({path.stat().st_size / 1e6:.2f} MB)")
        print("=" * 80)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"torch.load failed: {e}")
            print("trying unsafe pickle...")
            with open(path, "rb") as f:
                obj = pickle.load(f)
        describe(obj)
        print()


if __name__ == "__main__":
    main()
