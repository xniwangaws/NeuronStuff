"""PSNR comparison: Jim's S3Diff outputs vs our Phase E CPU fp32 references."""
import os
import glob
import json
import numpy as np
from PIL import Image

OUT_LR = int(os.environ.get("LR", "256"))
cpu_dir = os.path.expanduser("~/s3diff_jim/cpu_ref/")
jim_dir = os.path.expanduser(f"~/s3diff_jim/outputs/{OUT_LR}/")

def psnr(a, b):
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    if a.shape != b.shape:
        b_im = Image.fromarray((b * 255).astype(np.uint8))
        b_im = b_im.resize((a.shape[1], a.shape[0]), Image.BICUBIC)
        b = np.asarray(b_im, dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(-10 * np.log10(mse))

results = []
for cpu_path in sorted(glob.glob(os.path.join(cpu_dir, "*_cpu_fp32.png"))):
    name = os.path.basename(cpu_path).replace("_cpu_fp32.png", "")
    jim_path = os.path.join(jim_dir, f"{name}_jim_neuron_{OUT_LR}.png")
    if not os.path.exists(jim_path):
        print(f"missing: {jim_path}")
        continue
    cpu_im = Image.open(cpu_path).convert("RGB")
    jim_im = Image.open(jim_path).convert("RGB")
    p = psnr(cpu_im, jim_im)
    print(f"{name:10s} cpu={cpu_im.size} jim={jim_im.size} PSNR={p:.2f} dB")
    results.append({"name": name, "psnr_db": p, "cpu_size": cpu_im.size, "jim_size": jim_im.size})

if results:
    psnrs = [r["psnr_db"] for r in results]
    print(f"\nMean PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Range: {min(psnrs):.2f} - {max(psnrs):.2f} dB")
    with open(os.path.join(jim_dir, "psnr.json"), "w") as f:
        json.dump({"mean": float(np.mean(psnrs)), "min": min(psnrs), "max": max(psnrs),
                   "results": results}, f, indent=2)
