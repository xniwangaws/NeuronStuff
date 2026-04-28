"""
Assemble S3Diff benchmark report: unified JSON + side-by-side grids + CSV.

Reads:
  results/{L4,H100}_NVIDIA*_{1K,2K,4K}_bf16.json
  results/trn2_{1K,2K,4K}.json
  results/{L4,H100}_sr_{1K,2K,4K}.png
  results/trn2_sr_{1K,2K,4K}_bridge_v3.png (or similar)

Writes:
  results/summary.csv
  results/summary.md
  results/comparison_{1K,2K,4K}.png  (H100 / L4 / Trn2 side by side)
"""
import csv
import glob
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont


RESULTS = os.path.dirname(os.path.abspath(__file__)) + "/results"


def load_all():
    rows = []
    # GPU
    for f in sorted(glob.glob(f"{RESULTS}/*.json")):
        name = os.path.basename(f)
        if "cpu_vae" in name:
            continue  # skip superseded
        with open(f) as fh:
            d = json.load(fh)
        # Normalize
        row = {
            "device": d.get("device", ""),
            "accel": d.get("gpu_name") or d.get("accel_name", ""),
            "resolution": d.get("resolution", ""),
            "dtype": d.get("dtype", ""),
            "num_runs": d.get("num_runs"),
            "load_time_s": d.get("load_time_s"),
            "cold_start_s": d.get("cold_start_s"),
            "steady_mean_s": d.get("steady_mean_s"),
            "p50_s": d.get("p50_s"),
            "p95_s": d.get("p95_s"),
            "peak_alloc_gb": d.get("peak_alloc_gb"),
            "peak_device_gb": d.get("peak_device_gb") or d.get("peak_hbm_gb"),
            "notes": d.get("traced_components", ""),
            "file": name,
        }
        rows.append(row)
    return rows


def write_csv(rows, out):
    keys = ["accel", "resolution", "dtype", "load_time_s", "cold_start_s",
            "steady_mean_s", "p50_s", "p95_s", "peak_device_gb", "num_runs", "notes"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_comparison(resolution, out_path):
    """Make side-by-side H100 / L4 / Trn2 image for a given resolution.

    Neuron output is bridge-based; GPU outputs were on lq_00 (london2 bus-grid).
    This is a minor inconsistency we note in the report.
    """
    imgs = {}
    for key, patterns in [
        ("H100", [f"{RESULTS}/H100_sr_{resolution}.png"]),
        ("L4",   [f"{RESULTS}/L4_sr_{resolution}.png"]),
        ("Trn2", [f"{RESULTS}/trn2_sr_{resolution}_bridge_v3.png",
                  f"{RESULTS}/trn2_sr_{resolution}_bridge.png",
                  f"{RESULTS}/trn2_sr_{resolution}.png"]),
    ]:
        for p in patterns:
            if os.path.exists(p):
                imgs[key] = Image.open(p).convert("RGB")
                break

    if len(imgs) < 2:
        print(f"[{resolution}] not enough images, skipping comparison")
        return

    # Resize to a common width (640 thumbnail) preserving aspect
    w = 640
    thumbs = []
    for key in ("H100", "L4", "Trn2"):
        if key not in imgs:
            thumbs.append((key, None))
            continue
        im = imgs[key]
        ratio = w / im.width
        h = int(im.height * ratio)
        thumbs.append((key, im.resize((w, h), Image.LANCZOS)))

    # Stack horizontally
    label_h = 40
    thumb_heights = [t[1].height for t in thumbs if t[1] is not None]
    H = max(thumb_heights) + label_h
    total_w = sum((t[1].width if t[1] is not None else w) for t in thumbs)
    canvas = Image.new("RGB", (total_w, H), (20, 20, 20))
    x = 0
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    for key, img in thumbs:
        if img is None:
            continue
        canvas.paste(img, (x, label_h))
        draw.text((x + 8, 8), key, fill=(255, 255, 255), font=font)
        x += img.width

    canvas.save(out_path)
    print(f"[{resolution}] wrote {out_path}")


def main():
    rows = load_all()

    # CSV
    write_csv(rows, f"{RESULTS}/summary.csv")
    print(f"wrote {RESULTS}/summary.csv ({len(rows)} rows)")

    # Markdown table
    lines = [
        "# S3Diff benchmark — 9-cell matrix (3 devices × 3 resolutions, BF16)",
        "",
        "| Device | Res | Load (s) | Cold (s) | Steady (s) | P50 (s) | P95 (s) | Peak device (GB) | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = {"1K": 0, "2K": 1, "4K": 2}
    dev_order = {"NVIDIA H100 80GB HBM3": 0, "NVIDIA L4": 1, "Trainium2 (LNC=2)": 2}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r["resolution"], 99),
                                              dev_order.get(r["accel"], 99)))
    for r in rows_sorted:
        lines.append(
            f"| {r['accel']} | {r['resolution']} | "
            f"{r['load_time_s']} | {r['cold_start_s']} | {r['steady_mean_s']} | "
            f"{r['p50_s']} | {r['p95_s']} | {r['peak_device_gb']} | {r['notes']} |"
        )

    lines += [
        "",
        "## Configuration",
        "- S3Diff: 1-step SR on SD-Turbo UNet, x4 upscale",
        "- BF16 everywhere (official --mixed_precision bf16 on GPU; Neuron UNet traced with --auto-cast=none)",
        "- Neuron: UNet traced on trn2.3xlarge (LNC=2); VAE encode/decode + text encoder + DEResNet on CPU",
        "- Neuron traced at latent tile 96x96 with baked de_mod (valid for the specific LQ image's degradation score)",
        "- GPU baseline uses london2 bus-grid input; Neuron uses Golden Gate Bridge input (input differs — see note below)",
        "",
        "## Notes",
        "- Neuron end-to-end latency dominated by **CPU VAE encode/decode** (~90% of time); Neuron UNet is fast (<5s at 1K tiled).",
        "- For a full apples-to-apples, VAE should also be traced on Neuron (future work).",
        "- Neuron SDK 2.29.0, DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`, venv `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`",
        "- Accuracy: Neuron vs CPU eager on bridge LQ 1K: PSNR 19.7 dB, mean |diff| 14.8/255. Dominated by tile blending seam; raw UNet cosine ~0.9955.",
    ]
    with open(f"{RESULTS}/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {RESULTS}/summary.md")

    # Comparison images
    for res in ("1K", "2K", "4K"):
        make_comparison(res, f"{RESULTS}/comparison_{res}.png")


if __name__ == "__main__":
    main()
