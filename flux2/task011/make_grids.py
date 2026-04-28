#!/usr/bin/env python3
"""Build side-by-side comparison grids for the FLUX.2-dev benchmark matrix.

Rows = 10 fixed-seed prompts (seed 42..51, p00..p09).
Cols = one column per available device/precision at the chosen resolution.
Output:
    results/grid_1024.png
    results/grid_2048.png
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TASK_ROOT = Path("/Users/xniwang/oppo-opencode/working/flux2")
OUT_DIR = TASK_ROOT / "task011" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, absolute directory, resolution)
SOURCES = {
    1024: [
        ("L4 NF4",        TASK_ROOT / "task001/results/l4_nf4_1024"),
        ("H100 BF16",     TASK_ROOT / "task002/results/h100_bf16_1024"),
        ("H100 FP8",      TASK_ROOT / "task002/results/h100_fp8_1024"),
        ("Neuron BF16",   TASK_ROOT / "task010/results/neuron_bf16_1024"),
    ],
    2048: [
        ("H100 BF16",     TASK_ROOT / "task002/results/h100_bf16_2048"),
        ("H100 FP8",      TASK_ROOT / "task002/results/h100_fp8_2048"),
        ("Neuron BF16",   TASK_ROOT / "task010/results/neuron_bf16_2048"),
    ],
}

# 10 prompts -> filename pattern seed00{42+p}_p{pp}.png
NUM_PROMPTS = 10
CELL = 512            # target cell size (pixels, square)
LABEL_H = 40          # header height for column labels
PADDING = 4           # inter-cell gutter (pixels)


def load_font(size: int = 18) -> ImageFont.ImageFont:
    """Pick any available TTF; fall back to PIL's default bitmap font."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


def prompt_filename(p: int) -> str:
    seed = 42 + p
    return f"seed{seed:04d}_p{p:02d}.png"


def available_sources(resolution: int) -> list[tuple[str, Path]]:
    present = []
    for label, d in SOURCES[resolution]:
        sample = d / prompt_filename(0)
        if sample.exists():
            present.append((label, d))
        else:
            print(f"[{resolution}] skip column '{label}' (missing: {sample})")
    return present


def build_grid(resolution: int, out_path: Path) -> None:
    cols = available_sources(resolution)
    if not cols:
        print(f"[{resolution}] no sources available, skipping")
        return

    n_cols = len(cols)
    grid_w = n_cols * CELL + (n_cols + 1) * PADDING
    grid_h = LABEL_H + NUM_PROMPTS * CELL + (NUM_PROMPTS + 1) * PADDING
    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = load_font(20)

    # Column header labels
    for ci, (label, _) in enumerate(cols):
        x = PADDING + ci * (CELL + PADDING)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (CELL - tw) // 2, (LABEL_H - th) // 2),
                  label, fill=(240, 240, 240), font=font)

    # Images
    for pi in range(NUM_PROMPTS):
        y = LABEL_H + PADDING + pi * (CELL + PADDING)
        for ci, (_, d) in enumerate(cols):
            x = PADDING + ci * (CELL + PADDING)
            fp = d / prompt_filename(pi)
            try:
                im = Image.open(fp).convert("RGB")
            except FileNotFoundError:
                draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 10, 10))
                continue
            if im.size != (CELL, CELL):
                im = im.resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im, (x, y))

    canvas.save(out_path, optimize=True)
    print(f"[{resolution}] wrote {out_path}  {canvas.size[0]}x{canvas.size[1]}  "
          f"({n_cols} cols x {NUM_PROMPTS} rows)")


def main() -> None:
    build_grid(1024, OUT_DIR / "grid_1024.png")
    build_grid(2048, OUT_DIR / "grid_2048.png")


if __name__ == "__main__":
    main()
