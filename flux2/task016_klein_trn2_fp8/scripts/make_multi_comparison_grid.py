#!/usr/bin/env python3
"""Create a same-seed BF16/MLP-FP8/all-linear-FP8 comparison grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--all-linear-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=384)
    return parser.parse_args()


def images_by_name(directory: Path) -> dict[str, Path]:
    return {
        path.name: path for path in sorted(directory.glob("seed*_cat.png"))
    }


def cell(image_path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 32), "white")
    canvas.paste(fitted, (0, 32))
    ImageDraw.Draw(canvas).text((8, 9), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    variants = [
        ("BF16", images_by_name(args.bf16_dir)),
        ("FP8 MLP W8A8", images_by_name(args.mlp_dir)),
        ("FP8 all Linear W8A8", images_by_name(args.all_linear_dir)),
    ]
    names = sorted(set.intersection(*(set(images) for _, images in variants)))
    if not names:
        raise SystemExit("No matching seed*_cat.png files found.")

    cell_height = args.cell_size + 32
    grid = Image.new(
        "RGB",
        (args.cell_size * len(variants), cell_height * len(names)),
        "#d0d0d0",
    )
    for row, name in enumerate(names):
        seed = name.removeprefix("seed").split("_", 1)[0]
        for column, (label, images) in enumerate(variants):
            grid.paste(
                cell(images[name], f"seed {seed} — {label}", args.cell_size),
                (column * args.cell_size, row * cell_height),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
