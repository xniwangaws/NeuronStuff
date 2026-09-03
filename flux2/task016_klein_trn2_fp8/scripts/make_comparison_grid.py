#!/usr/bin/env python3
"""Create a compact same-seed BF16/FP8 visual comparison grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--fp8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=384)
    return parser.parse_args()


def cell(image_path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 32), "white")
    canvas.paste(fitted, (0, 32))
    ImageDraw.Draw(canvas).text((8, 9), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    bf16_images = {
        path.name: path for path in sorted(args.bf16_dir.glob("seed*_cat.png"))
    }
    fp8_images = {
        path.name: path for path in sorted(args.fp8_dir.glob("seed*_cat.png"))
    }
    names = sorted(bf16_images.keys() & fp8_images.keys())
    if not names:
        raise SystemExit("No matching seed*_cat.png files found.")

    cell_height = args.cell_size + 32
    grid = Image.new(
        "RGB",
        (args.cell_size * 2, cell_height * len(names)),
        "#d0d0d0",
    )
    for row, name in enumerate(names):
        seed = name.removeprefix("seed").split("_", 1)[0]
        grid.paste(
            cell(bf16_images[name], f"seed {seed} — BF16", args.cell_size),
            (0, row * cell_height),
        )
        grid.paste(
            cell(
                fp8_images[name],
                f"seed {seed} — FP8 MLP dynamic",
                args.cell_size,
            ),
            (args.cell_size, row * cell_height),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
