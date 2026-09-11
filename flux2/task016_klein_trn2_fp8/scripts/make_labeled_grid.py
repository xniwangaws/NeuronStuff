#!/usr/bin/env python3
"""Create a same-seed comparison grid from any number of labeled result dirs.

Example:
  python scripts/make_labeled_grid.py \
    --column "BF16, TE/VAE on CPU=results/bf16" \
    --column "BF16, all Neuron=results/bf16_neuron_aux" \
    --column "FP8 all-Linear, all Neuron=results/fp8_all_linear_neuron_aux" \
    --output results/comparison_grid_cpu_aux_vs_neuron_aux.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--column",
        action="append",
        required=True,
        help='"LABEL=DIR"; repeat once per column (left to right).',
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=384)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    return parser.parse_args()


def images_by_name(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob("seed*_cat.png"))}


def cell(image_path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 32), "white")
    canvas.paste(fitted, (0, 32))
    ImageDraw.Draw(canvas).text((8, 9), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    variants = []
    for spec in args.column:
        label, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f'--column expects "LABEL=DIR", got {spec!r}')
        variants.append((label.strip(), images_by_name(Path(directory))))

    names = sorted(set.intersection(*(set(images) for _, images in variants)))
    if args.seeds:
        wanted = {f"seed{s}_cat.png" for s in args.seeds}
        names = [n for n in names if n in wanted]
    if not names:
        raise SystemExit("No matching seed*_cat.png files found across all columns.")

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
                cell(images[name], f"seed {seed} - {label}", args.cell_size),
                (column * args.cell_size, row * cell_height),
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"wrote {args.output} ({len(names)} rows x {len(variants)} columns)")


if __name__ == "__main__":
    main()
