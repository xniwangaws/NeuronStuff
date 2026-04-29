"""
Build 3-device comparison grids from results/images/{l4,trn2,h100}/prompt{K}_steps{N}.png.

Output: results/grid/prompt{K}_steps{N}.png — horizontal stack [L4 | Trn2 | H100]
with device labels at the top.
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

DEVICES = ["l4", "neuron", "h100"]
LABELS = {"l4": "L4 (g6.4xlarge) BF16",
          "neuron": "Trn2 (trn2.3xlarge) BF16",
          "h100": "H100 (p5.4xlarge) BF16"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--steps", type=int, nargs="+", default=[25, 50])
    p.add_argument("--n_prompts", type=int, default=5)
    return p.parse_args()


def load_font():
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, 28)
    return ImageFont.load_default()


def main():
    args = parse_args()
    out_dir = os.path.join(args.root, "grid")
    os.makedirs(out_dir, exist_ok=True)
    font = load_font()

    for steps in args.steps:
        for k in range(args.n_prompts):
            imgs = []
            for dev in DEVICES:
                path = os.path.join(args.root, "images", dev, f"prompt{k}_steps{steps}.png")
                if not os.path.exists(path):
                    print(f"[skip] missing {path}")
                    imgs = []
                    break
                imgs.append(Image.open(path).convert("RGB"))
            if not imgs:
                continue

            w, h = imgs[0].size
            label_h = 48
            canvas = Image.new("RGB", (w * len(imgs), h + label_h), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (img, dev) in enumerate(zip(imgs, DEVICES)):
                draw.text((i * w + 10, 10), LABELS[dev], fill="black", font=font)
                canvas.paste(img, (i * w, label_h))

            out = os.path.join(out_dir, f"prompt{k}_steps{steps}.png")
            canvas.save(out)
            print(f"[grid] {out}")


if __name__ == "__main__":
    main()
