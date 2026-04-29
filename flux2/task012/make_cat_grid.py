from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/xniwang/oppo-opencode/working/flux2")
SOURCES = [
    ("L4 NF4",         ROOT / "task012/results/l4_nf4_1024_cat_50step"),
    ("H100 BF16",      ROOT / "task012/results/h100_bf16_1024_cat_50step"),
    ("H100 FP8",       ROOT / "task012/results/h100_fp8_1024_cat_50step"),
    ("Neuron BF16",    ROOT / "task012/results/bench_neuron_cat_50/neuron_bf16_1024_cat_50step"),
]
CELL = 512
LABEL = 40
PAD = 4
N = 10

font = None
for c in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
    if Path(c).exists():
        font = ImageFont.truetype(c, 20); break
font = font or ImageFont.load_default()

cols = [(lbl, d) for lbl, d in SOURCES if (d / "seed0042_cat.png").exists()]
w = len(cols) * CELL + (len(cols) + 1) * PAD
h = LABEL + N * CELL + (N + 1) * PAD
img = Image.new("RGB", (w, h), (20, 20, 20))
draw = ImageDraw.Draw(img)
for ci, (lbl, _) in enumerate(cols):
    x = PAD + ci * (CELL + PAD)
    bbox = draw.textbbox((0, 0), lbl, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (CELL - tw)//2, (LABEL - th)//2), lbl, fill=(240, 240, 240), font=font)
for pi in range(N):
    y = LABEL + PAD + pi * (CELL + PAD)
    for ci, (_, d) in enumerate(cols):
        x = PAD + ci * (CELL + PAD)
        fp = d / f"seed{42 + pi:04d}_cat.png"
        im = Image.open(fp).convert("RGB")
        if im.size != (CELL, CELL):
            im = im.resize((CELL, CELL), Image.LANCZOS)
        img.paste(im, (x, y))

out = ROOT / "task012/results/grid_1024_cat_50step.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, optimize=True)
print(f"wrote {out}  {img.size}")
