"""Inspect SDXL VAE decoder structure and capture intermediate shapes via CPU forward."""
import json
import torch
from diffusers import AutoencoderKL

VAE_PATH = "/home/ubuntu/sdxl-base"
OUT = "/home/ubuntu/work_a/shape_map.json"

vae = AutoencoderKL.from_pretrained(
    VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
)
vae.eval()

# Print named modules at low depth.
print("=== Top-level named_modules (decoder + post_quant_conv) ===")
for name, m in vae.named_modules():
    depth = name.count(".")
    if depth <= 3 and (name.startswith("decoder") or name.startswith("post_quant_conv") or name == ""):
        print(f"  {name}: {type(m).__name__}")

shape_log = {}

def hook(name):
    def fn(mod, inp, out):
        if isinstance(out, torch.Tensor):
            shape_log[name] = list(out.shape)
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            shape_log[name] = list(out[0].shape)
    return fn

handles = []
# Hook key submodules.
targets = ["post_quant_conv",
           "decoder.conv_in",
           "decoder.mid_block",
           "decoder.up_blocks.0",
           "decoder.up_blocks.1",
           "decoder.up_blocks.2",
           "decoder.up_blocks.3",
           "decoder.conv_norm_out",
           "decoder.conv_act",
           "decoder.conv_out"]

# Hook nested resnets / upsamplers within each up_block too.
for name, m in vae.named_modules():
    for t in targets:
        if name == t:
            handles.append(m.register_forward_hook(hook(name)))
    if name.startswith("decoder.up_blocks") and (name.endswith(".upsamplers.0") or ".resnets." in name):
        # only the final shapes from each resnet/upsampler
        if name.count(".") <= 4:
            handles.append(m.register_forward_hook(hook(name)))
    if name == "decoder.mid_block.resnets.0" or name == "decoder.mid_block.resnets.1" or name == "decoder.mid_block.attentions.0":
        handles.append(m.register_forward_hook(hook(name)))

with torch.no_grad():
    latent = torch.randn(1, 4, 256, 256, dtype=torch.bfloat16)
    image = vae.decode(latent).sample

shape_log["INPUT_latent"] = list(latent.shape)
shape_log["OUTPUT_image"] = list(image.shape)

for h in handles:
    h.remove()

with open(OUT, "w") as f:
    json.dump(shape_log, f, indent=2, sort_keys=True)

print("\n=== Captured shapes ===")
for k in sorted(shape_log.keys()):
    print(f"  {k}: {shape_log[k]}")
print(f"\nWrote {OUT}")
