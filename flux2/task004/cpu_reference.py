"""CPU BF16 reference for FLUX.2-dev on Neuron host.

Generates golden latents + final image for the NxDI port validation
(cpu_mode() comparison per nxdi-model-porting.md:133). Small resolution (512²)
to keep CPU runtime manageable.

Usage: python cpu_reference.py [--resolution 512] [--prompt-index 0]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline


PROMPTS = [
    "a high-resolution photograph of a red panda sitting on a tree branch in a misty forest, volumetric lighting",
    "an oil painting of a medieval castle at sunset, dramatic clouds, style of Caspar David Friedrich",
    "a futuristic cyberpunk city street at night, neon signs in Japanese, rain-slicked pavement, cinematic",
]

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--model", default="/home/ubuntu/flux2_weights")
    parser.add_argument("--out-dir", default="cpu_reference")
    parser.add_argument("--prompt-index", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED + args.prompt_index)

    print(f"[load] CPU BF16 load of {args.model}")
    t0 = time.perf_counter()
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # CPU stays on CPU — no .to() / no offload
    load_time = time.perf_counter() - t0
    print(f"[load] done in {load_time:.1f}s")

    # Capture intermediate tensors for later NxDI comparison.
    prompt = PROMPTS[args.prompt_index]
    gen = torch.Generator(device="cpu").manual_seed(SEED + args.prompt_index)

    print(f"[gen] {args.resolution}² x {args.steps} steps: '{prompt[:60]}...'")
    t = time.perf_counter()
    result = pipe(
        prompt=prompt,
        height=args.resolution, width=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
        output_type="pt",  # return tensors instead of PIL
        return_dict=True,
    )
    dt = time.perf_counter() - t
    image_tensor = result.images[0]
    print(f"[gen] done in {dt:.1f}s, tensor shape={tuple(image_tensor.shape)}, dtype={image_tensor.dtype}")

    # Save outputs
    torch.save(image_tensor.detach().cpu(), out / f"image_tensor_p{args.prompt_index:02d}.pt")
    from torchvision.transforms.functional import to_pil_image
    to_pil_image((image_tensor.clamp(0, 1) * 255).to(torch.uint8)).save(
        out / f"image_p{args.prompt_index:02d}.png"
    )
    meta = {
        "model": args.model,
        "precision": "BF16",
        "device": "cpu",
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": SEED + args.prompt_index,
        "prompt": prompt,
        "prompt_index": args.prompt_index,
        "load_time_s": load_time,
        "inference_time_s": dt,
        "image_tensor_shape": list(image_tensor.shape),
    }
    (out / f"meta_p{args.prompt_index:02d}.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out}/image_p{args.prompt_index:02d}.png and tensor")


if __name__ == "__main__":
    main()
