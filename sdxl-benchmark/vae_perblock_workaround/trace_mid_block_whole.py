"""Trace whole mid_block as one NEFF (now that we have --target=trn2).
If this succeeds, replace the 3-split (03a/03b/03c) with a single 03_mid_block to
reduce inter-NEFF bf16 boundary error."""
import sys, time, torch
sys.path.insert(0, "/home/ubuntu/work_a")
import torch_neuronx
from diffusers import AutoencoderKL
from trace_subneffs import MidBlockWrap, COMPILER_ARGS

VAE_PATH = "/home/ubuntu/sdxl-base"
WORK = "/home/ubuntu/work_a"

vae = AutoencoderKL.from_pretrained(
    VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
)
vae.eval()
mod = MidBlockWrap(vae); mod.eval()
x = torch.randn(1, 512, 256, 256, dtype=torch.bfloat16)

t0 = time.time()
traced = torch_neuronx.trace(mod, (x,), compiler_args=COMPILER_ARGS,
                              compiler_workdir=f"{WORK}/compile_03_mid_block_whole")
print(f"trace OK in {time.time()-t0:.1f}s")
torch.jit.save(traced, f"{WORK}/traced_03_mid_block.pt")
print("saved")
