"""ChainedVAEDecoder — wraps a list of pre-traced sub-NEFFs and chains them
to reproduce the SDXL VAE decoder forward at 2048x2048 output.

The chain order matches what `trace_subneffs.py` produced:
    01_post_quant_conv  -> 02_conv_in -> 03a_mid_resnet0 -> 03b_mid_attn ->
    03c_mid_resnet1     -> 04_up_block_0 -> 05_up_block_1 -> 06_up_block_2 ->
    07_up_block_3       -> 08_conv_out_block

Each sub-NEFF was traced to take a single bf16 tensor and return a single
bf16 tensor (we wrapped the temb=None / kwargs quirks at trace time), so the
runtime chain is just a sequential apply.
"""
from pathlib import Path
import torch
import torch.nn as nn

DEFAULT_ORDER = [
    "01_post_quant_conv",
    "02_conv_in",
    "03a_mid_resnet0",
    "03b_mid_attn",
    "03c_mid_resnet1",
    "04_up_block_0",
    "05_up_block_1",
    "06_up_block_2",
    "07_up_block_3",
    "08_conv_out_block",
]


class ChainedVAEDecoder(nn.Module):
    """Calls each sub-NEFF in sequence. All tensors stay bf16 between stages."""
    def __init__(self, work_dir, order=None, device_load=True):
        super().__init__()
        self.work_dir = Path(work_dir)
        self.order = list(order or DEFAULT_ORDER)
        self.subs = nn.ModuleList()
        self.names = []
        for n in self.order:
            pt = self.work_dir / f"traced_{n}.pt"
            if not pt.exists():
                raise FileNotFoundError(pt)
            m = torch.jit.load(str(pt))
            self.subs.append(m)
            self.names.append(n)

    def forward(self, latent):
        x = latent
        for n, m in zip(self.names, self.subs):
            x = m(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
        return x
