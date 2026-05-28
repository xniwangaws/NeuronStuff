"""ChainedUNet — wraps the 9 per-block NEFFs and chains them as the SDXL UNet
forward at latent 256x256 (image 2048x2048).

Each sub-NEFF was traced with explicit positional-tensor inputs so we don't have
to deal with kwargs at runtime. The skip-connection threading mirrors diffusers'
own UNet2DConditionModel.forward but uses our pre-traced sub-modules.

Order of execution:
   p0_stem(sample, t_emb, text_embeds, time_ids) -> (h0, emb)
   p1_down0(h0, emb)                              -> (h1, r0a, r0b, r0c)
   p2_down1(h1, emb, ehs)                         -> (h2, r1a, r1b, r1c)
   p3_down2(h2, emb, ehs)                         -> (h3, r2a, r2b)
   p4_mid (h3, emb, ehs)                          -> m
   p5_up0 (m, r1c, r2a, r2b, emb, ehs)            -> u0   (top 3 of stack)
   p6_up1 (u0, r0c, r1a, r1b, emb, ehs)           -> u1   (next 3 of stack)
   p7_up2 (u1, h0, r0a, r0b, emb)                 -> u2   (bottom 3 of stack)
   p8_head(u2)                                    -> output

The full skip stack after all down blocks (in append order) is:
   [h0, r0a, r0b, r0c, r1a, r1b, r1c, r2a, r2b]   -- 9 entries
Up blocks pop in groups of 3 from the top.
"""
from pathlib import Path
import torch
import torch.nn as nn

DEFAULT_ORDER = [
    "p0_stem",
    "p1_down0",
    "p2_down1",
    "p3_down2",
    "p4_mid",
    "p5_up0",
    "p6_up1",
    "p7_up2",
    "p8_head",
]


class ChainedUNet(nn.Module):
    """Run each sub-NEFF in sequence with proper skip threading."""
    def __init__(self, work_dir, order=None):
        super().__init__()
        self.work_dir = Path(work_dir)
        self.order = list(order or DEFAULT_ORDER)
        self.subs = nn.ModuleDict()
        for n in self.order:
            pt = self.work_dir / f"traced_{n}.pt"
            if not pt.exists():
                raise FileNotFoundError(pt)
            self.subs[n] = torch.jit.load(str(pt))

    def forward(self, sample, t_emb, text_embeds, time_ids, encoder_hidden_states):
        """sample: (B,4,256,256)
        t_emb: (B,320) -- pre-projected time embedding (from time_proj on host)
        text_embeds: (B,1280)
        time_ids: (B,6)
        encoder_hidden_states: (B,77,2048)
        Returns: (B,4,256,256)
        """
        ehs = encoder_hidden_states
        h0, emb = self.subs["p0_stem"](sample, t_emb, text_embeds, time_ids)
        h1, r0a, r0b, r0c = self.subs["p1_down0"](h0, emb)
        h2, r1a, r1b, r1c = self.subs["p2_down1"](h1, emb, ehs)
        h3, r2a, r2b = self.subs["p3_down2"](h2, emb, ehs)
        m = self.subs["p4_mid"](h3, emb, ehs)
        u0 = self.subs["p5_up0"](m, r1c, r2a, r2b, emb, ehs)
        u1 = self.subs["p6_up1"](u0, r0c, r1a, r1b, emb, ehs)
        u2 = self.subs["p7_up2"](u1, h0, r0a, r0b, emb)
        out = self.subs["p8_head"](u2)
        return out
