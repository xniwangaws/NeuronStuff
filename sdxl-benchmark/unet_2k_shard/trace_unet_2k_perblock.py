"""Per-block trace of SDXL UNet at latent 256x256 (image 2048x2048), bf16, batch=1.

Splits UNet into independent NEFFs to dodge NCC_EVRF007 / NCC_EOOM002:

  P0: stem        -> conv_in + time/add_embed (small)
  P1: down_blocks[0]   (DownBlock2D, no cross-attn)
  P2: down_blocks[1]   (CrossAttnDownBlock2D, transformer_layers_per_block=2)
  P3: down_blocks[2]   (CrossAttnDownBlock2D, transformer_layers_per_block=10) -- biggest
  P4: mid_block        (CrossAttnMid)
  P5: up_blocks[0]     (CrossAttnUpBlock2D, 3 resnets, transformer_layers_per_block=10) -- biggest
  P6: up_blocks[1]     (CrossAttnUpBlock2D, 3 resnets, transformer_layers_per_block=2)
  P7: up_blocks[2]     (UpBlock2D, 3 resnets, no cross-attn)
  P8: head        -> conv_norm_out + SiLU + conv_out (small)

If any single block still exceeds limits, we'll split that block further.
Each saved NEFF takes a flat tuple of bf16 tensors.
"""
import os, sys, copy, time, math, argparse, traceback
os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

# --- whn09 NKI flash attention (same as trace_unet_2k.py) -------------
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)


def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))
    out = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)
    _flash_fwd_call(q, k, v, scale, out, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    return out.reshape((bs, n_head, q_len, d_head))


class KernelizedAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attention_mask is not None or query.shape[3] > query.shape[2] or query.shape[3] > 128 or value.shape[2] == 77:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = attention_wrapper_without_swap(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


# === Block wrappers ====================================================
# All wrappers take/return only torch.Tensor (no Optional kwargs) so trace works.

class StemWrap(nn.Module):
    """conv_in + time_embedding + add_embedding -> (sample_after_conv_in, emb)"""
    def __init__(self, unet):
        super().__init__()
        self.conv_in = unet.conv_in
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding
        self.add_time_proj = unet.add_time_proj
        self.add_embedding = unet.add_embedding
        # SDXL: time_proj uses Timesteps module producing fp32 sin/cos. We pre-compute outside.
        # Here we accept already-projected timestep features.

    def forward(self, sample, t_emb_in, text_embeds, time_ids):
        # sample: (B,4,256,256), t_emb_in: (B, 320) (already from time_proj),
        # text_embeds: (B,1280), time_ids: (B,6)
        h = self.conv_in(sample)
        emb = self.time_embedding(t_emb_in)  # (B, 1280)
        # SDXL add embedding: cat(text_embeds, time_proj(time_ids)) -> add_embedding
        # add_time_proj is a Timesteps op; we pre-flatten time_ids to apply
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((time_ids.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds.to(text_embeds.dtype)], dim=-1)
        aug_emb = self.add_embedding(add_embeds)
        emb = emb + aug_emb
        return h, emb


class DownBlock0Wrap(nn.Module):
    """down_blocks[0] (DownBlock2D, no cross-attn).
    in: (sample (B,320,256,256), emb (B,1280))
    out: (sample (B,320,128,128), res0 (B,320,256,256), res1 (B,320,256,256), res2 (B,320,128,128))
    """
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
    def forward(self, sample, emb):
        s, residuals = self.blk(sample, emb)
        return s, residuals[0], residuals[1], residuals[2]


class CrossAttnDownBlockWrap(nn.Module):
    """For down_blocks[1] (640) and down_blocks[2] (1280)."""
    def __init__(self, blk, has_downsampler):
        super().__init__()
        self.blk = blk
        self.has_downsampler = has_downsampler
    def forward(self, sample, emb, ehs):
        s, residuals = self.blk(sample, emb, encoder_hidden_states=ehs)
        # down_blocks[1] has 3 residuals (2 attn + 1 downsample)
        # down_blocks[2] has 2 residuals (2 attn, no downsample)
        if self.has_downsampler:
            return s, residuals[0], residuals[1], residuals[2]
        else:
            return s, residuals[0], residuals[1]


class MidBlockWrap(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
    def forward(self, sample, emb, ehs):
        return self.blk(sample, emb, encoder_hidden_states=ehs)


class CrossAttnUpBlockWrap(nn.Module):
    """For up_blocks[0] and up_blocks[1] (both have 3 residuals)."""
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
    def forward(self, sample, res0, res1, res2, emb, ehs):
        return self.blk(sample, (res0, res1, res2), emb, encoder_hidden_states=ehs)


class UpBlock2Wrap(nn.Module):
    """up_blocks[2]: UpBlock2D, no cross-attn, 3 residuals."""
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
    def forward(self, sample, res0, res1, res2, emb):
        return self.blk(sample, (res0, res1, res2), emb)


class HeadWrap(nn.Module):
    """conv_norm_out + SiLU + conv_out."""
    def __init__(self, unet):
        super().__init__()
        self.conv_norm_out = unet.conv_norm_out
        self.act = nn.SiLU()
        self.conv_out = unet.conv_out
    def forward(self, sample):
        x = self.conv_norm_out(sample)
        x = self.act(x)
        return self.conv_out(x)


# === Compile driver ===================================================

WORKDIR = "/home/ubuntu/work_e2e/unet_neff"
COMPILE_BASE = "/home/ubuntu/work_e2e/compile/unet_perblock"
DTYPE = torch.bfloat16
LATENT = 256
B = 1


def make_inputs(unet):
    """Make example inputs for each block by running the UNet wrappers on CPU."""
    sample = torch.randn(B, 4, LATENT, LATENT, dtype=DTYPE)
    timestep = torch.tensor([999.0])
    encoder_hidden_states = torch.randn(B, 77, 2048, dtype=DTYPE)
    text_embeds = torch.randn(B, 1280, dtype=DTYPE)
    time_ids = torch.randn(B, 6, dtype=DTYPE)

    # time_proj is a fp32 module; pre-compute outside the stem to keep stem all-bf16
    t_emb = unet.time_proj(timestep)  # fp32 -> (1, 320)
    t_emb = t_emb.to(DTYPE)
    return {
        "sample": sample, "t_emb": t_emb,
        "encoder_hidden_states": encoder_hidden_states,
        "text_embeds": text_embeds, "time_ids": time_ids,
    }


COMPILER_ARGS_BIG = [
    "--model-type=unet-inference",
    "--lnc=2",
    "--optlevel=1",
    "--tensorizer-options=--inst-count-limit=15000000",
    "--internal-max-instruction-limit=15000000",
]
COMPILER_ARGS_SMALL = [
    "--model-type=unet-inference",
    "--lnc=2",
    "--optlevel=2",
]


def compile_one(name, mod, inputs, big=True):
    """Compile sub-module to NEFF. inputs is a tuple of tensors.
    Apply NKI patch only at trace time (CPU forward of NKI kernel fails)."""
    out_pt = os.path.join(WORKDIR, f"traced_{name}.pt")
    if os.path.exists(out_pt):
        sz = os.path.getsize(out_pt) / 1e6
        print(f"[{name}] EXISTS ({sz:.1f} MB), skipping")
        return out_pt
    workdir = os.path.join(COMPILE_BASE, name)
    os.makedirs(workdir, exist_ok=True)
    print(f"[{name}] tracing... input shapes: {[tuple(t.shape) for t in inputs]}", flush=True)
    # Patch attention with NKI version *just before trace*
    orig_call = diffusers.models.attention_processor.AttnProcessor2_0.__call__
    orig_scores = Attention.get_attention_scores
    diffusers.models.attention_processor.AttnProcessor2_0.__call__ = KernelizedAttnProcessor2_0.__call__
    Attention.get_attention_scores = get_attention_scores_neuron
    try:
        t0 = time.time()
        args = COMPILER_ARGS_BIG if big else COMPILER_ARGS_SMALL
        traced = torch_neuronx.trace(
            mod, inputs,
            compiler_workdir=workdir,
            compiler_args=args,
        )
        torch.jit.save(traced, out_pt)
        el = time.time() - t0
        sz = os.path.getsize(out_pt) / 1e6
        print(f"[{name}] DONE in {el:.1f}s, size={sz:.1f} MB", flush=True)
    finally:
        diffusers.models.attention_processor.AttnProcessor2_0.__call__ = orig_call
        Attention.get_attention_scores = orig_scores
    return out_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="comma-separated block names to compile (e.g. p3_down2,p5_up0)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(COMPILE_BASE, exist_ok=True)

    print(f"[load] UNet (BF16) from /home/ubuntu/sdxl-base", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        "/home/ubuntu/sdxl-base", subfolder="unet", variant="fp16",
        torch_dtype=DTYPE, low_cpu_mem_usage=True,
    )
    unet.eval()

    inp = make_inputs(unet)
    sample0 = inp["sample"]
    t_emb = inp["t_emb"]
    ehs = inp["encoder_hidden_states"]
    text_embeds = inp["text_embeds"]
    time_ids = inp["time_ids"]

    # === Run UNet on CPU once to materialize each block's input shapes/values ===
    print("[cpu-fwd] running UNet on CPU once to capture per-block I/O...", flush=True)
    with torch.no_grad():
        # stem
        stem = StemWrap(unet)
        h0, emb = stem(sample0, t_emb, text_embeds, time_ids)
        print(f"[cpu-fwd] stem out: h0={tuple(h0.shape)} emb={tuple(emb.shape)}")

        # down blocks
        h1, *res0 = DownBlock0Wrap(unet.down_blocks[0])(h0, emb)
        print(f"[cpu-fwd] down0: out={tuple(h1.shape)} residuals={[tuple(r.shape) for r in res0]}")
        # down_blocks[1] has downsampler
        h2_out = CrossAttnDownBlockWrap(unet.down_blocks[1], True)(h1, emb, ehs)
        h2, res1 = h2_out[0], list(h2_out[1:])
        print(f"[cpu-fwd] down1: out={tuple(h2.shape)} residuals={[tuple(r.shape) for r in res1]}")
        # down_blocks[2] no downsampler
        h3_out = CrossAttnDownBlockWrap(unet.down_blocks[2], False)(h2, emb, ehs)
        h3, res2 = h3_out[0], list(h3_out[1:])
        print(f"[cpu-fwd] down2: out={tuple(h3.shape)} residuals={[tuple(r.shape) for r in res2]}")

        # mid
        m = MidBlockWrap(unet.mid_block)(h3, emb, ehs)
        print(f"[cpu-fwd] mid: out={tuple(m.shape)}")

        # up blocks
        # SDXL skip stack (LIFO): conv_in_h0 + res0(3) + res1(3) + res2(2) = 9 residuals popped 3 at a time
        # Actually: down_blocks emit residuals[0..N], they're stacked in order, popped from top.
        # The order of pop for up_blocks[0] is: top 3 of stack = (res2[1], res2[0], res1[2])
        # But down_blocks[2] only emits 2 residuals (no downsampler). So stack ends like:
        #   [h0, r0a, r0b, r0c, r1a, r1b, r1c, r2a, r2b]  -- 9 entries
        # up_blocks[0] pops 3: r2b, r2a, r1c -> tuple in pop order is (r1c, r2a, r2b) (the diffusers code calls res_hidden_states_tuple[-1] then [:-1])
        # We pass them as (r0=last_popped_in_diffusers_order). The diffusers up block expects:
        #   res_hidden_states_tuple = (...all_residuals...) and pops .last each resnet.
        # Easiest: pass last 3 of stack for up0, etc., as a tuple in stack order.
        stack = [h0, res0[0], res0[1], res0[2], res1[0], res1[1], res1[2], res2[0], res2[1]]
        u0_in = (stack[-3], stack[-2], stack[-1])
        u0 = CrossAttnUpBlockWrap(unet.up_blocks[0])(m, *u0_in, emb, ehs)
        print(f"[cpu-fwd] up0: out={tuple(u0.shape)}")
        u1_in = (stack[-6], stack[-5], stack[-4])
        u1 = CrossAttnUpBlockWrap(unet.up_blocks[1])(u0, *u1_in, emb, ehs)
        print(f"[cpu-fwd] up1: out={tuple(u1.shape)}")
        u2_in = (stack[-9], stack[-8], stack[-7])
        u2 = UpBlock2Wrap(unet.up_blocks[2])(u1, *u2_in, emb)
        print(f"[cpu-fwd] up2: out={tuple(u2.shape)}")

        out = HeadWrap(unet)(u2)
        print(f"[cpu-fwd] head: out={tuple(out.shape)}")

    # === Plan ===
    # Each entry: (name, build_module_fn, example_input_tuple, big_compiler_args)
    plan = [
        ("p0_stem", StemWrap(copy.deepcopy(unet)), (sample0, t_emb, text_embeds, time_ids), False),
        ("p1_down0", DownBlock0Wrap(copy.deepcopy(unet.down_blocks[0])), (h0, emb), False),
        ("p2_down1", CrossAttnDownBlockWrap(copy.deepcopy(unet.down_blocks[1]), True), (h1, emb, ehs), True),
        ("p3_down2", CrossAttnDownBlockWrap(copy.deepcopy(unet.down_blocks[2]), False), (h2, emb, ehs), True),
        ("p4_mid", MidBlockWrap(copy.deepcopy(unet.mid_block)), (h3, emb, ehs), True),
        ("p5_up0", CrossAttnUpBlockWrap(copy.deepcopy(unet.up_blocks[0])), (m, stack[-3], stack[-2], stack[-1], emb, ehs), True),
        ("p6_up1", CrossAttnUpBlockWrap(copy.deepcopy(unet.up_blocks[1])), (u0, stack[-6], stack[-5], stack[-4], emb, ehs), True),
        ("p7_up2", UpBlock2Wrap(copy.deepcopy(unet.up_blocks[2])), (u1, stack[-9], stack[-8], stack[-7], emb), False),
        ("p8_head", HeadWrap(copy.deepcopy(unet)), (u2,), False),
    ]
    if args.list:
        for n, _, ex, big in plan:
            print(f"  {n} big={big} input_shapes={[tuple(t.shape) for t in ex]}")
        return

    only = set(args.only.split(",")) if args.only else None
    for name, mod, ex, big in plan:
        if only and name not in only:
            continue
        try:
            mod.eval()
            compile_one(name, mod, ex, big=big)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            # Don't abort whole run — record and move on so we can split later
            with open(os.path.join(WORKDIR, f"FAIL_{name}.txt"), "w") as f:
                f.write(traceback.format_exc())


if __name__ == "__main__":
    main()
