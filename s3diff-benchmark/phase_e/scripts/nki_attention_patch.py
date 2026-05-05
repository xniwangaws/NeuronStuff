"""Monkey-patch for S3Diff UNet self-attention (attn1) using nkilib attention_cte kernel.

Keeps pre-proj (to_q/k/v incl. LoRA), residual, and out-proj in eager. Replaces only the
Q@K^T / softmax / @V block with nkilib.core.attention.attention_cte.attention_cte.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from nkilib.core.attention.attention_cte import attention_cte as _attention_cte_kernel


def _run_attention_cte(q_bhd: torch.Tensor, k_bhd: torch.Tensor, v_bhd: torch.Tensor,
                      scale: float) -> torch.Tensor:
    """Invoke attention_cte kernel.

    Inputs are shape (B*H, S, d) bf16 (query/key/value after view+flatten).
    We invoke with tp_q=True, tp_k=False, tp_out=False:
      q:   (B*H, S, d)
      k:   (B*H, d, S)  <- need a transpose
      v:   (B*H, S, d)
      out: (B*H, S, d)
    """
    # For tp_k=False, k layout is (batch, d, seqlen). Input comes from reshape: (B*H, S, d).
    k_bhd_t = k_bhd.transpose(1, 2).contiguous()  # (B*H, d, S)
    # Kernel is framework-agnostic; the torch binding is called by the kernel dispatcher
    # when inputs are torch.Tensor on Neuron. We must call the underlying callable: the
    # Kernel wrapper object (`Kernel(func=..., lnc=1)`) exposes __call__.
    out = _attention_cte_kernel(
        q_bhd, k_bhd_t, v_bhd,
        scale=float(scale),
        causal_mask=False,
        tp_q=True, tp_k=False, tp_out=False,
    )
    return out  # (B*H, S, d)


class NkiSelfAttnProcessor:
    """Drop-in processor replacing AttnProcessor2_0 for self-attention only.

    Uses nkilib attention_cte for the Q-K-V block. Behaviour outside the kernel
    (residual, norms, LoRA-aware to_q/k/v, to_out) matches AttnProcessor2_0 to keep
    the rest of the graph untouched.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        assert encoder_hidden_states is None, "NkiSelfAttnProcessor is for self-attention (attn1) only"
        assert attention_mask is None, "NkiSelfAttnProcessor expects no attention mask"

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # (B, S, H*D) -> (B, H, S, D) -> (B*H, S, D)
        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).reshape(batch_size * attn.heads,
                                                                                      -1, head_dim).contiguous()
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).reshape(batch_size * attn.heads,
                                                                                    -1, head_dim).contiguous()
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).reshape(batch_size * attn.heads,
                                                                                      -1, head_dim).contiguous()

        if attn.norm_q is not None:
            # norm_q / norm_k in Attention expect (B, H, S, D); we have (B*H, S, D). Reshape back.
            q4 = q.view(batch_size, attn.heads, -1, head_dim)
            q4 = attn.norm_q(q4)
            q = q4.reshape(batch_size * attn.heads, -1, head_dim)
        if attn.norm_k is not None:
            k4 = k.view(batch_size, attn.heads, -1, head_dim)
            k4 = attn.norm_k(k4)
            k = k4.reshape(batch_size * attn.heads, -1, head_dim)

        scale = getattr(attn, "scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Kernel expects inputs on Neuron in bf16. Caller controls dtype.
        out = _run_attention_cte(q, k, v, scale=scale)  # (B*H, S, D)

        # (B*H, S, D) -> (B, H, S, D) -> (B, S, H*D)
        hidden_states = out.view(batch_size, attn.heads, -1, head_dim).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def patch_unet_self_attn(unet) -> int:
    """Replace every attn1 (self-attention) processor in the UNet with NkiSelfAttnProcessor.

    Returns the number of modules patched.
    """
    proc = NkiSelfAttnProcessor()
    n = 0
    for name, mod in unet.named_modules():
        if mod.__class__.__name__ == "Attention" and name.endswith(".attn1"):
            mod.set_processor(proc)
            n += 1
    return n
