"""
FP8 Linear shim for BFL flux2 repo.

Replaces selected nn.Linear layers in a Flux2 model with FP8Linear that:
  - Holds weight (float8_e4m3fn), weight_scale (float32 scalar), input_scale (float32 scalar)
  - Uses torch._scaled_mm for the matmul
  - Supports BF16 bias (klein uses bias=False everywhere in attention/MLP projections,
    but we keep bias support generic)

Matches the FBGEMM-FP8 per-tensor layout used in the BFL FP8 safetensors ckpt:
  <name>.weight         dtype=float8_e4m3fn, shape=[out_features, in_features]
  <name>.weight_scale   dtype=float32, shape=[]
  <name>.input_scale    dtype=float32, shape=[]
"""

import torch
import torch.nn as nn
from safetensors import safe_open


# Patterns (partial match on module path) that MUST stay BF16 (ckpt has no scales for them).
BF16_KEEP_PATTERNS = (
    "img_in",
    "txt_in",
    "time_in.",          # time_in.in_layer / time_in.out_layer
    "guidance_in.",
    "modulation",        # double_stream_modulation_img / _txt / single_stream_modulation
    "final_layer",       # final_layer.linear + final_layer.adaLN_modulation
)


class FP8Linear(nn.Module):
    """Per-tensor FP8 linear layer using torch._scaled_mm.

    Stores:
      self.weight        : float8_e4m3fn [out_features, in_features]
      self.weight_scale  : float32 scalar buffer
      self.input_scale   : float32 scalar buffer
      self.bias          : optional float/bfloat16 [out_features]

    Forward: y = scaled_mm(quant(x / input_scale), weight.t(),
                           scale_a=input_scale, scale_b=weight_scale,
                           out_dtype=x.dtype, bias=bias)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # placeholder weight in fp8 (real data assigned later via load_fp8_state)
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn,
                        device=device),
            persistent=True,
        )
        self.register_buffer("weight_scale",
                             torch.ones((), dtype=torch.float32, device=device),
                             persistent=True)
        self.register_buffer("input_scale",
                             torch.ones((), dtype=torch.float32, device=device),
                             persistent=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16,
                                                 device=device))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        # Flatten leading dims
        x2 = x.reshape(-1, orig_shape[-1])
        # _scaled_mm requires leading dim multiple of 16
        M = x2.shape[0]
        pad = (16 - (M % 16)) % 16
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, x2.shape[-1])], dim=0)
        # Quantize input to fp8 (per-tensor). Stay in the input dtype (bf16) to avoid
        # allocating a float32 copy of the activation — important at 2K where x2 is large.
        inv_scale = (1.0 / self.input_scale).to(orig_dtype)
        x_fp8 = (x2 * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # free the bf16 mid-allocation early
        del x2
        # Matmul: y = (x_fp8 * sa) @ (w_fp8 * sb).T  = sa * sb * x_fp8 @ w_fp8.T
        y = torch._scaled_mm(
            x_fp8,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=self.bias.to(orig_dtype) if self.bias is not None else None,
            out_dtype=orig_dtype,
        )
        if pad:
            y = y[:M]
        return y.reshape(*orig_shape[:-1], self.out_features)


def _should_replace(qualified_name: str) -> bool:
    """Return True if this Linear path should be replaced with FP8Linear.

    qualified_name is the dotted path inside the Flux2 root module.
    """
    # Explicit BF16 keep-patterns
    for pat in BF16_KEEP_PATTERNS:
        if pat in qualified_name:
            return False
    # Only replace nn.Linear modules inside double_blocks / single_blocks
    return ("double_blocks." in qualified_name) or ("single_blocks." in qualified_name)


def convert_model_to_fp8(model: nn.Module, device: str = "cuda") -> tuple[int, int]:
    """Walk model, swap matching nn.Linear with FP8Linear.

    Returns (n_replaced, n_kept_bf16_linear).
    """
    n_replaced = 0
    n_kept = 0
    # Collect first (can't modify tree during iteration)
    to_replace = []
    to_keep = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if _should_replace(name):
                to_replace.append((name, mod))
            else:
                to_keep.append(name)

    for qname, mod in to_replace:
        parent_name, _, leaf_name = qname.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        in_f, out_f = mod.in_features, mod.out_features
        has_bias = mod.bias is not None
        fp8 = FP8Linear(
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            device=device,
        )
        if leaf_name.isdigit():
            parent[int(leaf_name)] = fp8
        else:
            setattr(parent, leaf_name, fp8)
        # drop reference to old BF16 linear; free memory
        del mod
        n_replaced += 1

    n_kept = len(to_keep)
    return n_replaced, n_kept


@torch.no_grad()
def load_fp8_state_dict(model: nn.Module, safetensors_path: str, device: str = "cuda"):
    """Load the FP8 BFL safetensors into a model that has already been converted.

    For FP8Linear modules: expects <name>.weight (fp8), .weight_scale, .input_scale.
    For BF16 modules: standard .weight / .bias.
    For RMSNorm (QKNorm.{query,key}_norm): .scale (BF16) — falls through to state_dict.
    """
    # We can't use load_state_dict directly because FP8Linear uses buffers and
    # we want strict matching. Build a flat state dict first.
    flat_sd = {}
    with safe_open(safetensors_path, framework="pt", device=str(device)) as sf:
        for k in sf.keys():
            flat_sd[k] = sf.get_tensor(k)

    # Map into model's modules
    missing = []
    unexpected = []
    model_keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, FP8Linear):
            w_key = f"{name}.weight"
            ws_key = f"{name}.weight_scale"
            is_key = f"{name}.input_scale"
            if w_key not in flat_sd:
                missing.append(w_key); continue
            mod.weight.data.copy_(flat_sd[w_key].to(torch.float8_e4m3fn))
            mod.weight_scale.data.copy_(flat_sd[ws_key].to(torch.float32).reshape(()))
            mod.input_scale.data.copy_(flat_sd[is_key].to(torch.float32).reshape(()))
            model_keys.update([w_key, ws_key, is_key])
            if mod.bias is not None:
                b_key = f"{name}.bias"
                if b_key in flat_sd:
                    mod.bias.data.copy_(flat_sd[b_key].to(mod.bias.dtype))
                    model_keys.add(b_key)
        elif isinstance(mod, nn.Linear):
            w_key = f"{name}.weight"
            if w_key in flat_sd:
                mod.weight.data.copy_(flat_sd[w_key].to(mod.weight.dtype))
                model_keys.add(w_key)
            else:
                missing.append(w_key)
            if mod.bias is not None:
                b_key = f"{name}.bias"
                if b_key in flat_sd:
                    mod.bias.data.copy_(flat_sd[b_key].to(mod.bias.dtype))
                    model_keys.add(b_key)
        else:
            # Handle RMSNorm / LayerNorm params the usual way: enumerate parameters+buffers
            for p_name, _ in list(mod.named_parameters(recurse=False)) + \
                            list(mod.named_buffers(recurse=False)):
                full = f"{name}.{p_name}" if name else p_name
                # Only if this param truly belongs to THIS mod (skip redundant top-level passes)
                if full in flat_sd and full not in model_keys:
                    tgt = mod.get_parameter(p_name) if p_name in dict(mod.named_parameters(recurse=False)) else mod.get_buffer(p_name)
                    try:
                        tgt.data.copy_(flat_sd[full].to(tgt.dtype))
                        model_keys.add(full)
                    except Exception:
                        pass

    unexpected = sorted(set(flat_sd.keys()) - model_keys)
    return missing, unexpected
