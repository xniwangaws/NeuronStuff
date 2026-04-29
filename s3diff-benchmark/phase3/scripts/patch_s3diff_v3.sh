#!/bin/bash
# Phase 3 patches to S3Diff repo: make de_mod flow via ContextVar (trace-friendly).
# Idempotent: safe to run multiple times.
set -e

REPO=${1:-/home/ubuntu/s3diff/repo}

if [ ! -d "$REPO/src" ]; then
  echo "Error: $REPO/src not found"
  exit 1
fi

# --- 1. Create de_mod context module (new file) ---
cat > "$REPO/src/de_mod_ctx.py" <<'EOF'
"""Phase 3: thread-local de_mod dict for trace-friendly LoRA forward.

S3Diff's original design writes `module.de_mod = tensor` at runtime, but this
escapes torch_neuronx.trace. Phase 3 routes de_mod through this ContextVar so
the tensor is a traced input to the UNet.
"""
import contextvars

# Dict[layer_name -> Tensor[B, R, R]]
DE_MOD_CTX = contextvars.ContextVar("de_mod_ctx", default=None)


def set_de_mods(mapping):
    """Return a token; caller must reset with it in finally block."""
    return DE_MOD_CTX.set(mapping)


def reset_de_mods(token):
    DE_MOD_CTX.reset(token)


def get_de_mod(layer_name):
    """Lookup de_mod for a layer.  Returns None if no context set (eager mode)."""
    m = DE_MOD_CTX.get()
    if m is None:
        return None
    return m.get(layer_name)
EOF

# --- 2. Patch model.py (my_lora_fwd) ---
# Only patch if not already patched.
MODEL=$REPO/src/model.py
if ! grep -q "de_mod_ctx" "$MODEL"; then
  # Add import at top (after existing imports)
  python3 - <<PY
import re
src = open("$MODEL").read()
# Insert import near other imports
if "from de_mod_ctx import" not in src:
    lines = src.splitlines()
    # Insert after the last existing local import or after line 1
    insert_at = 0
    for i, line in enumerate(lines[:30]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "from de_mod_ctx import get_de_mod as _phase3_get_de_mod")
    src = "\n".join(lines)

# Replace 'self.de_mod' reads with a fallback that prefers ctx
# Original einsum lines:
#   _tmp = torch.einsum('...khw,...kr->...rhw', _tmp, self.de_mod)
# After:
#   _tmp = torch.einsum('...khw,...kr->...rhw', _tmp, _phase3_get_de_mod(getattr(self, '_phase3_layer_name', None)) if _phase3_get_de_mod(getattr(self, '_phase3_layer_name', None)) is not None else self.de_mod)
# Cleaner: use a helper
helper = '''

def _phase3_de_mod(self):
    """Phase 3: prefer ContextVar-supplied de_mod (trace) over self.de_mod (eager)."""
    lname = getattr(self, "_phase3_layer_name", None)
    if lname is not None:
        v = _phase3_get_de_mod(lname)
        if v is not None:
            return v
    return self.de_mod

'''
if "def _phase3_de_mod" not in src:
    # Insert helper BEFORE my_lora_fwd
    src = src.replace("def my_lora_fwd", helper + "def my_lora_fwd", 1)

# Replace self.de_mod in the einsum lines
src = src.replace(
    "_tmp = torch.einsum('...khw,...kr->...rhw', _tmp, self.de_mod)",
    "_tmp = torch.einsum('...khw,...kr->...rhw', _tmp, _phase3_de_mod(self))",
)
src = src.replace(
    "_tmp = torch.einsum('...lk,...kr->...lr', _tmp, self.de_mod)",
    "_tmp = torch.einsum('...lk,...kr->...lr', _tmp, _phase3_de_mod(self))",
)

open("$MODEL", "w").write(src)
print("patched model.py")
PY
else
  echo "model.py already patched (found 'de_mod_ctx')"
fi

# --- 3. Patch s3diff_tile.py: tag each LoRA module with its layer name ---
TILE=$REPO/src/s3diff_tile.py
if ! grep -q "_phase3_layer_name" "$TILE"; then
  python3 - <<PY
src = open("$TILE").read()
# Find the 2 places where LoRA forward is monkey-patched and tag _phase3_layer_name right after
# VAE block (name in self.vae_lora_layers):
src = src.replace(
    "            if name in self.vae_lora_layers:\n                module.forward = my_lora_fwd.__get__(module, module.__class__)",
    "            if name in self.vae_lora_layers:\n                module.forward = my_lora_fwd.__get__(module, module.__class__)\n                module._phase3_layer_name = \"vae::\" + name",
    1,
)
src = src.replace(
    "        for name, module in unet.named_modules():\n            if name in self.unet_lora_layers:\n                module.forward = my_lora_fwd.__get__(module, module.__class__)",
    "        for name, module in unet.named_modules():\n            if name in self.unet_lora_layers:\n                module.forward = my_lora_fwd.__get__(module, module.__class__)\n                module._phase3_layer_name = \"unet::\" + name",
    1,
)
open("$TILE", "w").write(src)
print("patched s3diff_tile.py")
PY
else
  echo "s3diff_tile.py already patched"
fi

echo "=== Phase 3 patch done ==="
grep -n "phase3\|de_mod_ctx" "$MODEL" | head -5
grep -n "_phase3_layer_name" "$TILE" | head -5
