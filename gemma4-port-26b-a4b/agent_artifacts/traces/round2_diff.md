# Round 2 Diff: round-1 vs PR #106 (gemma-4-31B-IT) vs real 26B-A4B config

Date: 2026-06-03

## A. Round-1 model dimensions vs real `config.json` (FIRST CRITICAL FIX)

Round 1 had to *guess* the 26B-A4B params (gated HF model). The real config
has these surprises:

| Field | Round-1 guess | Actual | Impact |
|---|---|---|---|
| `hidden_size` | 2304 | **2816** | All linear shapes wrong |
| `num_attention_heads` | 8 | **16** | TP plan wrong |
| `num_key_value_heads` (sliding) | 4 | **8** | KV cache wrong |
| `num_global_key_value_heads` | 2 | 2 | OK |
| `intermediate_size` (dense MLP) | 9216 | **2112** | MLP much smaller than guessed |
| `moe_intermediate_size` | – | **704** | – |
| `num_experts` | – | 128 | OK |
| `top_k_experts` | – | 8 | OK |
| `final_logit_softcapping` | none | **30.0** | Output logits must be tanh-clamped |
| `hidden_size_per_layer_input` | 256 (assumed) | **0** | **No PLE on 26B-A4B**, all PLE code dead |
| `vocab_size_per_layer_input` | 262144 | 262144 | irrelevant if PLE is 0 |
| `num_kv_shared_layers` | 0 | 0 | OK |
| `enable_moe_block` | True | True | OK |
| `attention_k_eq_v` | True | True | OK |
| `head_dim` (sliding) / `global_head_dim` | 256 / 512 | 256 / 512 | OK |
| `layer_types` pattern | "5 sliding : 1 full" | confirmed (last layer == full) | OK |

**Decision**: drop `Gemma4PLE_u`, drop the per-layer-embedding gate logic in
the decoder. Use HF config directly via `load_config` rather than relying on
the shim defaults.

## B. Architecture diffs round-1 vs PR #106 (gemma-4-31B-IT)

Both PR #106 and round-1 inherit from `NeuronAttentionBase`. Diffs that matter:

### B.1 Norm strategy
- PR #106 uses `Gemma4RMSNorm` everywhere (with `weight` parameter, multiply
  by `weight`). Q/K/V follow the same RMSNorm style. Final lm_head wrapped
  in `SoftcappedLMHead`.
- Round-1 uses `nn.LayerNorm` for the *outer* norms based on a generic-MoE
  knowledge-base lesson, and `CustomRMSNorm` for inner norms.
- **PR #106 is validated on 31B-IT and produces coherent text.** Round-1's
  outer-LayerNorm pattern was speculative for gemma4. Switch to PR #106's
  RMSNorm-everywhere convention.

### B.2 attention `forward` style
- PR #106 inherits `forward` from `NeuronAttentionBase`, only overrides:
  - `apply_rotary_embedding` (partial RoPE for global)
  - `prep_qkv_tensors` (post-projection v_norm)
  - `perform_prefill` (NKI d=256 SWA kernel hook)
  - and *passes `sliding_window=None`* to base, doing windowed masking at
    the decoder layer level (Discovery #27 in their PR).
- Round-1 invents `_apply_qk_norm` / `_apply_v_norm` hooks which **do not
  exist in NeuronAttentionBase**. Those wouldn't actually fire — silent bug.
- **Borrow PR #106's hook pattern verbatim.**

### B.3 KV cache management
- PR #106 has a custom `Gemma4KVCacheManager` with per-layer kv shapes
  (16 heads × 256 dim for SWA, 4 heads × 512 dim for global). For 26B-A4B
  the per-layer kv shapes are (8 × 256) and (2 × 512); same pattern.
- Round-1 doesn't override the cache manager — would fail at compile because
  `NeuronAttentionBase` allocates a uniform cache.
- **Borrow `Gemma4KVCacheManager` and adapt counts to 26B-A4B.**

### B.4 QK-scaling-in-norm-weight trick
- PR #106 cancels NxDI's automatic `1/sqrt(head_dim)` by pre-scaling the
  q_norm WEIGHTS by `sqrt(head_dim)` (gemma4 uses `scaling=1.0`).
  This is a state-dict-level fix, not a forward override.
- Round-1 didn't address it at all — would compile fine but produce wrong
  numerics.
- **Borrow the q_layernorm weight-scale step.**

### B.5 Final logit softcapping
- PR #106: `SoftcappedLMHead` wraps the lm_head linear, applies
  `cap * tanh(x / cap)` with cap=30.0 in fp32.
- Round-1: missing entirely.
- **Borrow `SoftcappedLMHead`.**

### B.6 NKI flash-attention kernels
- Both 31B-IT and 26B-A4B have `head_dim=256` (sliding) and
  `head_dim=512` (full). Same kernels apply directly:
  - `nki_flash_attn_d256_swa.py` → SWA layers (window=1024, d=256)
  - `nki_flash_attn_large_d.py` → global layers (d=512)
  - `ndxi_patch.py.apply_patch()` monkey-patches `NeuronAttentionBase`
    to swap kernels in for `head_dim > 128`.
- Round-1 didn't reference any kernel.
- **Copy all three files verbatim into `neuron_port/`.**

## C. MoE: round-1's contribution that PR #106 lacks

PR #106 is for 31B-IT which is **dense**, no MoE. Round-1's MoE block is
genuinely 26B-A4B-specific work to keep:

- `NeuronGemma4Router_u` (with `scale` and `per_expert_scale` learned tensors,
  fp32 routing, softmax-then-topk, top-k normalization, scalar root-size).
  Matches HF `Gemma4TextRouter` line-for-line.
- `NeuronGemma4MoEBlock_u` (calls `initialize_moe_module` from NxDI moe_v2).
- Decoder layer composes dense MLP + parallel MoE branch following HF
  source lines 1429-1441 (router reads pre-MLP residual, mlp output and MoE
  output combined as `mlp_branch + moe_branch`).

## D. Final round-2 plan

1. Drop PLE entirely (not used on 26B-A4B).
2. Replace outer LayerNorm with PR #106's `Gemma4RMSNorm` everywhere.
3. Replace the speculative `_apply_qk_norm` hook with PR #106's
   `prep_qkv_tensors` override (apply v_norm there).
4. Add `apply_rotary_embedding` partial-RoPE override.
5. Add `Gemma4KVCacheManager` with 26B-A4B-specific layer kv configs.
6. Add `SoftcappedLMHead` (cap=30.0).
7. Wire `convert_hf_to_neuron_state_dict` to:
   - Strip HF prefixes (`language_model.`, `model.`, etc.)
   - Pre-scale `q_layernorm.weight` by `sqrt(head_dim)` (per-layer head_dim)
   - Copy `k_proj.weight` → `v_proj.weight` for global layers
   - Tied lm_head (handle `lm_head.linear.weight` for the softcap wrapper)
   - Add `rank_util.rank` tensors
8. Keep MoE router, MoE block, expert weight names identical to HF
   (`router.proj.weight`, `router.scale`, `router.per_expert_scale`,
   `experts.gate_up_proj`, `experts.down_proj`).
9. Add `apply_patch()` call site documented in README so users invoke it
   before constructing the model class.
