# Gemma-4-26B-A4B-it Architecture Analysis

**Source**: `transformers_src/src/transformers/models/gemma4/{modeling,configuration}_gemma4.py`
**Target**: NeuronX Distributed Inference (NxDI), AWS Trainium (trn2.x or trn1.32xlarge)
**Mode**: Dry-run (no hardware)

---

## 1. Model overview

Gemma-4-26B-A4B-it is Google's first **MoE Gemma**. From the customer brief and the strict
config defaults the 26B-A4B variant is parameterised as follows (overrides on top of
`Gemma4TextConfig`):

| Field | Value |
|---|---|
| `model_type` | `gemma4_text` (text decoder); top-level container is `gemma4` |
| `num_hidden_layers` | 30 |
| `hidden_size` | 2304 |
| `intermediate_size` (dense MLP) | 9216 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` (sliding) | 4 |
| `num_global_key_value_heads` | 2 (full-attn layers, K=V) |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (full) | 512 |
| `attention_k_eq_v` | True (V projection reuses K, on full layers only) |
| `vocab_size` | 262_144 |
| `max_position_embeddings` | 131_072 (extendable to 256K with rope scaling) |
| `sliding_window` | 1024 (per the customer brief; default config is 512) |
| `enable_moe_block` | True |
| `num_experts` | 128 |
| `top_k_experts` | 8 |
| `moe_intermediate_size` | per checkpoint (≈ intermediate_size for the A4B sizing) |
| `tie_word_embeddings` | True |
| `rms_norm_eps` | 1e-6 |
| `attention_bias` | False |
| `hidden_activation` | `gelu_pytorch_tanh` |
| Per-Layer-Embeddings (PLE) | enabled (`hidden_size_per_layer_input=256`, `vocab_size_per_layer_input=262144`) |

Layer types follow a **5:1 sliding/full pattern** (`(i+1) % 6 == 0` ⇒ `full_attention`),
last layer forced to `full_attention`. With 30 layers that gives 5 full-attention layers
(indices 5, 11, 17, 23, 29) and 25 sliding-attention layers.

## 2. Architectural components (HF source)

### 2.1 Decoder layer (`Gemma4TextDecoderLayer`, line 1370)

```
hidden_states (x)
  ├── input_layernorm (RMSNorm)
  ├── self_attn (Gemma4TextAttention)
  ├── post_attention_layernorm (RMSNorm)   <-- gemma2/3 pattern: post-norm + residual after attn
  ├── + residual
  ├── pre_feedforward_layernorm (RMSNorm)
  ├── mlp (Gemma4TextMLP, dense SwiGLU)    <-- always present
  ├── if enable_moe_block:
  │     mlp_branch  = post_feedforward_layernorm_1(mlp(x))
  │     moe_branch  = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(residual_pre_mlp_flat), top_k_index, top_k_weights))
  │     hidden      = mlp_branch + moe_branch
  ├── post_feedforward_layernorm (RMSNorm)
  ├── + residual
  ├── if PLE enabled:
  │     PLE residual block (input_gate -> activation -> mul per-layer-input -> projection -> norm + residual)
  └── * layer_scalar (learned 1-d buffer)
```

Key architectural facts driving the port:
- **Dense MLP + sparse MoE run in parallel** every MoE layer. Routing input is the
  *pre-MLP residual* (i.e. the post-attention output), not the MLP's input — this is a
  Gemma4-specific quirk vs Mixtral / Qwen3-MoE.
- The MoE branch goes through both a `pre_feedforward_layernorm_2` (RMSNorm, no scale)
  and a `post_feedforward_layernorm_2` (RMSNorm). The router itself also has its own
  internal RMSNorm.
- `layer_scalar` is a learned scalar applied at the end of each layer.
- PLE injects an extra residual signal computed from a separate, smaller embedding table
  (`embed_tokens_per_layer`), per-layer.

### 2.2 Attention (`Gemma4TextAttention`, line 1178)

Hybrid per-layer behaviour:
- **Sliding layers** (`is_sliding=True`): `head_dim=256`, `num_key_value_heads=4`, full
  Q+K+V projections, sliding-window mask, RoPE with `rope_type="default"` and `rope_theta=10_000`.
- **Full layers**: `head_dim=512` (`global_head_dim`), `num_key_value_heads=2`
  (`num_global_key_value_heads`), partial RoPE (`partial_rotary_factor=0.25`, only first
  25 % of head_dim is rotated, `rope_theta=1_000_000`), and **K=V** (`attention_k_eq_v=True`):
  `v_proj` is `None` and value tensors reuse `key_states` *before* RoPE/k_norm. This is
  effectively a unified KV projection on the full-attention layers.
- **Q/K RMSNorm** (`q_norm`, `k_norm`) on per-head dimension after projection (Qwen3-style).
- **V RMSNorm** without scale (`with_scale=False`) — this is a no-op-scale normalisation
  that just centres magnitude.
- **KV sharing**: `num_kv_shared_layers` consecutive layers at the *end* of the stack can
  share KV (controlled via `shared_kv_states` dict). Configured as 0 in 26B-A4B by default.
- No attention bias, no softcap on logits.

### 2.3 RoPE (`Gemma4TextRotaryEmbedding`, line 1088)

Two independent RoPE configurations (one per `layer_type`):
- `sliding_attention`: `rope_type="default"`, `rope_theta=10_000`, full-rotary, dim = head_dim (256).
- `full_attention`: `rope_type="proportional"`, `rope_theta=1_000_000`,
  `partial_rotary_factor=0.25`, dim = global_head_dim (512). Only the **first 128** dims
  out of 512 are rotated; the rest pass through unchanged.

`apply_rotary_pos_emb` is the standard rotate-half formulation. NxDI's `RotaryEmbedding`
supports both `rope_theta` and `partial_rotary_factor`, so the partial p-RoPE is a
supported configuration; we just instantiate **two RotaryEmbedding objects per attention
class**, one per layer type, and pick by `layer_type`.

### 2.4 MoE block

- `Gemma4TextRouter`: RMSNorm (no scale) → scale parameter → projection → softmax → top-K
  → re-normalise → per-expert scale (additional learned scalar). Inputs are the flattened
  pre-MLP residual `[B*S, H]`.
- `Gemma4TextExperts`: stores `gate_up_proj` `[E, 2*I, H]` and `down_proj` `[E, H, I]`
  as 3-D parameters (no per-expert nn.Linear). Forward dispatches via one-hot expert
  mask, gathers tokens per active expert, computes `act(gate)*up` then `down`, weights
  by `top_k_weights`, and `index_add`s into a zero buffer.
- 128 experts total, 8 active per token. **No "shared expert"** in the source code path
  (the customer brief mentions "8 active / 128 total + 1 shared"; the +1 shared is
  effectively the dense `mlp` branch that runs in parallel with the expert mixture and
  is summed in. This is the Gemma4 design, not a separate shared-expert weight tensor.)

### 2.5 Embeddings, norm, head

- `Gemma4TextScaledWordEmbedding`: nn.Embedding scaled by `sqrt(hidden_size)` at fwd time.
- Final `Gemma4RMSNorm` (eps=1e-6) before `lm_head`.
- `lm_head`: standard `nn.Linear(hidden_size, vocab_size, bias=False)`.
- **Tied weights**: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`.

### 2.6 Multimodal (vision + audio)

Out of scope for the *first* compile. The 26B-A4B is multimodal in HF
(`Gemma4ForConditionalGeneration`), but the port targets the **text-only** causal-LM
backbone first (`Gemma4ForCausalLM`). Vision tower (`Gemma4VisionModel`,
`Gemma4MultimodalEmbedder`, audio tower) are deferred — they live in the source but
are not wired in `modeling_gemma4_neuron.py`. We document this as a follow-up.

## 3. Comparison: gemma3 vs gemma4

Notable additions in gemma4 (not in gemma3):
1. **MoE block** (`enable_moe_block`, `Gemma4TextRouter`, `Gemma4TextExperts`).
2. **Per-Layer-Embeddings** (PLE: `embed_tokens_per_layer`,
   `per_layer_model_projection`, `per_layer_input_gate`, `per_layer_projection`).
3. **K=V unified projection** on full layers (`attention_k_eq_v`).
4. **Asymmetric heads** between sliding (256-dim, 4 KV) and full (512-dim, 2 KV).
5. **Partial RoPE** on full-attn layers only (proportional p-RoPE).
6. **KV sharing** option across the last N layers (off by default for 26B-A4B).
7. **`layer_scalar`** learned residual scaling at end of each decoder layer.
8. **`v_norm`** RMSNorm without scale (centring-only).

## 4. NxDI mapping plan

| Gemma4 component | NxDI reuse | Notes |
|---|---|---|
| `Gemma4RMSNorm` | `CustomRMSNorm` (or torch RMSNorm via base attn) | But: per genericmoe_v16 the *outer* layernorms (input_layernorm, post_attention_layernorm, pre/post_feedforward_layernorm, final norm) **must be `nn.LayerNorm`** to avoid gibberish. Q/K/V/router internal norms keep RMSNorm semantics. |
| `Gemma4TextMLP` (dense) | `ColumnParallelLinear` + `RowParallelLinear` | SwiGLU pattern, gemma4-specific: `gelu_pytorch_tanh`. Note: dense MLP coexists with MoE branch. |
| `Gemma4TextRouter` | Fresh port (`Gemma4TextRouter_u`) | Has scale + per-expert scale unique to gemma4. |
| `Gemma4TextExperts` | NxDI `MoE` module from `moe_v2.py` (RouterTopK + ExpertMLPs) | Pattern matches Mixtral / Qwen3-MoE. State-dict converter concatenates `gate_proj+up_proj` if checkpoint uses split form (gemma4 source already stores `gate_up_proj` packed, so this mostly trivial). |
| `Gemma4TextAttention` | `NeuronAttentionBase` | Two attention classes: `NeuronGemma4SlidingAttention` (head_dim=256, sliding_window=1024, full RoPE) and `NeuronGemma4FullAttention_u` (head_dim=512, partial RoPE 0.25, K=V unified). The `_u` suffix flags the K=V variant since NxDI's base does not natively support K=V tying. |
| `Gemma4TextDecoderLayer` | Fresh `NeuronGemma4DecoderLayer` | Composes one of the two attention variants by layer_type, plus dense MLP + optional MoE branch + optional PLE residual + layer_scalar. |
| `Gemma4TextRotaryEmbedding` | `RotaryEmbedding` (NxDI) | Instantiated twice (one per layer_type). For partial RoPE on full layers, set `partial_rotary_factor=0.25`. |
| `Gemma4TextScaledWordEmbedding` | `ParallelEmbedding` | Apply `sqrt(hidden_size)` scaling either in the embedding wrapper or as a constant `embed_scale` buffer (matches HF). |
| Per-Layer-Embeddings | Fresh `Gemma4PLE_u` module | Maintains a second packed embedding `embed_tokens_per_layer` and a context projection. Output sliced per layer index. **Adds a second resident embedding ~256·30=7680-d × 262144 vocab — large; we will keep it in TP-replicated form in v0 then revisit sharding.** |
| `Gemma4ForCausalLM` | `NeuronBaseForCausalLM` pattern | Tied weights handled via `update_state_dict_for_tied_weights()`. |
| Vision / audio towers | **Not ported** in v0 | Customer goal is text inference. Mark as TODO. |

## 5. Parallelism strategy

Following the EP plan declared inside `Gemma4TextConfig.base_model_ep_plan` and the GenericMoE
v16 successful pattern:

- **TP degree**: target `TP=32` on a trn2.3xlarge (LNC=2 → 4 logical cores per device,
  but with 26B params we will need a multi-device setup → `trn2.48xlarge` (32 logical
  cores at LNC=2) is the realistic compile target). For *initial* compile validation a
  smaller config will work too: `TP=16` (genericmoe v16 reference). With 8 query heads
  any TP > 8 needs the `GQA.CONVERT_TO_MHA` warning (safe to ignore).
- **EP**: keep `EP=1` for v0. Token generation does not yet support EP>1 in NxDI's MoE
  module per the genericmoe port notes. Expert weights then shard along intermediate
  dim across TP ranks (what NxDI calls `moe_tp_experts`).
- **SP / CP**: off in v0. Sliding window already keeps activation memory in check.

Heads divisibility check at TP=32: `num_attention_heads=8` cannot be split across 32
ranks. So **realistic max TP for attention = 8** without head replication. We will
recommend **TP=8** as the first hardware compile target (single trn2.3xlarge, LNC=2 gives
4 logical cores per device — TP=8 spans both NeuronCores of a trn2 chip in LNC=1 mode
or 2 NeuronCore-V3 chips in LNC=2). The MLP/MoE intermediate dims (9216 and per-expert
intermediate) are still cleanly split at 8.

## 6. Known compile-time risks

1. **Two attention sub-classes per layer type**: NxDI's `attn_cls` is set once on the
   `NeuronConfig`. We override per-layer by storing both rotary embeddings at the model
   level and reading `layer_type` inside the attention forward. This is similar to how
   Qwen3-MoE handles per-layer differences, but the *head_dim asymmetry* (256 vs 512)
   between sliding and full layers means we cannot reuse one fused QKV projection.
   Implementation: separate Q/K/V projections allocated per-layer with the correct
   `head_dim`, and we expose `head_dim` as a per-layer attribute.
2. **K=V on full layers**: HF source sets `v_proj=None` and reuses `key_states` *before*
   `k_norm`/RoPE for value (then applies `v_norm` to the same tensor). NxDI's
   `NeuronAttentionBase` always allocates a `v_proj`. Workaround: allocate a `v_proj`
   anyway, but in `convert_hf_to_neuron_state_dict` **copy `k_proj.weight` into
   `v_proj.weight`** for full layers. This wastes some weights but stays inside the base
   class. Mark with `_u` suffix.
3. **PLE residual**: introduces a forward-time recompute path that NxDI's traced model
   may not handle gracefully. The PLE inputs depend on `input_ids`, not just `inputs_embeds`,
   so the inference compile path needs `input_ids` as a graph input (already true for
   NxDI). v0 keeps PLE on; if compile fails it can be ablated for an initial smoke test.
4. **Sliding-window length 1024 < seq_len 2048**: per `genericmodel` lessons, the
   `get_last_kv_window` helper assumes `actual_seq_len >= window_size`; short prompts
   trigger an out-of-bounds gather. NxDI 2.27+ already includes the padding fix, but we
   should pin the SDK requirement in the README and test with a 1-token prompt before
   shipping.
5. **MoE compile-time HLO verifier**: per genericmoe v16, MoE compilations need
   `--internal-hlo2tensorizer-options='--verify-hlo=false'` in `compiler_args`.
6. **Embedding scale `sqrt(hidden_size)`** must live in **bf16** for runtime correctness
   (HF source explicitly downcasts to match the HF behaviour, see comment in
   `Gemma4TextModel.__init__`).
7. **vocab_size 262 144** is large; `ColumnParallelLinear` lm_head with `gather_output`
   will produce a 262 144-wide tensor per token. Should be fine for sampling, but
   memory-hot.
8. **Layer-scalar buffer**: registered as `register_buffer("layer_scalar", torch.ones(1))`
   in HF. We register identically. Confirm checkpoint conversion preserves it.

## 7. State-dict mapping (HF -> Neuron)

HF state-dict keys (per layer `model.layers.{i}.`):

```
self_attn.q_proj.weight                 ->  q_proj.weight
self_attn.q_norm.weight                 ->  q_norm.weight
self_attn.k_proj.weight  (sliding only? present on non-shared layers) ->  k_proj.weight
self_attn.k_norm.weight                 ->  k_norm.weight
self_attn.v_proj.weight  (None for full-attn K=V layers) ->  duplicate from k_proj for full layers (v0 hack)
self_attn.v_norm.weight  (with_scale=False -> empty)     ->  drop
self_attn.o_proj.weight                 ->  o_proj.weight
mlp.gate_proj.weight                    ->  mlp.gate_proj.weight
mlp.up_proj.weight                      ->  mlp.up_proj.weight
mlp.down_proj.weight                    ->  mlp.down_proj.weight
input_layernorm.weight                  ->  input_layernorm.weight
post_attention_layernorm.weight         ->  post_attention_layernorm.weight
pre_feedforward_layernorm.weight        ->  pre_feedforward_layernorm.weight
post_feedforward_layernorm.weight       ->  post_feedforward_layernorm.weight
layer_scalar (buffer)                   ->  layer_scalar
# MoE-only (every 6th layer index per the layer pattern, also gated on enable_moe_block):
router.norm.weight (if any)             ->  router internal norm
router.proj.weight                      ->  router.proj.weight
router.scale                            ->  router.scale
router.per_expert_scale                 ->  router.per_expert_scale
experts.gate_up_proj  (E, 2I, H)        ->  moe.experts.gate_up_proj
experts.down_proj     (E, H, I)         ->  moe.experts.down_proj
post_feedforward_layernorm_1.weight     ->  post_feedforward_layernorm_1.weight
post_feedforward_layernorm_2.weight     ->  post_feedforward_layernorm_2.weight
pre_feedforward_layernorm_2.weight      ->  pre_feedforward_layernorm_2.weight
# PLE-only:
per_layer_input_gate.weight             ->  per_layer_input_gate.weight
per_layer_projection.weight             ->  per_layer_projection.weight
post_per_layer_input_norm.weight        ->  post_per_layer_input_norm.weight

# Top-level:
model.embed_tokens.weight               ->  embed_tokens.weight
model.embed_tokens_per_layer.weight     ->  embed_tokens_per_layer.weight
model.per_layer_model_projection.weight ->  per_layer_model_projection.weight
model.per_layer_projection_norm.weight  ->  per_layer_projection_norm.weight
model.norm.weight                       ->  norm.weight
lm_head.weight                          ->  lm_head.weight (tied -> auto-copy from embed_tokens)
```

## 8. First hardware-compile recommendation

```
batch_size      = 1
seq_len         = 4096      # >= sliding_window (1024) by 4x; safe
tp_degree       = 8         # max for 8 query heads without head replication
use_fp16        = True      # gives bf16 (NxDI naming quirk)
neuron_config   = MoENeuronConfig (with attn_cls = our gemma4 attn)
torch_dtype     = torch.bfloat16
compiler_args   = "--auto-cast=matmult --internal-hlo2tensorizer-options='--verify-hlo=false'"
LNC             = 2 (trn2)  # NEURON_LOGICAL_NC_CONFIG=2
```

First smoke test: 8-token prompt, 16 generated tokens. If the K=V hack proves brittle
for full-attn layers we will iterate by *actually* nulling `v_proj` and using a
`forward()` override (per `OVERRIDING_FORWARD_GUIDANCE.md`).
