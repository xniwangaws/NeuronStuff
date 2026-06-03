# Gemma-4-26B-A4B-it - NeuronX Distributed Inference port

NxDI port of `google/gemma-4-26B-A4B-it` for AWS Trainium 2.

**Lineage**: round 2. Architecture and weight conversion borrow heavily from
Jim Burtoft's PR #106 (`gemma-4-31B-IT`); MoE block + router are
26B-A4B-specific (the 31B model is dense). NKI flash-attention kernels
copied verbatim from PR #106 — head dimensions match (256 sliding /
512 global) so they apply unchanged.

## Files

| File | Role |
|---|---|
| `modeling_gemma4_neuron.py` | NxDI implementation: attention, dense MLP, MoE block + router, decoder layer, KV cache manager, model, causal-LM head, HF→Neuron state-dict converter. |
| `configuration_gemma4_neuron.py` | Lightweight HF config shim for static parsing. The real config classes live in `modeling_gemma4_neuron.py`. |
| `nki_flash_attn_d256_swa.py` | NKI sliding-window flash attention kernel for `head_dim=256` (used on SWA layers). Verbatim from PR #106. |
| `nki_flash_attn_large_d.py` | NKI flash attention kernel for `head_dim > 128` (used on global `head_dim=512` layers). Verbatim from PR #106. |
| `ndxi_patch.py` | Runtime monkey-patches: `get_last_kv_window` LongTensor fix, NKI kernel hooks for `head_dim > 128`, multimodal forward bypass. Verbatim from PR #106. |
| `__init__.py` | Package init. |
| `README.md` | This file. |

## What was reused from existing NxDI

- `NeuronAttentionBase` — Q/K/V/o projections, KV cache, GQA sharding, mask
  builders. We override `apply_rotary_embedding` (partial RoPE), `prep_qkv_tensors`
  (post-projection v_norm), and `perform_prefill` (NKI d=256 SWA kernel).
- `RotaryEmbedding` — instantiated per-layer with the right `dim` for partial
  RoPE on global layers.
- `ColumnParallelLinear` / `RowParallelLinear` / `ParallelEmbedding` — for dense
  MLP, lm_head, token embedding.
- `initialize_moe_module` (NxDI MoE v2) — handles expert dispatch and sharded
  `gate_up_proj` / `down_proj`. We feed it our own `top_k_index` /
  `top_k_weights` from the gemma4 router.
- `KVCacheManager` — subclassed to support per-layer heterogeneous shapes
  (8×256 SWA vs 2×512 global, after TP sharding).
- `NeuronBaseForCausalLM` / `NeuronBaseModel` — generation loop, sampling,
  weight loading.

## What was ported fresh (and why)

| Class | Reason |
|---|---|
| `Gemma4RMSNorm` | gemma4 RMSNorm with weight (init to 1), not the `(1 + weight)` style of earlier Gemma. |
| `Gemma4VNorm` | gemma4 v_norm (RMSNorm with `with_scale=False`). |
| `Gemma4ScaledEmbedding` | Token embedding scaled by `sqrt(hidden_size)` (gemma4 source line 1459). |
| `SoftcappedLMHead` | Wraps lm_head and applies `cap * tanh(x / cap)` with `cap=30.0` in fp32. |
| `Gemma4KVCacheManager` | Per-layer KV cache shapes for the heterogeneous SWA/global head dims. |
| `NeuronGemma4Attention` | Per-layer head_dim/kv_heads, partial RoPE for global, K=V handled at weight level, NKI d=256 SWA prefill. |
| `NeuronGemma4Router` | gemma4 router with `scale` and `per_expert_scale` learned tensors, FP32 routing, soft+top-k+normalise+per-expert-scale. **26B-A4B-specific**: PR #106 has no router. |
| `NeuronGemma4DecoderLayer` | Dense MLP + parallel MoE branch (router reads pre-MLP residual; outputs combined as `mlp_branch + moe_branch`), then `layer_scalar` multiply. |
| `convert_hf_to_neuron_state_dict` | Prefix stripping, `embed_tokens` → `embed_tokens.embedding` rename, q/k_norm → q/k_layernorm rename, q_layernorm.weight pre-scaling by `sqrt(head_dim)` (cancels NxDI's automatic 1/sqrt-scale), `attention_k_eq_v` weight copy, tied lm_head, `rank_util` tensors. |

## Round 2 corrections vs round 1

The HF model was gated; round 1 inferred shapes from documentation. The real
`config.json` differs significantly:

| Field | Round 1 guess | Real | Note |
|---|---|---|---|
| `hidden_size` | 2304 | **2816** | – |
| `num_attention_heads` | 8 | **16** | TP plan changes |
| `num_key_value_heads` | 4 | **8** | KV cache shape changes |
| `intermediate_size` | 9216 | **2112** | dense MLP much smaller |
| `final_logit_softcapping` | none | **30.0** | softcap on output logits |
| `hidden_size_per_layer_input` | 256 | **0** | **no PLE on 26B-A4B** |

Round 1 also lacked: NKI flash kernels, KV cache manager, softcapping,
QK-scaling-via-norm-weight trick, attention partial RoPE override, vnorm
in `prep_qkv_tensors`. All of those came over from PR #106 in round 2.

## Tensor-parallel / sharding strategy

- Dense MLP: column-parallel `gate_proj` + `up_proj`, row-parallel `down_proj`.
- MoE experts: TP-sharded along intermediate dim (NxDI moe_v2 default).
- Attention: per-layer Q/K/V/o (different head_dim per layer-type, so no
  fused QKV); Q/K/V column-parallel along head dim; O row-parallel.
- Router: replicated across TP (identical routing on every rank).
- LM head: column-parallel with gather (or on-device sampling).

**Recommended first compile**: TP=8, seq_len=256, batch_size=1, LNC=2,
bfloat16. On trn2.3xlarge with `NEURON_LOGICAL_NC_CONFIG=2` you get 4
logical cores; for TP=8 use `LNC=1` (8 logical cores) or move to
trn2.48xlarge.

## How to compile (on trn2)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd ~/gemma4-port

# Apply ndxi_patch + run smoke compile (256 seq_len for first pass)
python scripts/smoke_compile.py 2>&1 | tee ~/compile.log
```

For longer compiles, set `GEMMA4_SEQ_LEN=4096` and `GEMMA4_TP_DEGREE=8`.

## Known limitations / TODOs

- **Multimodal towers** (vision, audio) are **not ported**. Text-only.
- **K=V tying** done at state-dict level (k_proj cloned into v_proj for
  global layers). Wastes ~2 % HBM on global layers only.
- **MoE compile time**: 30 layers × 128 experts × bf16 — expect 30-60 min for
  the first compile. MoE compilations require
  `--internal-hlo2tensorizer-options='--verify-hlo=false'` per the genericmoe
  v16 KB.
- **NxDI ≥ 0.10** required (for `get_last_kv_window` patch and per-layer
  `layer_to_cache_size_mapping`).
- **Apply `ndxi_patch.apply_patch()` once at process start** before
  constructing the model class — see `scripts/smoke_compile.py`.
