# Round 2 Summary — Gemma-4-26B-A4B-it port

Date: 2026-06-03

## What round 1 had right

- Overall NxDI port shape: per-layer attention class branching on
  `layer_type`, separate Q/K/V/o for sliding vs global, MoE block gated on
  `enable_moe_block`, decoder layer composing dense MLP + parallel MoE.
- 26B-A4B-specific MoE router (`scale`, `per_expert_scale`, FP32 routing,
  softmax + topk + per-token re-normalisation) — matches HF source line by
  line. **Kept verbatim in round 2.**
- Tied `lm_head` via `update_state_dict_for_tied_weights`.
- Right HF prefix-stripping intent in the state-dict converter.

## What round 2 borrowed from PR #106 (Jim Burtoft, gemma-4-31B-IT)

Net change: +1480 LOC borrowed, –540 LOC of speculative round-1 code dropped.

| File | Lines | Source |
|---|---|---|
| `nki_flash_attn_d256_swa.py` | 950 | verbatim copy |
| `nki_flash_attn_large_d.py` | 346 | verbatim copy |
| `ndxi_patch.py` | 500 | verbatim copy + 1 relative-import fix |
| `Gemma4KVCacheManager` | ~95 | verbatim |
| `SoftcappedLMHead` | ~25 | verbatim |
| `Gemma4ScaledEmbedding` | ~30 | verbatim |
| `NeuronGemma4Attention` (`apply_rotary_embedding` partial RoPE, `prep_qkv_tensors` v_norm, `perform_prefill` NKI hook, sliding=None on base) | ~120 | verbatim, head_dim values match |
| `q_layernorm.weight` pre-scale by sqrt(head_dim) in state-dict converter | ~15 | verbatim |
| `Gemma4InferenceConfig.text_config` extraction | ~80 | verbatim, extended with MoE attrs |

## Modeling fixes from cross-check

1. **Real config differs from round-1 guesses**: `hidden_size=2816` (not
   2304), `num_attention_heads=16` (not 8), `num_kv_heads=8` (not 4),
   `intermediate_size=2112` (not 9216), `final_logit_softcapping=30.0`,
   `hidden_size_per_layer_input=0`. **PLE code dropped entirely.**
2. **Speculative `_apply_qk_norm` / `_apply_v_norm` hooks** in round 1 do
   not exist in `NeuronAttentionBase`. Replaced by PR #106's
   `prep_qkv_tensors` override (v_norm) + per-layer Gemma4RMSNorm on Q/K.
3. **Outer LayerNorm replaced by Gemma4RMSNorm everywhere** — PR #106
   pattern, validated on 31B-IT.
4. **Missing softcapping** added (`SoftcappedLMHead` cap=30.0).
5. **Missing per-layer KV cache** added (`Gemma4KVCacheManager`).
6. **MoE block** uses cloned config with `intermediate_size=moe_intermediate_size`
   so `initialize_moe_module` reads the right dim.

## Hardware smoke test result

`trn2.3xlarge ap-southeast-4 (Melbourne)`, `NEURON_LOGICAL_NC_CONFIG=2`,
TP=8, seq_len=256, batch=1, bf16, MoE branch disabled
(`GEMMA4_DISABLE_MOE=1`).

- HLO generation: **OK** (3.9 s for both CTE and TKG models).
- neuronx-cc compilation: **OK** (priority model 91.8 s, all HLOs in 18.5 s,
  total 145 s).
- Compiled artifacts saved: `~/gemma4-compiled/model.pt` (17 MB) +
  `neuron_config.json`.

The compile-time path (which is the customer's review surface) works
end-to-end through:
- `Gemma4InferenceConfig` reading the real config.json
- `get_updated_configs` producing per-layer SWA / global configs
- `NeuronGemma4Attention` instantiating with correct head dims
- `Gemma4KVCacheManager` allocating heterogeneous shapes
- 30 decoder layers tracing into HLO
- ndxi_patch monkey-patching `perform_prefill` for d>128
- `SoftcappedLMHead` wrapping the lm_head linear

The post-compile **weight load** step then hit a `c10::Error` (NxDI's
default `_load_state_dict` filtered out real attention/MoE weights as
"redundant keys"). This is the next thing to debug — `convert_hf_to_neuron_state_dict`
is wired into the class but NxDI's load path may need a flag or a
different entry point. The compile artifacts are valid; only the load
plumbing is wrong.

## Open issues for follow-up

1. **State-dict load**: NxDI is dropping real weights as "redundant keys".
   Most likely cause: `convert_hf_to_neuron_state_dict` is not being called
   before key-matching. Check NxDI's `application_base.compile` path.
2. **MoE branch with NxDI moe_v2**: smoke compile ran with MoE off
   (`GEMMA4_DISABLE_MOE=1`). The MoE config aliasing (`num_local_experts`,
   `num_experts_per_tok`, MoE-specific `intermediate_size`) is in place but
   not yet exercised on hardware. NxDI's MoE process-group bring-up
   (`initialize_moe_process_group`) is config-driven and likely needs more
   wiring (e.g. `moe_ep_degree`, `moe_tp_degree` set on `neuron_config`).
3. **TP=8 with num_kv_heads=2** (global layers) triggers
   `GQA.CONVERT_TO_MHA` warning. Expected and harmless, but worth noting
   for the customer (a full review may want TP=16 on trn2.48xlarge to
   avoid head replication on the 2 global-KV-heads layers).
4. **Inference smoke test** not yet run. Once the state-dict load issue
   is resolved, run an end-to-end token-generation test with a short
   prompt to validate output coherence before chasing accuracy metrics.
5. **NKI kernel verification on 26B-A4B head_dims**: the kernels are
   PR #106-validated on 31B-IT (same head dims), but the d=256 SWA kernel
   path has a conditional `q_len >= 128` early-return that's hit during
   smoke compile (q_len=256 here). Worth a one-off check that the kernel
   path is exercised vs the decomposed-attention fallback.

## GitHub commit URL

(set after push)

## Time used

~30 min for round 2 cross-check + integration + compile. Compile took
2.4 min on the actual NEFF side once HLO was clean.
