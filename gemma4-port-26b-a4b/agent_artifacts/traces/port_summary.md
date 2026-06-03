# Gemma-4-26B-A4B-it - Port Summary (Dry-Run)

**Date**: 2026-06-03
**Customer**: Sheldon (via Trainium GA team)
**Source**: `google/gemma-4-26B-A4B-it` (HuggingFace)
**Target**: NeuronX Distributed Inference (NxDI) on AWS Trainium 2
**Status**: Code-only dry-run. **No hardware compile, no inference, no validation.**
**Code location**: `/Users/xniwang/NeuronStuff/gemma4-port-26b-a4b/neuron_port/`

## Model

| Property | Value |
|---|---|
| Architecture | Gemma 4 (text decoder + MoE) |
| Total params | 25.2 B |
| Active params | 3.8 B |
| Layers | 30 |
| Hidden size | 2304 |
| Attention heads | 8 |
| KV heads (sliding) | 4 |
| KV heads (full) | 2 |
| Head dim (sliding / full) | 256 / 512 |
| Sliding window | 1024 (5:1 sliding/full layer pattern) |
| Experts (active / total) | 8 / 128 (+ 1 dense MLP that runs in parallel = "shared expert") |
| Vocab | 262 144 |
| Max position | 131 072 (extendable to 256K) |
| dtype | bfloat16 |
| Tied embeddings | yes |

## Architecture decisions

1. **Two attention sub-shapes per model**: Sliding layers and full-attention layers
   have different `head_dim` (256 vs 512) and different `num_kv_heads` (4 vs 2).
   We allocate separate Q/K/V/o linear layers per layer, but reuse one
   `NeuronGemma4Attention_u` class that branches on `layer_type`.
2. **Partial RoPE on full-attn layers**: 25 % of `head_dim` rotated. NxDI's
   `RotaryEmbedding(dim=head_dim*0.25, ...)` handles this directly.
3. **K=V on full-attn layers** (`attention_k_eq_v=True`): Implemented via
   state-dict copy (k_proj weights → v_proj weights) rather than a forward
   override. Wastes ~2 % HBM but stays inside `NeuronAttentionBase`.
4. **Outer LayerNorm, inner RMSNorm**: Per the genericmoe v16 KB lesson —
   input/post-attention/pre-MLP/post-MLP/final norms are `nn.LayerNorm`
   (prevents activation drift across deep layers on Neuron); Q/K/V/router
   internal norms are `CustomRMSNorm` (faster + numerically correct for those
   inner positions).
5. **MoE**: NxDI's `MoE` v2 + a fresh gemma4-specific router (`scale` and
   `per_expert_scale` learned tensors not present in NxDI's `RouterTopK`).
   Router compute is **FP32** with softmax (KB-driven).
6. **PLE (Per-Layer-Embeddings)** ported fresh as `Gemma4PLE_u`. Replicated
   across TP ranks in v0 (≈ 2 GB extra HBM per rank); shard later if needed.
7. **Multimodal towers omitted**. Customer's primary goal is text inference.
   Vision + audio + multimodal-embedder are stubs to be added in v1.
8. **Tied lm_head**: handled via `update_state_dict_for_tied_weights`
   (genericmodel KB lesson #1).
9. **TP recommendation**: TP=8 first (limited by 8 query heads). Larger TP works
   with `GQA.CONVERT_TO_MHA` warning but costs HBM. EP=1 in v0.

## Components reused from existing NxDI

| Component | NxDI / NxD class |
|---|---|
| Attention base | `neuronx_distributed_inference.modules.attention.attention_base.NeuronAttentionBase` |
| RoPE | `neuronx_distributed_inference.modules.attention.utils.RotaryEmbedding` |
| RMSNorm (inner) | `neuronx_distributed_inference.modules.custom_calls.CustomRMSNorm` |
| Parallel layers | `neuronx_distributed.parallel_layers.layers.{ColumnParallelLinear, RowParallelLinear, ParallelEmbedding}` |
| MoE | `neuronx_distributed_inference.modules.moe_v2.initialize_moe_module` |
| Causal-LM base | `neuronx_distributed_inference.models.model_base.{NeuronBaseModel, NeuronBaseForCausalLM}` |
| MoE-aware NeuronConfig | `neuronx_distributed_inference.models.config.MoENeuronConfig` |
| Inference config | `neuronx_distributed_inference.models.config.InferenceConfig` |

## Components ported fresh (with `_u` suffix where they have no 1:1 HF map)

| Class | Reason |
|---|---|
| `NeuronGemma4Attention_u` | Two head_dims, partial RoPE, K=V tying — combo not in NxDI base. |
| `NeuronGemma4Router_u` | Extra `scale` + `per_expert_scale` learned tensors. |
| `NeuronGemma4MoEBlock_u` | Routes from pre-MLP residual (gemma4-specific input). |
| `Gemma4PLE_u` | Per-Layer-Embedding pipeline — no NxDI equivalent. |
| `_RMSNormNoScale_u` | Centring-only RMSNorm (gemma4 v_norm + router input norm). |
| `NeuronGemma4MLP` | Standard SwiGLU but with gemma4's `gelu_pytorch_tanh` activation; clean reuse possible (`mlp` name preserved). |
| `NeuronGemma4DecoderLayer` | Composes dense MLP + parallel MoE branch + PLE residual + `layer_scalar` — gemma4-specific layout. |
| `NeuronGemma4Model` | Standard pattern; uses NxDI base. Name preserved (1:1 with `Gemma4TextModel`). |
| `NeuronGemma4ForCausalLM` | 1:1 with `Gemma4ForCausalLM`. |
| `convert_hf_to_neuron_state_dict` | K=V duplication for full-attention layers; HF prefix stripping. |

## Open questions / known risks for first hardware compile

1. **K=V state-dict trick**: replicating k_proj weights into v_proj works at the
   parameter level, but a downstream NxDI assertion that "v_proj initialised from
   v_proj.weight shape" might still fire if it inspects the original HF checkpoint
   structure. Fallback: forward override per `OVERRIDING_FORWARD_GUIDANCE.md`.
2. **Per-layer attention head_dim**: NxDI's `NeuronAttentionBase` is designed for
   homogeneous `head_dim` across all layers. We pass `head_dim` per-instance, but the
   base class might cache compile-time constants assuming uniformity. If this surfaces,
   we'll split into two attention classes (`NeuronGemma4SlidingAttention` and
   `NeuronGemma4FullAttention_u`) and select via `layer_idx` in `NeuronGemma4DecoderLayer`.
3. **PLE in traced graph**: PLE depends on `input_ids` (not just `inputs_embeds`). NxDI
   does pass `input_ids` through compile, but the PLE forward computes a separate
   embedding table lookup that may need explicit registration as a sharded module. If
   the trace fails: ablate PLE in v0 (set `hidden_size_per_layer_input=0` in the
   inference config), validate the rest of the model, then re-enable PLE in v1.
4. **MoE compile time**: 30 layers × 128 experts × bf16 = significant HLO. Expect 30-60
   min compile (genericmoe v16 reference: 14.7 min for 32 layers × 16 experts at TP=16).
   Allocate at least 60 min budget on first hardware run.
5. **vocab_size 262 144 lm_head**: 262 144 × 2304 × bf16 = 1.2 GB. Tied with
   embed_tokens, so total is just 1.2 GB once. Fine, but compile might do a non-tied
   trace if `tie_word_embeddings` isn't propagated correctly into the framework's
   assumption — pin via `add_derived_config()` (already done in our wrapper).
6. **Sliding-window padding fix**: NxDI 2.27+ required. If user is on an older SDK,
   the patch from `PORTING_SLIDING_WINDOW.md` (lines 280-298) needs to be applied to
   `attention/utils.py:get_last_kv_window`.
7. **`layer_scalar` checkpoint key**: HF stores it as a buffer. Need to verify
   `convert_hf_to_neuron_state_dict` doesn't strip it.
8. **Number of layers % LNC compatibility**: 30 layers does not divide cleanly into
   LNC=2's 4 logical cores or LNC=1's 8 logical cores. NxDI handles uneven layer
   placement, but watch for memory imbalance warnings.

## Next steps when hardware is provisioned

1. **Step 1 — Compile smoke test**: Allocate trn2.3xlarge in sa-east-1 (or trn2.48xlarge
   for headroom), activate `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`, set
   `NEURON_LOGICAL_NC_CONFIG=2`. Compile with:
   ```
   batch_size = 1
   seq_len    = 4096
   tp_degree  = 8
   use_fp16   = True
   compiler_args = "--auto-cast=matmult \
                    --internal-hlo2tensorizer-options='--verify-hlo=false'"
   ```
   Expected: 30-60 min. Watch for OOM; bump to TP=16 if needed.

2. **Step 2 — Inference smoke test**: 1-token, 8-token, and 100-token prompts. Verify
   output is coherent (catches the LayerNorm-vs-RMSNorm gibberish failure mode early).

3. **Step 3 — Validation**: Use `agent_artifacts/tmp/validation_config.json`, target
   ≥ 95 % greedy token match against the HF golden reference. If < 95 %, iterate per
   the genericmoe playbook: check norm types, router precision, K=V state-dict copy.

4. **Step 4 — Multimodal**: Add vision tower (`Gemma4VisionModel` →
   `NeuronGemma4VisionModel`) and the multimodal embedder. Likely a separate compile
   per the MLLama pattern.

5. **Step 5 — PR back to customer**: Push final compiled artifacts + measurements to
   `https://github.com/xniwangaws/NeuronStuff` and ping Sheldon.
