# Gemma-4-26B-A4B-it - NeuronX Distributed Inference port (dry-run)

This directory contains a **dry-run** port of `google/gemma-4-26B-A4B-it` to NxDI for AWS
Trainium. No code in this directory has run on hardware. It is structured so a follow-up
session on a trn2 instance can compile + smoke-test it without rewriting anything.

## Files

| File | Role |
|---|---|
| `configuration_gemma4_neuron.py` | Adapter classes that wrap HF `Gemma4TextConfig` for NxDI's `InferenceConfig` and `MoENeuronConfig`. Factories defer NxDI imports so the file is laptop-importable. |
| `modeling_gemma4_neuron.py` | Full NxDI implementation: attention, dense MLP, MoE block + router, decoder layer, model, causal-LM head, and HF→Neuron state-dict converter. |
| `__init__.py` | Lazy package init (avoids eager-importing NxDI on a laptop). |
| `README.md` | This file. |

## What was reused from existing NxDI

- **`NeuronAttentionBase`** — Q/K/V/o projections via `ColumnParallelLinear` /
  `RowParallelLinear`, KV-cache management, flash-attention kernel selection, GQA
  sharding, sliding-window mask. We override `_apply_qk_norm` / `_apply_v_norm` so
  Gemma4's per-head RMSNorm + V centring lands in the right place.
- **`RotaryEmbedding`** — instantiated twice, once per layer_type. Partial RoPE on full-
  attention layers is supported via a smaller rotated dim (computed from
  `partial_rotary_factor`).
- **`ColumnParallelLinear` / `RowParallelLinear` / `ParallelEmbedding`** — used for the
  dense MLP, lm_head, and token embedding.
- **`MoE` v2 (`initialize_moe_module`)** — handles expert dispatch, sharded
  `gate_up_proj` / `down_proj`, and (potential) all-to-all routing. We feed it
  pre-computed `top_k_index` / `top_k_weights` from our own router so the gemma4-specific
  `scale` / `per_expert_scale` parameters keep working.
- **`CustomRMSNorm`** — used for the *inner* norms (q_norm, k_norm, router norm). NxDI's
  Neuron-optimised RMSNorm is correct for these.
- **`NeuronBaseForCausalLM` / `NeuronBaseModel`** — the generation loop, sampling, weight
  loading, and `setup_attr_for_model` plumbing.

## What had to be ported fresh (and why)

| New class / function | Why no 1:1 reuse |
|---|---|
| `NeuronGemma4Router_u` | Gemma4 router has two extra learnable tensors (`scale` per-hidden-dim and `per_expert_scale` per-expert) plus an internal RMSNorm-without-scale. NxDI's `RouterTopK` does not carry these. |
| `NeuronGemma4MoEBlock_u` | Wraps the gemma4 router + NxDI experts so the *input to routing* can be the pre-MLP residual (gemma4-specific) rather than the MLP input. |
| `Gemma4PLE_u` | Per-Layer-Embeddings pipeline (extra packed embedding + projection + per-layer slicing) has no NxDI equivalent. |
| `_RMSNormNoScale_u` | Centring-only RMSNorm (gemma4's `with_scale=False`). |
| `NeuronGemma4Attention_u` | Two-mode head sizing (sliding 256-d / full 512-d), partial-RoPE on full layers only, K=V tying on full layers, Q/K/V per-head RMSNorm. The base class doesn't ship this combo, so we subclass and override the QK-norm / V-norm hooks. |
| `NeuronGemma4DecoderLayer` | Composes dense MLP + MoE branch + PLE residual + `layer_scalar` — gemma4-specific layout with norms in unusual positions (post-attention norm before residual; mlp/moe parallel branches). |
| `convert_hf_to_neuron_state_dict` | Strip `model.` prefix, expand K-into-V on full-attention layers (because `v_proj` is `None` in HF for K=V layers), tie lm_head. |

## Tensor-parallel / sharding strategy

- Dense MLP: column-parallel `gate_proj` + `up_proj`, row-parallel `down_proj`. Standard
  SwiGLU sharding (matches Llama / Mistral pattern).
- MoE experts: `moe_tp_experts` plan — expert intermediate dim sharded across TP ranks,
  experts replicated. Expert-parallelism (EP) is **not** enabled in v0 (token
  generation does not yet support EP > 1 in NxDI; see the genericmoe port notes).
- Attention: separate Q/K/V for each layer (different head_dim per layer-type), so no
  fused QKV. Q/K/V column-parallel along head dim; O row-parallel.
- Router projection: kept replicated across TP (every rank must agree on routing).
- LM head: column-parallel with gather (or sampled on-device when configured).

## TP recommendation

| Constraint | Implication |
|---|---|
| `num_attention_heads = 8` | Max TP across attention without head replication is 8. Above 8 you get the safe-to-ignore `GQA.CONVERT_TO_MHA` warning, but it costs memory. |
| `intermediate_size = 9216` | Cleanly divisible by 1, 2, 4, 6, 8, 9, 12, 16, 18, ... Most TP values are fine. |
| `num_experts = 128`, `top_k = 8` | Plenty of routing diversity even at TP=8; experts naturally shard along the intermediate dim. |
| 26B params @ bf16 = 52 GB | Needs at least 4 NeuronCores worth of HBM (16 GB each → 64 GB usable). Recommend TP=8 minimum. |

**Recommended first compile**: `TP=8`, `seq_len=4096`, `batch_size=1`, `LNC=2`,
`use_fp16=True` (bfloat16). On a trn2.3xlarge this maps to one chip in LNC=1 mode or
both NeuronCores of the chip in LNC=2; on trn2.48xlarge you have headroom for TP=16 or 32.

## Known limitations / TODOs

- **Multimodal towers** (vision, audio) are **not ported**. Customer's first goal is
  text inference; the vision/audio encoders + multimodal embedder live in HF source
  but we omit them in v0.
- **K=V tying** is implemented at the **state-dict level** by copying `k_proj` weights
  into `v_proj`. This wastes a small amount of HBM (~2 % of weights for full-attn layers
  only) but stays inside the NxDI base class. If memory becomes tight a forward override
  is the next step.
- **PLE v0 keeps `embed_tokens_per_layer` replicated**. At 262 144 × 30 × 256 ≈ 2 GB in
  bf16 this is significant; a future version should shard along the packed dim.
- **Compiled with `--verify-hlo=false`**. MoE compilations need this flag (per the
  genericmoe v16 success summary) — make sure the compile script passes it via
  `compiler_args`.
- **NxDI 2.27+** required for the `get_last_kv_window` padding fix that makes
  sliding-window attention safe with prompts shorter than `sliding_window`.

## How to compile (when you have hardware)

```bash
# On a trn2 instance with SDK 2.29 DLAMI
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# (a) Make this directory importable
export PYTHONPATH="/path/to/gemma4-port-26b-a4b:$PYTHONPATH"

# (b) Use the validation config template at
#     agent_artifacts/tmp/validation_config.json (paths are pre-filled)

# (c) Compile script — follow the pattern in
#     skills/.../assets/example_phimoe_usage.py, with:
#       model_class       = neuron_port.modeling_gemma4_neuron:NeuronGemma4ForCausalLM
#       config_class      = neuron_port.modeling_gemma4_neuron:Gemma4InferenceConfig
#       neuron_config_class = neuron_port.modeling_gemma4_neuron:Gemma4NeuronConfig
#       use_fp16          = True
#       tp_degree         = 8
#       seq_len           = 4096
#       batch_size        = 1
#       compiler_args     = "--auto-cast=matmult \
#                            --internal-hlo2tensorizer-options='--verify-hlo=false'"

rm -rf /var/tmp/neuron-compile-cache  # safety
```

## How to validate

After compilation, point the validation tool at
`agent_artifacts/tmp/validation_config.json` (already templated):

```bash
python scripts/validate_model.py \
    --config agent_artifacts/tmp/validation_config.json \
    --mode token \
    --batch-size 1 \
    --seq-len 4096 \
    2>&1 | tee agent_artifacts/tmp/validation.log
```

Pass criterion: `>= 95 %` greedy token match against the HF golden reference.
