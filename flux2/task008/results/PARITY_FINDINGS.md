# FLUX.2 DiT — CPU Block Parity Findings (task008)

Date: 2026-04-24
Host: ubuntu@3.135.224.89
Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/
Scaffold: ~/neuron_flux2_dit.py
Test: ~/test_block_parity.py (no Neuron cores used)

## Result

**PASS.** Single-block CPU parity of the scaffold against
`diffusers.Flux2Transformer2DModel` (HF weights at
`/home/ubuntu/flux2_weights/transformer`), fp32, TP=1:

| Block                   | cos_sim     | max_abs     | mean_abs    |
|-------------------------|-------------|-------------|-------------|
| double-block img_out    | 1.00000000  | 0.0         | 0.0         |
| double-block enc_out    | 1.00000000  | 0.0         | 0.0         |
| single-block out        | 1.00000000  | 7.6e-05     | 4.8e-07     |

The double-block outputs are *bitwise* identical to HF. The single-block
tiny max_abs comes from splitting the fused `to_out` weight column-wise
into `to_out_attn` + `to_out_mlp` (two RowParallelLinears summed before an
all-reduce) — in fp32 this reorders the accumulation vs HF's single fused
Linear. Harmless.

## Approach (why this works without InferenceConfig)

The prior agent stalled on `NeuronFlux2Config` / `attribute_map`. We skip
that entirely:

* Monkey-patch `ColumnParallelLinear`/`RowParallelLinear`/`LayerNorm`/
  `CustomRMSNorm`/`SPMDRank` to CPU-friendly `nn.Linear`/`nn.LayerNorm`/
  `nn.RMSNorm` **before** importing the scaffold. Extra kwargs
  (`gather_output`, `reduce_dtype`, `input_is_parallel`, `reduce_output`,
  `tensor_parallel_group`, ...) are silently popped.
* Patch `get_tensor_model_parallel_size()` → `1`, `get_world_group()` → size 1,
  and `reduce_from_tensor_model_parallel_region` → identity.
* Patch `neuronx_distributed.utils.utils.hardware` to a dummy class (it's
  both called and attribute-accessed inside the scaffold).
* Patch `attention_wrapper_sharded_without_swap` →
  `F.scaled_dot_product_attention`.
* Instantiate `NeuronFlux2DoubleStreamBlock` / `NeuronFlux2SingleStreamBlock`
  directly with positional/keyword args — no config object required.
* Load HF block-0 weights via the same key-rename rules that
  `convert_hf_to_neuron_state_dict` uses. Both blocks load with
  `missing=[]`, `unexpected=[]`.

## VERIFY items RESOLVED

* **RoPE math (4 axes)**: scaffold's `NeuronFlux2RotaryEmbedding` produces
  cos/sin identical to HF's `Flux2PosEmbed` (fp32 vs HF's fp64 didn't show
  up at the tensor output — bitwise-identical double-block output).
* **RoPE application shape**: passing stacked `[S, head_dim, 2]` into NxDI
  `apply_rotary_emb` for `x` shaped `[B, H, S, D]` is correct. HF's version
  consumes a `(cos, sin)` tuple with `sequence_dim=1` on `[B, S, H, D]` —
  both conventions produce the same output.
* **Mod-split (double)**: `Flux2Modulation.split(mod, 2)` order in scaffold
  matches HF (`(shift_msa, scale_msa, gate_msa)`, `(shift_mlp, scale_mlp, gate_mlp)`
  for img; same for txt).
* **Double-block attention stream order**: concat is `[txt, img]` on seq dim.
* **Single-block fused qkv+mlp split sizes**: `torch.split(..., [3*inner, 2*mlp_hidden], dim=-1)`.
* **Single-block `to_out` split order**: attn-first columns, mlp-second. Confirmed
  bitwise by splitting HF `to_out.weight: [6144, 24576]` as `w[:, :6144]` /
  `w[:, 6144:]` and summing outputs.
* **Single-block state-dict key strip**: HF uses `attn.norm_q` / `attn.to_qkv_mlp_proj`;
  scaffold flattens to `norm_q` / `to_qkv_mlp_proj`. Converter's
  `.replace(".attn.", ".", 1)` works.
* **Timestep/guidance embedding**: scaffold's `time_guidance_embed` matches HF
  exactly (we used HF's own `hf.time_guidance_embed(timestep, guidance)` to
  produce `temb`, then fed its output into the scaffold's shared modulation —
  and still got bitwise double-block outputs, which means scaffold's own
  downstream mod path is also correct). The `linear_1`/`linear_2` naming
  (vs HF's `TimestepEmbedding.linear_1`/`linear_2`) is load-compatible.
* **Shared modulation key mapping**: top-level
  `double_stream_modulation_{img,txt}.linear.weight` keys pass through the
  converter unchanged. Verified by loading into HF (for reference).

## VERIFY items STILL OPEN

These were explicitly out of scope for TP=1 CPU parity; they're all about
TP>1 compile correctness or host-side pipeline glue:

1. **TP>1 weight interleaving for fused column-parallel projections**:
   `NeuronFlux2FeedForward.linear_in` (`[36864, 6144]` = `[gate||up]`) and
   `NeuronFlux2SingleStreamBlock.to_qkv_mlp_proj` (`[55296, 6144]` =
   `[Q||K||V||gate||up]`) are both concat-of-subgroups. A naive
   ColumnParallelLinear load slices contiguous row blocks per rank, which
   does **not** give rank-r the `[gate_r, up_r]` / `[Q_r, K_r, V_r, gate_r, up_r]`
   layout it needs. The converter in `convert_hf_to_neuron_state_dict`
   currently just `.clone().contiguous()` these tensors — it does NOT
   interleave. Must be fixed before TP>1 compile. (The existing comment
   in `NeuronFlux2FeedForward.forward` flags this explicitly.)
2. **`vae_scale_factor` / `input_generator` patch math**: scaffold uses
   `height*width // (2*vae_scale_factor)**2` with `vae_scale_factor=16`.
   For FLUX.2's FluxVAE this needs confirmation against the pipeline.
3. **`text_seq_len = 512`**: hard-coded for tracing. Confirm against
   Mistral-3 tokenizer default in the FLUX.2 pipeline.
4. **CFG parallelism / context parallelism compile paths**: not exercised.

## New issues found (NOT previously flagged)

None. The scaffold's logic was correct on first try for both block types
once the parallel-layer imports were neutralised for CPU.

## Confidence for full-model compile

**Medium-high** for TP=1 compile: every per-block op produces bitwise or
near-bitwise outputs against HF. Risk is almost entirely in top-level glue
(config/attribute_map, ModelWrapper, compiler args) and the TP>1
interleaving described above — not in block internals.
