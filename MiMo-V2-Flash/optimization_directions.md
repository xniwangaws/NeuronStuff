# Optimization Directions for MiMo-V2-Flash on Trn2

Ranked by evidence strength and expected impact. Each entry lists: **evidence** (from profile data, not speculation), **hypothesis**, **expected gain**, **cost to validate**.

---

## Tier 1 — Strong profile evidence

### 1. Close the 100 ms host-side gap at c>1

- **Evidence**: BF16 BS=32 device time = 34.7 ms/token; measured vLLM TPOT at c=32 = 135 ms. 100 ms of per-token time is host-side.
- **Hypothesis**: Python MoE dispatch + vLLM V1 scheduler + collective setup together cost 100 ms per decode step at c=32, but only ~0 ms at c=1 (because c=1 uses fast-path scheduling and static routing).
- **Expected gain**: 50–60% throughput at c=32 if we can halve the host gap (135 ms → 85 ms would match PR).
- **Cost to validate**: 1–2 h with `py-spy record` while serving at c=32. No new capacity block needed if we can re-use the existing compiled NEFFs from S3.
- **Action**: Run `py-spy record --pid <vllm-engine-pid> --duration 60` during a bench, inspect the flame graph, identify top 3 host functions.

### 2. Switch the MoE blockwise kernel flag to `shard_on_intermediate` (FP8 standalone verified; vLLM unverified)

- **Scope where verified**: **FP8 standalone NxDI only** (`adapter.generate()` path). Not tested on:
  - **BF16 path** — uses `use_torch_block_wise=true` (Python fallback), so this flag does not apply.
  - **FP8 vLLM serving path** — same kernel underneath, but vLLM adds ~100 ms/token host overhead at c=32 which may dominate the 6 ms kernel gain. Needs empirical verification.
- **Evidence (FP8 standalone, MOE_EP=64, BS=32)**:
  - `use_shard_on_block_dynamic_while=true`: 137.87 tok/s
  - `use_shard_on_intermediate_dynamic_while=true`: **146.29 tok/s (+6%)**
- **Hypothesis**: `shard_on_intermediate` has a lower-overhead MoE kernel path on SDK 2.29. Not verified mechanistically.
- **Expected gain**:
  - FP8 standalone: +6% (proven)
  - FP8 vLLM: unknown — may be dominated by host overhead
  - BF16: not applicable
- **Cost to validate on vLLM**: One line change in `COMMON_MIMO_CONFIG` in `bench_mimo_v2_flash.sh`, recompile (~30 min), rerun the three-concurrency bench (~15 min). Can confirm or rule out the hypothesis.
- **Action**: (1) Update `bench_mimo_v2_flash.sh` (FP8 recipe) to use `use_shard_on_intermediate_dynamic_while: true`. (2) Re-bench c=1/16/32, compare to current 147 tok/s (c=32).

### 3. Flag ablation: apply BF16-recipe flags to the FP8 recipe

- **Scope**: FP8 serving path only (Neuron-FP8 checkpoint, `use_shard_on_intermediate_dynamic_while=true` baseline).
- **Observation**: PR ships two distinct config recipes. The old BF16 recipe (`bench_mimo_v2_flash.sh` at commit `305279f^`) enables several flags that the current FP8 recipe (`COMMON_MIMO_CONFIG` at HEAD) drops. The FP8 recipe is likely a **minimal** port — these flags were not explicitly removed because they broke FP8, they were just not carried forward.

| Flag | BF16 old recipe | FP8 current recipe | Tested by us on FP8? |
|---|:-:|:-:|:-:|
| `use_shard_on_intermediate_dynamic_while` | true | **not set** | ✅ (+6% standalone) |
| `skip_dma_token` | true | **not set** | ❌ |
| `use_index_calc_kernel` | true | **not set** | ❌ |
| `moe_mask_padded_tokens` | true | **not set** | ❌ |
| `disable_numeric_cc_token` | true | **not set** | ❌ |
| `scratchpad_page_size: 1024` | true | **not set** | ❌ |
| `sequence_parallel_enabled` | true | false | ❌ |
| `flash_decoding_enabled` | false (explicit) | not set | ❌ |
| `attn_kernel_enabled` | false (explicit) | not set | ❌ |

- **Hypothesis**: Some of these flags independently improve FP8 MoE TGEN / CTE throughput. They were engineered for large-batch CB serving and are not inherently tied to BF16.
- **Two possible reasons PR dropped them on FP8**:
  1. They break FP8 numerics or cause compile failures (tested but rejected).
  2. Not tested on FP8 — the FP8 recipe was written fresh.
  
  **We don't know which. Need to ask PR author (Henan) or test empirically.**

- **Proposed experiment (flag-by-flag ablation)**:
  ```
  Baseline:            PR current FP8 recipe               → ~147 tok/s (c=32, we measured)
  + shard_on_intermediate                                   → ~156 tok/s (extrapolated from standalone +6%)
  + skip_dma_token                                          → ?
  + use_index_calc_kernel                                   → ?
  + moe_mask_padded_tokens                                  → ?
  + disable_numeric_cc_token                                → ?
  + scratchpad_page_size: 1024                              → ?
  + sequence_parallel_enabled                               → ?
  + flash_decoding_enabled=false (already default? verify)  → ?
  ```

- **For each flag flip**:
  1. Enable the flag (one at a time, cumulative on previous wins)
  2. Recompile (~30-40 min)
  3. Greedy-decode the same deterministic prompt, compare output token IDs to baseline (**correctness gate** — any divergence means the flag broke numerics)
  4. If output matches: run `vllm bench serve` c=1/16/32 (~15 min), record tok/s
  5. If output diverges: flag disabled, move on

- **Expected aggregate gain**: hard to predict without data. If all ~7 flags independently help by 5-10%, compound could be 30-50%. If most are no-ops, the win is the few that matter.

- **Cost**: ~35-40 min per flag × 7 flags = ~4.5 h total. Fits in one 7-hour capacity block (~$260).

- **Critical**: sanity-check greedy output tokens at **every** step. This session did NOT do that for Exp 1 vs Exp 2, so the 6% gain there could theoretically be "faster because outputs diverged into a shorter path". To be rigorous, add the check.

- **Action**: Next capacity block. First hour: sync checkpoints + install + set up continuous S3 sync daemons (see `sop_capacity_block_survival.md`). Then ablation loop with correctness gate.

---

### 4. Re-enable `qkv_nki_kernel` and `flash_decoding`

- **Evidence**: PR README lists six NKI kernels explicitly disabled (`qkv_kernel_enabled`, `qkv_nki_kernel_enabled`, `qkv_cte_nki_kernel_fuse_rope`, `attn_kernel_enabled`, `strided_context_parallel_kernel_enabled`, `flash_decoding_enabled`). Disabling is a compatibility workaround for Q/K=192 / V=128 asymmetry.
- **Hypothesis**: Some of these six kernels actually tolerate the asymmetry and were disabled defensively. Each kernel typically gives 10–30% on its op.
- **Expected gain**: Unknown per kernel; cumulative 20–40% if 2–3 of them work.
- **Cost to validate**: One capacity block, 6 compiles (~40 min each) + 6 benches (~5 min each) = ~4.5 h of instance time (~$120 on a 7h block).
- **Action**: Flip one flag at a time, verify accuracy (generate a known prompt, cosine-sim against BF16 baseline), measure throughput.

---

## Tier 2 — Compiler-side (depends on AWS Neuron team)

### 5. Wait for `blockwise_mm_baseline_shard_hidden` NKI kernel in SDK 2.30

- **Evidence**: PR source comment says this kernel is the designed default for MoE blockwise but is missing from SDK 2.29's public surface. `_call_shard_hidden_kernel` is the default code path but fails to link.
- **Hypothesis**: When the kernel ships, MoE FP8 path can hit compiler-predicted mfu_max = 100% (vs current 38%).
- **Expected gain**: PR author's own estimate is "FP8 ≈ 2× BF16 path". Current: FP8 is ~same as BF16. Potential 2× uplift.
- **Cost to validate**: Zero — just wait for Neuron release, re-run same bench.
- **Action**: Watch `neuronxcc` release notes. Re-bench when the symbol appears in `neuronxcc.nki._private.blockwise_mm`.

### 6. Close the CTE mfu gap (38% actual vs 100% achievable)

- **Evidence**: `neuron-profile view` on FP8 BS=32 CTE NEFF reports `mfu_hlo_max_achievable_estimated_percent = 100%`, actual `mfu_estimated_percent = 38%`.
- **Hypothesis**: Intra-NEFF pipeline / dependency gaps. Not a config issue — a compiler scheduling issue.
- **Expected gain**: Up to 2.6× on CTE. CTE is only ~10% of TPOT at BS=32, so end-to-end gain ~10–15%.
- **Cost to validate**: File compiler ticket with NEFF hash + profile data. Out of our hands.
- **Action**: Include as evidence in a ticket to AWS Neuron.

---

## Tier 3 — Architecture / config sweeps

### 7. Try `enable_prefix_caching`

- **Evidence**: PR disables it. vLLM prefix cache amortizes CTE for repeated prefixes.
- **Hypothesis**: If workload has repeated system prompts, TTFT drops 3–5×.
- **Expected gain**: Workload-dependent; 0% if prompts are random, 50%+ if there's a shared system prompt.
- **Cost to validate**: One flag, recompile, rerun bench. ~40 min.
- **Action**: Test with a realistic chat-style dataset, not the `random` dataset PR uses.

### 8. Reduce `max_model_len` from 1024 to 512

- **Evidence**: PR uses 1024; KV cache scales linearly with seq_len.
- **Hypothesis**: Smaller KV cache → more active batch slots → more throughput (not latency).
- **Expected gain**: Up to 30% throughput if KV-cache-limited.
- **Cost to validate**: One flag, recompile, rerun.
- **Action**: Only if the target workload tolerates shorter context.

### 9. BS=128 throughput config

- **Evidence**: PR's `bench_mimo_v2_flash.sh` includes `bs128_tp1_ep64_opt` config we skipped.
- **Hypothesis**: BS=128 amortizes per-token overhead more; PR claims throughput grows ~3× from BS=32 to BS=128 elsewhere.
- **Expected gain**: 2–3× aggregate tok/s at c=32+.
- **Cost to validate**: One capacity block, long (~2 h) BS=128 compile.
- **Action**: Run after optimizations in Tier 1 land.

---

## Tier 4 — Only if nothing else helps

### 10. Write our own NKI blockwise MoE kernel

- **Evidence**: PR can't use `shard_hidden`, our FP8 runs all fall back to `shard_on_block` or `shard_on_intermediate`, both suboptimal per profile.
- **Hypothesis**: A hand-written NKI kernel matching `shard_hidden`'s intended layout could unlock the 2.6× CTE headroom and the MoE torch-fallback penalty.
- **Expected gain**: Potentially 2× FP8 throughput.
- **Cost**: 1–2 weeks of NKI engineering + testing + NxDI integration.
- **Action**: Only if #4 (wait for SDK 2.30) is blocked >2 months.

### 11. SDK + driver sweep

- **Evidence**: We ran driver 2.27.4 + neuronx-cc 2.24.5133. PR author may run a different patch level.
- **Hypothesis**: c>1 gap is sensitive to driver/runtime version (scheduling changes between patches).
- **Expected gain**: Could be 0, could be 30%. Unknown.
- **Cost**: Each SDK refresh needs one capacity block ($250–500).
- **Action**: Re-bench on next DLAMI release only if we're investigating c>1 gap and other options exhausted.

### 12. AZ / physical-machine variance investigation

- **Evidence**: c=16/c=32 deltas (29% and 35%) vs PR are suspicious — could reflect hardware heterogeneity in us-east-2b capacity pool.
- **Hypothesis**: Different physical machines serve different capacity blocks; network topology and HBM binning vary.
- **Expected gain**: Unknown; finding would be "variance is X%", not a fix.
- **Cost**: 3–4 additional capacity blocks ($1000–1500 total) to build a distribution.
- **Action**: Only justified if Tier 1+2 close most of the gap but a residual remains.

---

## Priority recommendation

| Order | Action | When |
|---|---|---|
| 1 | `py-spy` trace on c=32 serving to locate host gap | Next capacity block, within first hour |
| 2 | Flip to `shard_on_intermediate` — FP8 path only (free +6%) | Same capacity block, hour 2 |
| 3 | Flag ablation (skip_dma_token, use_index_calc_kernel, etc. from BF16 recipe into FP8) | Same capacity block, hours 2-6 |
| 4 | Re-enable NKI kernels one by one | Another capacity block |
| 5 | File compiler ticket for CTE mfu gap | No instance needed |
| 6 | Watch SDK 2.30 release notes for `shard_hidden` | No instance needed, months out |

If all of #1–#3 land, expect end-to-end throughput to move from our current 196 tok/s (c=32) toward PR's 302 tok/s, and possibly beyond.
