#!/usr/bin/env python3
"""BF16 BS=1 standalone NxDI smoke (reproduce PR's 29.92 tok/s claim)."""
import os, sys, time, traceback

MODEL_PATH = "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16"
COMPILED_PATH = "/opt/dlami/nvme/compiled/mimo_v2_flash_bf16_bs1/"
os.environ.setdefault("BASE_COMPILE_WORK_DIR", os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))))


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

    contrib_src = os.path.expanduser("~/neuronx-distributed-inference/contrib/models/MiMo-V2-Flash/src")
    sys.path.insert(0, contrib_src)
    from modeling_mimo_v2 import MiMoV2InferenceConfig, NeuronMiMoV2ForCausalLM

    # Exact PR "old bench" BF16 recipe for single-stream latency
    print("[bf16_bs1] PR recipe: moe_tp=64 moe_ep=1 BS=1 BF16 (no quantized)")
    neuron_config = MoENeuronConfig(
        tp_degree=64, ep_degree=1, logical_nc_config=2,
        batch_size=1, max_batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=1024, n_active_tokens=128, torch_dtype="bfloat16",
        capacity_factor=1.0, glu_mlp=True,
        moe_ep_degree=1, moe_tp_degree=64,
        context_encoding_buckets=[1024],
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config={"use_torch_block_wise": True},
        save_sharded_checkpoint=True,
        fused_qkv=False,
        flash_decoding_enabled=False,
        sequence_parallel_enabled=True,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        qkv_cte_nki_kernel_fuse_rope=False,
        attn_kernel_enabled=False,
        strided_context_parallel_kernel_enabled=False,
        normalize_top_k_affinities=True,
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(neuron_config, load_config=load_pretrained_config(hf_config=hf_config))

    t0 = time.time()
    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    print(f"[bf16_bs1] Instantiated in {time.time()-t0:.1f}s")

    if not os.path.exists(os.path.join(COMPILED_PATH, "weights")):
        print("[bf16_bs1] Compiling...")
        t0 = time.time()
        model.compile(COMPILED_PATH)
        print(f"[bf16_bs1] Compiled in {time.time()-t0:.1f}s")

    t0 = time.time()
    model.load(COMPILED_PATH, skip_warmup=False)
    print(f"[bf16_bs1] Loaded in {time.time()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    adapter = HuggingFaceGenerationAdapter(model)

    PROMPT = "Hello! Please introduce yourself in one sentence."
    inputs = tokenizer([PROMPT], return_tensors="pt", padding=True)
    gen_config = GenerationConfig(
        max_new_tokens=20, min_new_tokens=20,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )

    # Warmup + 3 timed runs
    _ = adapter.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=gen_config)

    import statistics
    times = []
    for run in range(3):
        t0 = time.time()
        output_ids = adapter.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=gen_config)
        dt = time.time() - t0
        times.append(dt)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[bf16_bs1] run={run} dt={dt:.3f}s tok/s={20/dt:.2f} text={text!r}")

    mean_dt = statistics.mean(times)
    print(f"[bf16_bs1] ==========")
    print(f"[bf16_bs1] mean_tok/s={20/mean_dt:.2f} (PR reports 29.92 tok/s)")
    print(f"[bf16_bs1] mean_dt={mean_dt:.3f}s stdev={statistics.stdev(times)*1000:.1f}ms")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
