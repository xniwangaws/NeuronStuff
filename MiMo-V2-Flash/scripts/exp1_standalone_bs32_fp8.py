#!/usr/bin/env python3
"""FP8 standalone BS=32 (no vLLM) — attribute host gap at concurrency > 1."""
import os, sys, time, traceback

MODEL_PATH = os.environ.get("MIMO_V2_FLASH_MODEL_PATH", "/opt/dlami/nvme/models/MiMo-V2-Flash-Neuron-FP8")
COMPILED_PATH = os.environ.get("MIMO_V2_FLASH_COMPILED_PATH", "/opt/dlami/nvme/compiled/mimo_v2_flash_bs32_exp1/")
TP_DEGREE = 64
SEQ_LEN = 1024
BATCH_SIZE = 32
CTX_BATCH_SIZE = 1
MOE_TP = 1
MOE_EP = 64
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "90"))
INPUT_LEN = int(os.environ.get("INPUT_LEN", "900"))
os.environ.setdefault("BASE_COMPILE_WORK_DIR", os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))))


def main():
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

    contrib_src = os.path.expanduser("~/neuronx-distributed-inference/contrib/models/MiMo-V2-Flash/src")
    sys.path.insert(0, contrib_src)
    from modeling_mimo_v2 import MiMoV2InferenceConfig, NeuronMiMoV2ForCausalLM

    print(f"[exp1] MODEL_PATH={MODEL_PATH} COMPILED={COMPILED_PATH} BS={BATCH_SIZE} MOE_EP={MOE_EP}")

    # Change this key to use_shard_on_intermediate_dynamic_while for the +6% variant
    blockwise = {"use_shard_on_block_dynamic_while": True, "block_sharding_strategy": "PING_PONG"}

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE, ep_degree=1, logical_nc_config=2,
        batch_size=BATCH_SIZE, max_batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE, tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN, n_active_tokens=128, torch_dtype="bfloat16",
        capacity_factor=1.0, glu_mlp=True,
        moe_ep_degree=MOE_EP, moe_tp_degree=MOE_TP,
        context_encoding_buckets=[SEQ_LEN],
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config=blockwise,
        save_sharded_checkpoint=True, quantized=True,
        quantized_checkpoints_path=MODEL_PATH, quantization_dtype="f8e4m3",
        quantization_type="blockwise_symmetric",
        quantization_block_axis=[1, 2], quantization_block_size=[128, 128],
        modules_to_not_convert=["embed_tokens", "lm_head", "norm", "router", "o_proj"],
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(neuron_config, load_config=load_pretrained_config(hf_config=hf_config))

    t0 = time.time()
    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    print(f"[exp1] Instantiated in {time.time()-t0:.1f}s")

    if not os.path.exists(os.path.join(COMPILED_PATH, "weights")):
        t0 = time.time()
        model.compile(COMPILED_PATH)
        print(f"[exp1] Compiled in {time.time()-t0:.1f}s")

    t0 = time.time()
    model.load(COMPILED_PATH, skip_warmup=False)
    print(f"[exp1] Loaded in {time.time()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    adapter = HuggingFaceGenerationAdapter(model)

    import torch, random, statistics
    vocab_size = tokenizer.vocab_size
    random.seed(0)
    input_ids = torch.tensor([[random.randint(100, vocab_size - 1) for _ in range(INPUT_LEN)] for _ in range(BATCH_SIZE)])
    attention_mask = torch.ones_like(input_ids)
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )

    _ = adapter.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)
    print("[exp1] Warmup done")

    times = []
    for run in range(3):
        t0 = time.time()
        _ = adapter.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)
        dt = time.time() - t0
        times.append(dt)
        tot = BATCH_SIZE * MAX_NEW_TOKENS
        print(f"[exp1] run={run} dt={dt:.3f}s tot_tok={tot} overall={tot/dt:.2f}tok/s TPOT_wall={dt/MAX_NEW_TOKENS*1000:.2f}ms")

    mean_dt = statistics.mean(times)
    tot = BATCH_SIZE * MAX_NEW_TOKENS
    print(f"[exp1] ==========")
    print(f"[exp1] N=3 mean_dt={mean_dt:.3f}s stdev={statistics.stdev(times)*1000:.1f}ms")
    print(f"[exp1] overall_throughput={tot/mean_dt:.2f}tok/s TPOT_wallclock={mean_dt/MAX_NEW_TOKENS*1000:.2f}ms")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
