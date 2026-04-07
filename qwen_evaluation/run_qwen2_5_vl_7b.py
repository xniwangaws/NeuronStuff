"""
Compile and test Qwen2.5-VL-7B-Instruct on Neuron.

Usage:
  python3 run_qwen2_5_vl_7b.py [--compile-only] [--text-only]
"""

import sys
import os
import argparse
import time
import logging

sys.path.insert(0, "/home/ubuntu/NeuronStuff/qwen_evaluation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen2_5_vl_7b")

MODEL_PATH = "/home/ubuntu/models/Qwen2.5-VL-7B-Instruct"
COMPILED_MODEL_PATH = "/home/ubuntu/models/Qwen2.5-VL-7B-Instruct-neuron"

# TP=4: 7 heads/rank, 1 KV head/rank (good fit for 7B with 28 heads, 4 KV heads)
TP_DEGREE = 4
BATCH_SIZE = 1
SEQ_LEN = 4096
CONTEXT_ENCODING_BUCKETS = [512, 1024, 2048, 4096]
TOKEN_GEN_BUCKETS = [512, 1024, 2048, 4096]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true", help="Only compile, don't run inference")
    parser.add_argument("--text-only", action="store_true", help="Text-only test (no vision)")
    parser.add_argument("--tp", type=int, default=TP_DEGREE, help="TP degree")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return parser.parse_args()


def compile_model(args):
    from qwen2_5_vl_7b import Qwen2_5_VLInferenceConfig, NeuronQwen2_5_VLForCausalLM
    from neuronx_distributed_inference.models.config import NeuronConfig

    logger.info(f"Compiling Qwen2.5-VL-7B with TP={args.tp}, BS={args.batch_size}, seq_len={args.seq_len}")

    # Text neuron config
    text_neuron_config = NeuronConfig(
        tp_degree=args.tp,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        seq_len=args.seq_len,
        buckets=CONTEXT_ENCODING_BUCKETS,
        token_generation_buckets=TOKEN_GEN_BUCKETS,
        torch_dtype="bfloat16",
        fused_qkv=True,
        on_device_sampling_config=None,
    )

    # Vision neuron config
    vision_neuron_config = NeuronConfig(
        tp_degree=args.tp,
        batch_size=1,
        max_batch_size=1,
        seq_len=args.seq_len,
        buckets=[1],  # 1 image at a time
        fused_qkv=True,
        torch_dtype="bfloat16",
    )

    logger.info("Loading config from %s", MODEL_PATH)
    config = Qwen2_5_VLInferenceConfig.from_pretrained(
        MODEL_PATH,
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
    )

    logger.info("Creating model...")
    model = NeuronQwen2_5_VLForCausalLM(config)

    logger.info("Compiling text model...")
    t0 = time.time()
    model.compile(compiled_model_path=COMPILED_MODEL_PATH)
    logger.info(f"Compilation done in {time.time() - t0:.1f}s")

    return model


def test_text_generation(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt")

    logger.info("Running text generation test...")
    t0 = time.time()
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
    )
    ttft = time.time() - t0
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"TTFT: {ttft:.3f}s")
    logger.info(f"Response: {response}")


def main():
    args = parse_args()

    model = compile_model(args)

    if not args.compile_only:
        logger.info("Loading compiled model...")
        model.load(COMPILED_MODEL_PATH)

        test_text_generation(model)

    logger.info("Done!")


if __name__ == "__main__":
    main()
