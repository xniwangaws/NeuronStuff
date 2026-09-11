#!/usr/bin/env python3
"""Load the three traced Qwen3 segments on logical cores 1/2/3 and validate.

The trace script's in-process validation loaded all three NEFFs onto one core
(24.5 GB > 24 GB HBM). Deployment places one segment per core, which is what
this script checks: numerics vs the pipeline-exact CPU reference, per-segment
placement, and chained latency.
"""
import json
import time
from pathlib import Path

import torch
import torch_neuronx
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS = "/mnt/nvme/flux2-klein/weights"
TE_DIR = Path("/mnt/nvme/flux2-klein/compiled_te")
MAX_LEN = 512
LAYERS = (9, 18, 27)
PLACEMENT = {"seg_a": 1, "seg_b": 2, "seg_c": 3}
PROMPTS = {
    "benchmark": "A cat holding a sign that says hello world",
    "uncond": "",
}


def tokenize(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN
    )
    return inputs["input_ids"], inputs["attention_mask"]


def assemble(h9, h18, h27):
    stacked = torch.stack([h9, h18, h27], dim=1)
    b, c, s, d = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(b, s, c * d)


def main():
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS + "/tokenizer")
    print("Loading full Qwen3-8B on CPU for pipeline-exact reference...", flush=True)
    lm = AutoModelForCausalLM.from_pretrained(
        WEIGHTS + "/text_encoder", torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    refs = {}
    with torch.no_grad():
        for name, prompt in PROMPTS.items():
            refs[name] = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                lm, tokenizer, prompt, dtype=torch.bfloat16, device=torch.device("cpu"),
                max_sequence_length=MAX_LEN, hidden_states_layers=list(LAYERS),
            )
    del lm

    segs = {}
    load_seconds = {}
    for name, core in PLACEMENT.items():
        t0 = time.perf_counter()
        traced = torch.jit.load(str(TE_DIR / f"text_encoder_{name}.pt"))
        torch_neuronx.experimental.set_neuron_cores(traced, start_nc=core, nc_count=1)
        segs[name] = traced
        load_seconds[name] = time.perf_counter() - t0
        print(f"  loaded {name} -> core {core} ({load_seconds[name]:.1f}s)", flush=True)

    def run(ids, mask):
        h9 = segs["seg_a"](ids, mask)
        h18 = segs["seg_b"](h9, mask)
        h27 = segs["seg_c"](h18, mask)
        return assemble(h9, h18, h27)

    report = {"placement": PLACEMENT, "load_seconds_including_first_exec": {}, "neuron_vs_pipeline": {}}
    examples = {n: tokenize(tokenizer, p) for n, p in PROMPTS.items()}
    with torch.no_grad():
        for name, (ids, mask) in examples.items():
            t0 = time.perf_counter()
            got = run(ids, mask)
            first = time.perf_counter() - t0
            ref = refs[name].float()
            got = got.float()
            cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
            report["neuron_vs_pipeline"][name] = {
                "cosine": cos,
                "max_abs_diff": (ref - got).abs().max().item(),
                "mean_abs_diff": (ref - got).abs().mean().item(),
                "ref_norm": ref.norm().item(),
                "out_norm": got.norm().item(),
                "first_call_seconds": first,
            }
            print(f"  {name}: cos={cos:.6f} max_abs={report['neuron_vs_pipeline'][name]['max_abs_diff']:.4f} first_call={first:.2f}s", flush=True)

        ids, mask = examples["benchmark"]
        for _ in range(3):
            run(ids, mask)
        times = []
        per_seg = {k: [] for k in segs}
        for _ in range(10):
            t0 = time.perf_counter()
            t1 = time.perf_counter(); h9 = segs["seg_a"](ids, mask); per_seg["seg_a"].append(time.perf_counter() - t1)
            t1 = time.perf_counter(); h18 = segs["seg_b"](h9, mask); per_seg["seg_b"].append(time.perf_counter() - t1)
            t1 = time.perf_counter(); segs["seg_c"](h18, mask); per_seg["seg_c"].append(time.perf_counter() - t1)
            times.append(time.perf_counter() - t0)
    report["neuron_latency_ms"] = {
        "chained_mean": 1000 * sum(times) / len(times),
        "chained_min": 1000 * min(times),
        **{f"{k}_mean": 1000 * sum(v) / len(v) for k, v in per_seg.items()},
    }
    print(f"Neuron TE chained latency: {report['neuron_latency_ms']['chained_mean']:.1f} ms "
          f"(a={report['neuron_latency_ms']['seg_a_mean']:.0f} b={report['neuron_latency_ms']['seg_b_mean']:.0f} c={report['neuron_latency_ms']['seg_c_mean']:.0f})", flush=True)
    (TE_DIR / "te_validation_report.json").write_text(json.dumps(report, indent=2))
    print("WROTE " + str(TE_DIR / "te_validation_report.json"), flush=True)


if __name__ == "__main__":
    main()
