#!/usr/bin/env python3
"""Trace FLUX.2-klein's Qwen3-8B text encoder onto Neuron as three segments.

The klein pipeline (diffusers _get_qwen3_prompt_embeds) only consumes hidden
states 9/18/27 for a fixed [1, 512] input, so layers 27..35 and lm_head never
need to run.

Attempt 1 traced layers 0..26 as one graph: compiled fine (20 min) and matched
the pipeline reference exactly, but the 18.9 GB NEFF does not fit in a single
LNC2 core's ~22 GiB HBM (NRT_RESOURCE on load). This version cuts the stack at
the natural hidden-state boundaries into three ~6 GB segments:

  seg_a: embed_tokens + layers  0..8   input_ids  -> h9
  seg_b:                layers  9..17  h9         -> h18
  seg_c:                layers 18..26  h18        -> h27

Each segment is a real transformers Qwen3Model (layer slice, shared weight
storage, final norm replaced by Identity so the output is the raw layer
output exactly like hidden_states[k] of the full model). HF's own mask /
RoPE / position handling is reused, not re-implemented; segments b/c feed
`inputs_embeds` instead of `input_ids`.

Reference = full 36-layer model through diffusers' own
Flux2KleinPipeline._get_qwen3_prompt_embeds, so validation is against exactly
what the pipeline consumes ([1, 512, 12288]).
"""
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_neuronx
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

WEIGHTS = "/mnt/nvme/flux2-klein/weights"
OUT_DIR = Path("/mnt/nvme/flux2-klein/compiled_te")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 512
LAYERS = (9, 18, 27)
SEGMENTS = (("seg_a", 0, 9), ("seg_b", 9, 18), ("seg_c", 18, 27))
COMPILER_ARGS = "--model-type=transformer -O1 --auto-cast=none"
PROMPTS = {
    "benchmark": "A cat holding a sign that says hello world",
    "uncond": "",
}


def build_segment(full: Qwen3Model, start: int, end: int, with_embed: bool):
    """Qwen3Model over layers[start:end], sharing parameter storage with `full`."""
    cfg = copy.deepcopy(full.config)
    cfg.num_hidden_layers = end - start
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types[start:end])
    with torch.device("meta"):
        seg = Qwen3Model(cfg)
    state = {}
    for i in range(start, end):
        for k, v in full.layers[i].state_dict().items():
            state[f"layers.{i - start}.{k}"] = v
    state["embed_tokens.weight"] = full.embed_tokens.weight
    state["norm.weight"] = full.norm.weight
    seg.load_state_dict(state, strict=True, assign=True)
    seg.rotary_emb = copy.deepcopy(full.rotary_emb)  # non-persistent buffer
    seg.norm = nn.Identity()  # output raw layer output == hidden_states[k]
    if not with_embed:
        seg.embed_tokens = nn.Embedding(2, cfg.hidden_size)  # unused, keeps trace small
    seg.eval()
    return seg


class IdsSegment(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, input_ids, attention_mask):
        return self.seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state


class HiddenSegment(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, hidden, attention_mask):
        return self.seg(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state


def tokenize(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    return inputs["input_ids"], inputs["attention_mask"]


def assemble(h9, h18, h27):
    """Same layout as the pipeline: stack(dim=1) -> permute -> [B, S, 3*D]."""
    stacked = torch.stack([h9, h18, h27], dim=1)
    b, c, s, d = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(b, s, c * d)


def compare(name, ref, got, report):
    ref = ref.float()
    got = got.float()
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten(), got.flatten(), dim=0
    ).item()
    max_abs = (ref - got).abs().max().item()
    report[name] = {
        "cosine": cos,
        "max_abs_diff": max_abs,
        "mean_abs_diff": (ref - got).abs().mean().item(),
        "ref_norm": ref.norm().item(),
        "out_norm": got.norm().item(),
    }
    print(f"  {name}: cos={cos:.6f} max_abs={max_abs:.4f}", flush=True)


def main():
    print("Loading tokenizer + Qwen3-8B (BF16, eager attn)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS + "/tokenizer")
    lm = AutoModelForCausalLM.from_pretrained(
        WEIGHTS + "/text_encoder",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    lm.eval()

    print("Pipeline-exact CPU reference (full 36-layer model)...", flush=True)
    refs = {}
    with torch.no_grad():
        for name, prompt in PROMPTS.items():
            t0 = time.perf_counter()
            refs[name] = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                lm,
                tokenizer,
                prompt,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                max_sequence_length=MAX_LEN,
                hidden_states_layers=list(LAYERS),
            )
            print(f"  cpu(full) {name}: {time.perf_counter() - t0:.2f}s", flush=True)

    segs = {}
    for name, start, end in SEGMENTS:
        seg = build_segment(lm.model, start, end, with_embed=(start == 0))
        segs[name] = (IdsSegment(seg) if start == 0 else HiddenSegment(seg)).eval()
    del lm.lm_head

    examples = {name: tokenize(tokenizer, p) for name, p in PROMPTS.items()}

    print("Segmented CPU sanity check vs pipeline reference...", flush=True)
    sanity = {}
    cpu_hidden = {}
    with torch.no_grad():
        for name, (ids, mask) in examples.items():
            h9 = segs["seg_a"](ids, mask)
            h18 = segs["seg_b"](h9, mask)
            h27 = segs["seg_c"](h18, mask)
            cpu_hidden[name] = (h9, h18, h27)
            compare(name, refs[name], assemble(h9, h18, h27), sanity)
    if any(v["cosine"] < 0.9999 for v in sanity.values()):
        raise SystemExit("Segmented CPU path does not match pipeline reference; abort")

    ids, mask = examples["benchmark"]
    h9, h18, _ = cpu_hidden["benchmark"]
    example_inputs = {
        "seg_a": (ids, mask),
        "seg_b": (h9, mask),
        "seg_c": (h18, mask),
    }

    report = {
        "compiler_args": COMPILER_ARGS,
        "segments": {},
        "cpu_sanity": sanity,
        "neuron_vs_pipeline": {},
    }
    traced = {}
    for name, _, _ in SEGMENTS:
        print(f"Tracing {name} on Neuron (compile)...", flush=True)
        t0 = time.perf_counter()
        traced[name] = torch_neuronx.trace(
            segs[name],
            example_inputs[name],
            compiler_workdir=str(OUT_DIR / f"compiler_workdir_{name}"),
            compiler_args=COMPILER_ARGS,
        )
        secs = time.perf_counter() - t0
        path = OUT_DIR / f"text_encoder_{name}.pt"
        torch.jit.save(traced[name], str(path))
        neff = OUT_DIR / f"compiler_workdir_{name}" / "graph.neff"
        report["segments"][name] = {
            "compile_seconds": secs,
            "model_path": str(path),
            "neff_bytes": neff.stat().st_size if neff.exists() else None,
        }
        print(f"  {name}: compiled in {secs:.1f}s -> {path}", flush=True)

    print("Neuron end-to-end validation (3 segments chained)...", flush=True)
    with torch.no_grad():
        for name, (p_ids, p_mask) in examples.items():
            n9 = traced["seg_a"](p_ids, p_mask)
            n18 = traced["seg_b"](n9, p_mask)
            n27 = traced["seg_c"](n18, p_mask)
            compare(name, refs[name], assemble(n9, n18, n27), report["neuron_vs_pipeline"])

        for _ in range(3):
            traced["seg_c"](traced["seg_b"](traced["seg_a"](ids, mask), mask), mask)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            traced["seg_c"](traced["seg_b"](traced["seg_a"](ids, mask), mask), mask)
            times.append(time.perf_counter() - t0)
    report["neuron_latency_ms"] = {
        "mean": 1000 * sum(times) / len(times),
        "min": 1000 * min(times),
        "max": 1000 * max(times),
    }
    print(f"Neuron TE latency (3 segs, same core): {report['neuron_latency_ms']['mean']:.1f} ms mean", flush=True)

    (OUT_DIR / "te_trace_report.json").write_text(json.dumps(report, indent=2))
    print("WROTE " + str(OUT_DIR / "te_trace_report.json"), flush=True)


if __name__ == "__main__":
    main()
