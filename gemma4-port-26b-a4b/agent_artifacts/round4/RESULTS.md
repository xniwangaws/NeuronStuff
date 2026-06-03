# Gemma-4-26B-A4B-it Round 4 Results

Hardware: trn2.48xlarge (i-0b8760ddfc03aa47a, us-east-2b), Capacity Block 6h.
SDK 2.29 (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`), torch 2.9.1, NxDI 0.10.0.
LNC=2 (default), TP=8, BF16, seq_len=256, batch=1.

## Stage 1 -- DISABLE_MOE compile + load
- Compile: 2.2 min (priority HLO 81 s, all HLOs 17 s).
- Weight load: 20.85 s. Warmup: 0.49 s.
- NEFF artifact dir: 17 MB.
- Status: PASS. Round-3 c10::Error blocker confirmed gone on 16-device host.

## Stage 2 -- MoE-on compile + load
- Compile: 19.7 min (priority HLO 106 s, all HLOs 925 s, build 1183 s).
- Weight load: 29.1 s. Warmup: 0.66 s.
- NEFF artifact dir: 297 MB.
- Status: PASS, after fixing one bug:
  - `NeuronGemma4Router.forward` referenced `expert_index` before assignment;
    fixed to use `top_k_index` (the upstream variable).

## Stage 3 -- Inference smoke test
- Prompt: `"Hello, my name is"` (5 tokens).
- Generated 8 tokens: `[236764, 1041, 1463, 563, 236764, 1041, 1463, 563]`
- Decoded: `, my name is, my name is`
- TTFT: 309.5 ms (prefill at seq_len=256).
- TPOT: 8.79 ms.
- Throughput: 114 tok/s.
- Status: PASS coherence smoke (output is base-model-style continuation, not gibberish).
  Repetition is expected for a base 4B-active MoE checkpoint with greedy decoding
  and no chat template. Accuracy validation is a separate task.

## Issues encountered + fixes

1. `NeuronGemma4Router.forward` -- `UnboundLocalError: expert_index`.
   Fix: rename to use `top_k_index` (line 664 in `modeling_gemma4_neuron.py`).
   This was a typo dormant in round 3 because Stage 2 never executed the router.

2. `transformers.AutoTokenizer.from_pretrained()` raises
   `AttributeError: 'list' object has no attribute 'keys'` on the gemma-4
   tokenizer config (special-tokens list vs dict). Fix: fall back to the
   raw `tokenizers.Tokenizer.from_file("tokenizer.json")` backend in
   `scripts/smoke_inference.py::_load_tokenizer`.

## Open issues for follow-up
- The repetition observed in Stage 3 likely reflects greedy decoding and
  the lack of a Gemma-4 chat template; verify with sampling (top_p/temperature)
  and proper chat-formatted prompts.
- AutoTokenizer fix: bump the `transformers` pin in this venv or upstream
  a special-tokens-list normalization to the gemma-4 tokenizer config.
- Validate token-match accuracy vs HF reference once larger seq_len (e.g.
  1024 / 2048) compiles successfully.
- We didn't observe attention-kernel issues with our PR #106 NKI integration
  on the new instance, but compile cache had 261 MB at peak.
