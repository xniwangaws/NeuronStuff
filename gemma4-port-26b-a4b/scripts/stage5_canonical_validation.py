#!/usr/bin/env python3
"""Stage 5: canonical Gemma-4 validation for the Trainium 2 port.

This script implements the user's canonical Gemma-4 validation pattern --
using ``processor.apply_chat_template`` (with ``enable_thinking``) and
``processor.parse_response`` -- across two backends:

* ``--mode neuron``  : load the compiled NeuronGemma4ForCausalLM from
  ~/gemma4-compiled/ and run prefill+decode using the NxDI forward API
  (no HF ``generate`` -- the compiled model only exposes
  ``forward(input_ids, attention_mask, position_ids)``, see Stage 4).
* ``--mode hf-cpu``  : load HF ``Gemma4ForConditionalGeneration`` on CPU
  bf16 and call ``model.generate`` with the same template/inputs.

Notes
-----
* The installed Gemma4Processor requires Pillow (it's a multimodal
  processor). On the Neuron venv we don't want to drag in PIL, and the
  *text-only* canonical workflow is fully covered by the GemmaTokenizer:
  it exposes ``apply_chat_template`` (with ``enable_thinking``) and
  ``parse_response``, identical to what the canonical example uses
  via ``AutoProcessor``. We therefore load the tokenizer directly and
  attach the saved chat template (the local checkpoint ships the .jinja
  separately because tokenizer_config.json doesn't embed it).
* For the HF baseline we use the cpu-baseline-venv (transformers
  5.10.0.dev0 -- the only version with full Gemma-4 support).
* For the Neuron path we use /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
  (transformers 4.57.6) which is enough to run the compiled model and to
  use the GemmaTokenizer + chat template; ``parse_response`` exists in
  4.57's tokenization_utils_base too.

Usage
-----
    # On Neuron venv, after compilation:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/gemma4-port
    PYTHONPATH=. python scripts/stage5_canonical_validation.py --mode neuron \
        --out /home/ubuntu/stage5_neuron_results.json

    # On HF baseline venv, no MoE acceleration:
    source /home/ubuntu/cpu-baseline-venv/bin/activate
    python ~/gemma4-port/scripts/stage5_canonical_validation.py --mode hf-cpu \
        --max-new-tokens 32 \
        --out /home/ubuntu/stage5_hfcpu_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "/home/ubuntu/gemma4-26b-a4b")
COMPILED_PATH = os.environ.get("GEMMA4_COMPILED_PATH", "/home/ubuntu/gemma4-compiled")
CHAT_TPL_PATH = os.environ.get("GEMMA4_CHAT_TPL", "/home/ubuntu/gemma4_chat_template.jinja")
TP_DEGREE = int(os.environ.get("GEMMA4_TP_DEGREE", "8"))
BATCH_SIZE = int(os.environ.get("GEMMA4_BATCH_SIZE", "1"))
SEQ_LEN = int(os.environ.get("GEMMA4_SEQ_LEN", "256"))
MOE_EP_DEGREE = int(os.environ.get("GEMMA4_MOE_EP_DEGREE", "1"))
MOE_TP_DEGREE = int(os.environ.get("GEMMA4_MOE_TP_DEGREE", str(TP_DEGREE)))
SEED = 42

# EOS list from generation_config.json (Stage 4): <eos>=1, <turn|>=106,
# comma=50 (used inside tool-response only). For chat answers we let
# either <turn|> or <eos> end the turn.
EOS_TOKEN_IDS = [1, 106]

# Three message sets at increasing complexity (canonical user spec).
SYSTEM_TEXT = "You are a helpful assistant."
MESSAGE_SETS = [
    ("joke",     "Write a short joke about saving RAM."),
    ("capital",  "What is the capital of France?"),
    ("quantum",  "Explain quantum entanglement in two sentences."),
]


# ---------------------------------------------------------------------------
# Tokenizer / chat-template helpers (venv-portable)
# ---------------------------------------------------------------------------
# The Neuron venv ships transformers 4.57.6, which:
#   * cannot load the Gemma-4 tokenizer via AutoTokenizer (it crashes on
#     extra_special_tokens, see Stage 4),
#   * does not have ``tokenizer.parse_response`` (added in 5.x).
# The HF-baseline venv ships transformers 5.10.0.dev0, which works fine.
# To keep ONE script across both, we wrap a small adapter that can fall back
# to the raw ``tokenizers`` library + a Jinja2 render of the saved
# .jinja chat template + a regex reimplementation of ``parse_response``.

# Schema for parse_response, copied from the GemmaTokenizer response_schema
# (top-level x-regex). With re.DOTALL this matches Gemma-4's
# <|channel>thought\n...<channel|>...<turn|> output.
_GEMMA4_PARSE_REGEX = (
    r"(\<\|channel\>thought\n(?P<thinking>.*?)\<channel\|\>)?"
    r"(?P<tool_calls>\<\|tool_call\>.*\<tool_call\|\>)?"
    r"(?P<content>(?:(?!\<turn\|\>)(?!\<\|tool_response\>).)+)?"
    r"(?:\<turn\|\>|\<\|tool_response\>)?"
)


class Gemma4ChatAdapter:
    """Backend-agnostic Gemma-4 chat tokenizer + parse_response.

    Tries transformers 5.x AutoTokenizer first (canonical, exact match for
    the user's spec). Falls back to a Jinja2 + tokenizers.Tokenizer
    pipeline that reproduces the same outputs on transformers 4.57.6.
    """

    def __init__(self, model_path: str, chat_tpl_path: str):
        self.model_path = model_path
        self.chat_tpl_path = chat_tpl_path
        self._tok = None  # transformers tokenizer (preferred)
        self._raw = None  # tokenizers.Tokenizer (fallback)
        self._template = None  # Jinja2 template (fallback)
        self._chat_tpl_text = ""

        # Try the canonical path first.
        try:
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(model_path)
            if not getattr(t, "chat_template", None):
                with open(chat_tpl_path) as f:
                    t.chat_template = f.read()
            # Sanity check: apply_chat_template must accept enable_thinking.
            t.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            self._tok = t
            self.kind = "transformers"
            print(f"  Gemma4ChatAdapter: using transformers {type(t).__name__}")
            return
        except Exception as e:
            print(f"  Gemma4ChatAdapter: transformers path failed ({e}), falling back")

        # Fallback path: tokenizers + Jinja2 + custom parse_response.
        from tokenizers import Tokenizer
        import jinja2
        self._raw = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        with open(chat_tpl_path) as f:
            self._chat_tpl_text = f.read()
        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=False,
            keep_trailing_newline=False,
        )
        # The Jinja chat template uses `raise_exception` (HF convention).
        def _raise(msg):
            raise jinja2.TemplateError(msg)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self._template = env.from_string(self._chat_tpl_text)
        self.kind = "fallback"
        print(f"  Gemma4ChatAdapter: using fallback tokenizers + jinja2")

    # -- canonical apply_chat_template ----------------------------------

    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        if self._tok is not None:
            return self._tok.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        # Fallback render via Jinja2.
        # Drive the official template; bos_token / eos_token mirrors what
        # tokenizer_config.json says.
        bos = "<bos>"
        text = self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=None,
            bos_token=bos,
            eos_token="<eos>",
        )
        if tokenize:
            return self._raw.encode(text, add_special_tokens=False).ids
        return text

    def encode(self, text):
        if self._tok is not None:
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            # transformers may return a flat list or a 1-D list-of-ints; normalise.
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return list(ids)
        return self._raw.encode(text, add_special_tokens=False).ids

    def decode(self, ids, *, skip_special_tokens=True):
        if self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
        return self._raw.decode(ids, skip_special_tokens=skip_special_tokens)

    # -- canonical parse_response ---------------------------------------

    def parse_response(self, response_text: str):
        if self._tok is not None and hasattr(self._tok, "parse_response"):
            return self._tok.parse_response(response_text)
        # Fallback: regex-driven, mirrors the schema for chat output.
        import re
        out = {"role": "assistant"}
        m = re.match(_GEMMA4_PARSE_REGEX, response_text, re.DOTALL)
        if not m:
            out["content"] = response_text
            return out
        if m.group("thinking"):
            out["thinking"] = m.group("thinking")
        if m.group("content"):
            out["content"] = m.group("content")
        if m.group("tool_calls"):
            out["tool_calls_raw"] = m.group("tool_calls")
        return out


def load_tokenizer_with_template(model_path: str, chat_tpl_path: str):
    return Gemma4ChatAdapter(model_path, chat_tpl_path)


def render_messages(tokenizer, system_text, user_text, *, enable_thinking):
    """Build the canonical chat-template prompt, returning text + ids."""
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer.encode(text)
    return text, torch.tensor([ids], dtype=torch.long)


def safe_parse_response(tokenizer, response_text):
    """Run tokenizer.parse_response, returning a JSON-friendly dict."""
    try:
        parsed = tokenizer.parse_response(response_text)
    except Exception as e:  # pragma: no cover -- best-effort
        return {"_error": f"{type(e).__name__}: {e}"}
    if isinstance(parsed, dict):
        return {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                    else str(v))
                for k, v in parsed.items()}
    return {"_repr": repr(parsed)}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_from_logits(logits, temperature, top_k, top_p, generator):
    if temperature is None or temperature == 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk_idx, topk_vals)
        logits = mask
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumprobs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_idx = sorted_idx[remove_sorted]
        logits[remove_idx] = float("-inf")
    probs = torch.softmax(logits.float(), dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_tok.item())


# ---------------------------------------------------------------------------
# Neuron path
# ---------------------------------------------------------------------------

def _neuron_logits_from_outputs(out):
    if hasattr(out, "logits") and out.logits is not None:
        lg = out.logits
        return lg[:, -1, :] if lg.dim() == 3 else lg
    if hasattr(out, "tokens") and out.tokens is not None:
        return None
    raise RuntimeError(f"Unexpected output: {type(out)}")


def neuron_generate(model, prompt_ids, *, max_new_tokens, do_sample,
                    temperature, top_k, top_p, seed, n_positions):
    """Manual prefill + decode loop matching Stage 4 plumbing.

    NxDI's compiled model exposes a static ``forward`` whose batch_size /
    seq_len are fixed at compile time (1 / 256 here). HF's ``generate``
    isn't compatible -- we adapt by padding the prompt to seq_len and
    advancing position_ids one token at a time.
    """
    seq_len = prompt_ids.shape[1]
    if seq_len > n_positions:
        raise RuntimeError(f"prompt {seq_len} > compiled seq_len {n_positions}")

    pad_len = n_positions - seq_len
    input_ids = torch.cat(
        [prompt_ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
    )
    attention_mask = torch.cat(
        [torch.ones(1, seq_len, dtype=torch.long),
         torch.zeros(1, pad_len, dtype=torch.long)],
        dim=1,
    )
    position_ids = torch.zeros(1, n_positions, dtype=torch.long)
    position_ids[0, :seq_len] = torch.arange(seq_len)

    gen = torch.Generator()
    gen.manual_seed(seed)

    timing = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids)
    timing["ttft_ms"] = (time.perf_counter() - t0) * 1000

    logits = _neuron_logits_from_outputs(outputs)
    if logits is None:
        next_tok = int(outputs.tokens[:, -1:].item())
    elif do_sample:
        next_tok = _sample_from_logits(logits[0], temperature, top_k, top_p, gen)
    else:
        next_tok = int(torch.argmax(logits[0]).item())

    generated = [next_tok]
    cur_pos = seq_len
    next_token = torch.tensor([[next_tok]], dtype=torch.long)
    t_gen = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if cur_pos >= n_positions:
            break
        attention_mask[0, cur_pos] = 1
        with torch.no_grad():
            outputs = model(input_ids=next_token,
                            attention_mask=attention_mask,
                            position_ids=torch.tensor([[cur_pos]]))
        cur_pos += 1
        logits = _neuron_logits_from_outputs(outputs)
        if logits is None:
            next_tok = int(outputs.tokens[:, -1:].item())
        elif do_sample:
            next_tok = _sample_from_logits(logits[0], temperature, top_k, top_p, gen)
        else:
            next_tok = int(torch.argmax(logits[0]).item())
        generated.append(next_tok)
        next_token = torch.tensor([[next_tok]], dtype=torch.long)
        if next_tok in EOS_TOKEN_IDS:
            break
    t_end = time.perf_counter()
    n_dec = max(1, len(generated) - 1)
    timing["tpot_ms"] = (t_end - t_gen) / n_dec * 1000
    timing["total_tokens"] = len(generated)
    return generated, timing


def load_neuron_model():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neuron_port import ndxi_patch  # noqa: E402
    ndxi_patch.apply_patch()

    from neuron_port.modeling_gemma4_neuron import (  # noqa: E402
        Gemma4InferenceConfig,
        Gemma4NeuronConfig,
        NeuronGemma4ForCausalLM,
    )

    neuron_config = Gemma4NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        on_device_sampling_config=None,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        attn_kernel_enabled=False,
        moe_ep_degree=MOE_EP_DEGREE,
        moe_tp_degree=MOE_TP_DEGREE,
        glu_mlp=True,
        glu_type="glu",
        router_act_fn="softmax",
        router_dtype="float32",
        disable_normalize_top_k_affinities=True,
    )

    def load_config_fn(config_obj):
        with open(os.path.join(MODEL_PATH, "config.json")) as f:
            cd = json.load(f)
        for k, v in cd.items():
            setattr(config_obj, k, v)

    cfg = Gemma4InferenceConfig(neuron_config=neuron_config, load_config=load_config_fn)
    print(f"Loading compiled NeuronGemma4ForCausalLM from {COMPILED_PATH} ...")
    t0 = time.perf_counter()
    model = NeuronGemma4ForCausalLM(MODEL_PATH, cfg)
    model.load(COMPILED_PATH)
    print(f"  load took {time.perf_counter() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# HF CPU path
# ---------------------------------------------------------------------------

def load_hf_cpu_model():
    """Load Gemma4ForConditionalGeneration on CPU bf16."""
    from transformers import Gemma4ForConditionalGeneration
    print(f"Loading Gemma4ForConditionalGeneration (bf16 CPU) from {MODEL_PATH} ...")
    t0 = time.perf_counter()
    # Note: avoid passing device_map="cpu" -- it triggers an accelerate dependency.
    # CPU is the default when no device_map is set.
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  load took {time.perf_counter() - t0:.1f}s")
    return model


def hf_cpu_generate(model, prompt_ids, *, max_new_tokens, do_sample,
                    temperature, top_k, top_p, seed):
    """Use HF ``generate`` with deterministic seed for sampling."""
    if do_sample:
        torch.manual_seed(seed)
    timing = {}
    t0 = time.perf_counter()
    gen_kwargs = dict(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_TOKEN_IDS,
        pad_token_id=0,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    dt = time.perf_counter() - t0
    full = out[0].tolist()
    gen_ids = full[prompt_ids.shape[1]:]
    timing["wall_s"] = dt
    timing["total_tokens"] = len(gen_ids)
    timing["tpot_ms"] = (dt / max(1, len(gen_ids))) * 1000
    return gen_ids, timing


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(label, user_text, tokenizer, *, mode, model, max_new_tokens, enable_thinking,
            n_positions=SEQ_LEN):
    print(f"\n----- [{mode}] [{label}] thinking={enable_thinking} -----")
    print(f"  user: {user_text!r}")

    text, ids = render_messages(tokenizer, SYSTEM_TEXT, user_text,
                                enable_thinking=enable_thinking)
    print(f"  rendered ({ids.shape[1]} tok) tail: {text[-160:]!r}")

    out_block = {
        "user_text": user_text,
        "system_text": SYSTEM_TEXT,
        "enable_thinking": enable_thinking,
        "rendered_prompt": text,
        "prompt_ids": ids[0].tolist(),
    }

    for kind in ("greedy", "sample"):
        do_sample = (kind == "sample")
        if mode == "neuron":
            gen_ids, timing = neuron_generate(
                model, ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=1.0, top_k=64, top_p=0.95,
                seed=SEED, n_positions=n_positions,
            )
        else:
            gen_ids, timing = hf_cpu_generate(
                model, ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=1.0, top_k=64, top_p=0.95,
                seed=SEED,
            )
        # Decode response keeping special tokens for parse_response,
        # then a clean text view too.
        response_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        response_clean = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = safe_parse_response(tokenizer, response_raw)
        print(f"  {kind.upper():6s} ids[:16]={gen_ids[:16]}")
        print(f"  {kind.upper():6s} text   ={response_clean!r}")
        print(f"  {kind.upper():6s} parsed ={parsed}")
        print(f"  {kind.upper():6s} timing ={timing}")
        out_block[kind] = {
            "tokens": gen_ids,
            "text_clean": response_clean,
            "text_raw": response_raw,
            "parsed": parsed,
            "timing": timing,
        }
    return out_block


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["neuron", "hf-cpu"], required=True)
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--thinking", choices=["both", "off", "on"], default="both")
    args = p.parse_args()

    print("=" * 80)
    print(f"Stage 5 canonical validation -- mode={args.mode}")
    print(f"  model path:    {MODEL_PATH}")
    print(f"  chat template: {CHAT_TPL_PATH}")
    print(f"  out:           {args.out}")
    print(f"  max_new_tok:   {args.max_new_tokens}")
    print("=" * 80)

    tok = load_tokenizer_with_template(MODEL_PATH, CHAT_TPL_PATH)
    tpl_len = len(getattr(tok, "_chat_tpl_text", "")) if tok.kind == "fallback" \
        else len(getattr(tok._tok, "chat_template", "") or "")
    print(f"  tokenizer adapter: kind={tok.kind}, "
          f"chat_template={tpl_len} chars, "
          f"has parse_response=True (adapter)")

    if args.mode == "neuron":
        model = load_neuron_model()
    else:
        model = load_hf_cpu_model()

    thinking_modes = []
    if args.thinking in ("both", "off"):
        thinking_modes.append(False)
    if args.thinking in ("both", "on"):
        thinking_modes.append(True)

    results = {"mode": args.mode, "max_new_tokens": args.max_new_tokens, "by_prompt": {}}
    for label, user_text in MESSAGE_SETS:
        results["by_prompt"][label] = {}
        for et in thinking_modes:
            block = run_one(label, user_text, tok,
                            mode=args.mode, model=model,
                            max_new_tokens=args.max_new_tokens,
                            enable_thinking=et)
            results["by_prompt"][label][f"thinking_{'on' if et else 'off'}"] = block
            # Persist incrementally so an interruption keeps partial data.
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
