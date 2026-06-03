#!/usr/bin/env python3
"""Stage 4: chat-templated inference for Gemma-4-26B-A4B-it on Trainium 2.

Implements the official Gemma-4 chat template manually (since installed
transformers 4.57.6 cannot load the Gemma-4 tokenizer via AutoTokenizer):

  <bos><|turn>user
  {user_text}<turn|>
  <|turn>model
  <|channel>thought<channel|>

Then runs greedy + sampling generations on three test prompts, decodes,
and writes a structured log usable for CPU-baseline comparison.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/gemma4-port
    PYTHONPATH=. python scripts/stage4_chat_inference.py 2>&1 | tee ~/stage4_chat.log
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from neuron_port import ndxi_patch  # noqa: E402

ndxi_patch.apply_patch()

from neuron_port.modeling_gemma4_neuron import (  # noqa: E402
    Gemma4InferenceConfig,
    Gemma4NeuronConfig,
    NeuronGemma4ForCausalLM,
)


MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "/home/ubuntu/gemma4-26b-a4b")
COMPILED_PATH = os.environ.get("GEMMA4_COMPILED_PATH", "/home/ubuntu/gemma4-compiled")
TP_DEGREE = int(os.environ.get("GEMMA4_TP_DEGREE", "8"))
BATCH_SIZE = int(os.environ.get("GEMMA4_BATCH_SIZE", "1"))
SEQ_LEN = int(os.environ.get("GEMMA4_SEQ_LEN", "256"))
MAX_NEW_TOKENS = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "64"))
MOE_EP_DEGREE = int(os.environ.get("GEMMA4_MOE_EP_DEGREE", "1"))
MOE_TP_DEGREE = int(os.environ.get("GEMMA4_MOE_TP_DEGREE", str(TP_DEGREE)))

# Generation defaults from generation_config.json:
#   do_sample=True, temperature=1.0, top_k=64, top_p=0.95
# EOS list: [1, 106, 50] (eos, <turn|>, comma — comma is for tool-response)
EOS_TOKEN_IDS = [1, 106, 50]
SEED = 42

# Special tokens (verified from tokenizer.json)
BOS_ID = 2
SOT_ID = 105    # <|turn>
EOT_ID = 106    # <turn|>
SOC_ID = 100    # <|channel>
EOC_ID = 101    # <channel|>

PROMPTS = [
    ("easy",   "What is 2+2?"),
    ("medium", "Name three primary colors."),
    ("open",   "Write a haiku about autumn."),
]


def build_chat_prompt(tokenizer, user_text: str) -> torch.Tensor:
    """Render the official Gemma-4 chat template manually.

    Format (no system, enable_thinking=False, add_generation_prompt=True):
      <bos><|turn>user\n{user_text}<turn|>\n<|turn>model\n<|channel>thought<channel|>
    """
    # We tokenize the *text* parts and stitch with explicit special-token IDs.
    pre_user_text = "user\n"            # role line content (no leading newline)
    user_payload   = user_text.strip()
    after_user     = "\nmodel\n"        # role line for model + newline before channel
    # Encode each text part with the raw tokenizer (no add_special_tokens).
    enc_pre_user = tokenizer.encode(pre_user_text)
    enc_payload  = tokenizer.encode(user_payload)
    enc_after    = tokenizer.encode(after_user)

    ids = []
    ids.append(BOS_ID)
    ids.append(SOT_ID)         # <|turn>
    ids += enc_pre_user
    ids += enc_payload
    ids.append(EOT_ID)         # <turn|>
    # The template emits a literal newline after <turn|> before the next <|turn>:
    # "<turn|>\n<|turn>model\n<|channel>thought<channel|>"
    enc_newline = tokenizer.encode("\n")
    ids += enc_newline
    ids.append(SOT_ID)
    ids += enc_after           # "model\n"
    ids.append(SOC_ID)
    ids += tokenizer.encode("thought")
    ids.append(EOC_ID)

    return torch.tensor([ids], dtype=torch.long)


def _sample_from_logits(logits, temperature, top_k, top_p, generator):
    """Apply temperature/top-k/top-p sampling. logits shape: [V]."""
    if temperature is None or temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    # top-k
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk_idx, topk_vals)
        logits = mask

    # top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        # tokens to remove: those whose *cumulative* prob *strictly* exceeds top_p,
        # but we always keep the first (highest) token
        remove_sorted = cumprobs > top_p
        # shift right by 1 so we keep the first token that pushes us over
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_idx = sorted_idx[remove_sorted]
        logits[remove_idx] = float("-inf")

    probs = torch.softmax(logits.float(), dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_tok.item())


def _is_eos(tok_id):
    return tok_id in EOS_TOKEN_IDS


def generate(model, tokenizer, prompt_ids, *, max_new_tokens, do_sample,
             temperature=1.0, top_k=64, top_p=0.95, seed=42):
    """Manual prefill + decode loop."""
    seq_len = prompt_ids.shape[1]
    n_positions = SEQ_LEN
    if seq_len > n_positions:
        raise RuntimeError(f"prompt {seq_len} > compiled seq_len {n_positions}")

    pad_len = n_positions - seq_len
    input_ids_padded = torch.cat(
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
        outputs = model(input_ids=input_ids_padded,
                        attention_mask=attention_mask,
                        position_ids=position_ids)
    timing["ttft_ms"] = (time.perf_counter() - t0) * 1000

    def _logits_from_outputs(out):
        if hasattr(out, "logits") and out.logits is not None:
            lg = out.logits
            return lg[:, -1, :] if lg.dim() == 3 else lg
        if hasattr(out, "tokens") and out.tokens is not None:
            # On-device sampling path — fall back to argmax assumption
            return None
        raise RuntimeError(f"Unexpected output: {type(out)}")

    logits = _logits_from_outputs(outputs)
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
        logits = _logits_from_outputs(outputs)
        if logits is None:
            next_tok = int(outputs.tokens[:, -1:].item())
        elif do_sample:
            next_tok = _sample_from_logits(logits[0], temperature, top_k, top_p, gen)
        else:
            next_tok = int(torch.argmax(logits[0]).item())
        generated.append(next_tok)
        next_token = torch.tensor([[next_tok]], dtype=torch.long)
        if _is_eos(next_tok):
            break
    t_end = time.perf_counter()
    n_dec = max(1, len(generated) - 1)
    timing["tpot_ms"] = (t_end - t_gen) / n_dec * 1000
    timing["total_tokens"] = len(generated)
    return generated, timing


def _load_tokenizer(model_path):
    from tokenizers import Tokenizer

    class _Wrapped:
        def __init__(self, t):
            self._t = t
            self.eos_token_id = EOS_TOKEN_IDS

        def encode(self, text):
            # Encode without prepending special tokens (we add BOS manually)
            return self._t.encode(text, add_special_tokens=False).ids

        def decode(self, ids, skip_special_tokens=True):
            return self._t.decode(ids, skip_special_tokens=skip_special_tokens)

    t = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    return _Wrapped(t)


def create_config(model_path: str) -> Gemma4InferenceConfig:
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
        with open(os.path.join(model_path, "config.json")) as f:
            cd = json.load(f)
        for k, v in cd.items():
            setattr(config_obj, k, v)

    return Gemma4InferenceConfig(
        neuron_config=neuron_config, load_config=load_config_fn
    )


def main():
    print("=" * 80)
    print("Gemma-4-26B-A4B-it Stage 4 chat inference")
    print(f"  model_path:    {MODEL_PATH}")
    print(f"  compiled_path: {COMPILED_PATH}")
    print(f"  tp_degree:     {TP_DEGREE}")
    print(f"  seq_len:       {SEQ_LEN}")
    print(f"  max_new_tokens:{MAX_NEW_TOKENS}")
    print("=" * 80)

    if not Path(COMPILED_PATH).exists():
        print(f"ERROR: compiled path missing", file=sys.stderr)
        return 1

    config = create_config(MODEL_PATH)
    print("Loading compiled model ...")
    t0 = time.perf_counter()
    model = NeuronGemma4ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH)
    print(f"Load took {time.perf_counter() - t0:.1f}s")

    print("Loading tokenizer ...")
    tok = _load_tokenizer(MODEL_PATH)

    results = {}
    for label, user_text in PROMPTS:
        prompt_ids = build_chat_prompt(tok, user_text)
        prompt_token_list = prompt_ids[0].tolist()
        print(f"\n----- prompt[{label}]: {user_text!r} -----")
        print(f"  rendered prompt ids ({len(prompt_token_list)} tokens): {prompt_token_list}")
        # Decode with special tokens to confirm template.
        rendered = tok.decode(prompt_token_list, skip_special_tokens=False)
        print(f"  rendered prompt text: {rendered!r}")
        results[label] = {
            "user_text": user_text,
            "prompt_ids": prompt_token_list,
            "rendered_prompt": rendered,
        }

        # Greedy
        toks_g, t_g = generate(
            model, tok, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        text_g = tok.decode(toks_g, skip_special_tokens=True)
        print(f"\n  GREEDY  ids: {toks_g}")
        print(f"  GREEDY text: {text_g!r}")
        print(f"  GREEDY timing: {t_g}")
        results[label]["greedy"] = {
            "tokens": toks_g, "text": text_g, "timing": t_g,
        }

        # Sampling
        toks_s, t_s = generate(
            model, tok, prompt_ids, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=1.0, top_k=64, top_p=0.95, seed=SEED,
        )
        text_s = tok.decode(toks_s, skip_special_tokens=True)
        print(f"\n  SAMPLE  ids: {toks_s}")
        print(f"  SAMPLE text: {text_s!r}")
        print(f"  SAMPLE timing: {t_s}")
        results[label]["sample"] = {
            "tokens": toks_s, "text": text_s, "timing": t_s,
            "params": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": SEED},
        }

    # Save JSON for downstream comparison.
    out_path = os.environ.get("STAGE4_OUT_JSON", "/home/ubuntu/stage4_neuron_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
