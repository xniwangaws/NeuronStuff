"""SDXL Neuron trace — parametric resolution version of trace_sdxl.py.

Runs verbatim trace_sdxl.py logic but reads SDXL_RES env var (1024/2048/4096)
to size latents. This delegates to the original script's working wrappers
instead of reimplementing them, avoiding trace output-type inference issues.

Run:
    SDXL_RES=1024 python trace_sdxl_res.py
    SDXL_RES=2048 python trace_sdxl_res.py
    SDXL_RES=4096 python trace_sdxl_res.py
"""
import copy, os, time
import torch, torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

RES = int(os.environ.get("SDXL_RES", "1024"))
LATENT = RES // 8
COMPILER_WORKDIR_ROOT = f"/home/ubuntu/sdxl/compile_dir_{RES}"
MODEL_ID = os.environ.get("SDXL_MODEL_ID", "/home/ubuntu/models/sdxl-base")
DTYPE = torch.bfloat16
UNET_DEVICE_IDS = [int(x) for x in os.environ.get("SDXL_UNET_CORES", "0,1,2,3").split(",")]
COMPILER_ARGS = ["--auto-cast", "matmult", "--optlevel", "1"]


# --- attention patch from notebook cell 8 (unchanged) ---------------------

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs


Attention.get_attention_scores = get_attention_scores_neuron


# --- wrappers from notebook cell 8 (verbatim) -----------------------------

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out_tuple


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(
            sample,
            timestep.float().expand((sample.shape[0],)),
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TraceableTextEncoder(nn.Module):
    """Wrap CLIP text encoder to return a flat tuple (JIT tracer can't infer dicts).

    CLIPTextModel returns BaseModelOutputWithPooling: (last_hidden_state, pooler_output, hidden_states, attentions).
    CLIPTextModelWithProjection returns CLIPTextModelOutput:   (text_embeds, last_hidden_state, hidden_states, attentions).
    We emit a flat tuple of all tensors including the 13 hidden_states so SDXL can index hidden_states[-2].
    """
    def __init__(self, text_encoder, kind):
        super().__init__()
        self.text_encoder = text_encoder
        self.kind = kind   # "clip_l" (no projection) or "clip_g" (with projection)

    def forward(self, text_input_ids):
        out = self.text_encoder(text_input_ids, output_hidden_states=True)
        if self.kind == "clip_g":
            # CLIPTextModelOutput: (text_embeds, last_hidden_state, hidden_states tuple, attentions)
            return (out[0], out[1], *out[2])
        else:
            # BaseModelOutputWithPooling: (last_hidden_state, pooler_output, hidden_states tuple, attentions)
            return (out[1], out[0], *out[2])  # synthesize text_embeds = pooler_output for compat


class TextEncoderOutputWrapper(nn.Module):
    def __init__(self, traceable_text_encoder, original_text_encoder):
        super().__init__()
        self.traceable_text_encoder = traceable_text_encoder
        self.config = original_text_encoder.config
        self.dtype = original_text_encoder.dtype
        self.device = original_text_encoder.device

    def forward(self, text_input_ids, output_hidden_states=True):
        out = self.traceable_text_encoder(text_input_ids)
        # out = (text_embeds/pooler, last_hidden, hidden_state_0, ..., hidden_state_N)
        return CLIPTextModelOutput(
            text_embeds=out[0],
            last_hidden_state=out[1],
            hidden_states=out[2:],
        )


def trace_and_save(module, example_inputs, save_path, label):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[trace] {label} -> {save_path}")
    t0 = time.time()
    traced = torch_neuronx.trace(module, example_inputs, compiler_args=COMPILER_ARGS)
    elapsed = time.time() - t0
    torch.jit.save(traced, save_path)
    size_gb = os.path.getsize(save_path) / 1e9
    print(f"[trace] {label} done in {elapsed:.1f}s, size {size_gb:.2f} GB")
    return elapsed


def main():
    os.makedirs(COMPILER_WORKDIR_ROOT, exist_ok=True)
    print(f"[config] RES={RES} LATENT={LATENT} workdir={COMPILER_WORKDIR_ROOT}")

    # Text encoders
    print("[load] pipeline for text-encoder tracing (BF16)")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    te1 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder, "clip_l"))
    te2 = copy.deepcopy(TraceableTextEncoder(pipe.text_encoder_2, "clip_g"))
    input_ids = torch.zeros((1, 77), dtype=torch.long)
    trace_and_save(te1, input_ids, f"{COMPILER_WORKDIR_ROOT}/text_encoder/model.pt", "CLIP-L")
    trace_and_save(te2, input_ids, f"{COMPILER_WORKDIR_ROOT}/text_encoder_2/model.pt", "CLIP-G")
    del te1, te2, pipe

    # UNet
    print(f"[load] pipeline for UNet tracing res={RES}")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    unet = copy.deepcopy(UNetWrap(pipe.unet))
    del pipe
    sample = torch.randn([2, 4, LATENT, LATENT], dtype=DTYPE)
    timestep = torch.tensor(999, dtype=torch.float32)
    encoder_hidden_states = torch.randn([2, 77, 2048], dtype=DTYPE)
    text_embeds = torch.randn([2, 1280], dtype=DTYPE)
    time_ids = torch.randn([2, 6], dtype=DTYPE)
    trace_and_save(unet,
                   (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
                   f"{COMPILER_WORKDIR_ROOT}/unet/model.pt", f"UNet@{RES}")
    del unet

    # VAE decoder + post_quant_conv
    print("[load] pipeline for VAE tracing")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    latent = torch.randn([1, 4, LATENT, LATENT], dtype=DTYPE)
    trace_and_save(copy.deepcopy(pipe.vae.decoder), latent,
                   f"{COMPILER_WORKDIR_ROOT}/vae_decoder/model.pt", f"VAE dec@{RES}")
    trace_and_save(copy.deepcopy(pipe.vae.post_quant_conv), latent,
                   f"{COMPILER_WORKDIR_ROOT}/vae_post_quant_conv/model.pt", f"VAE pqc@{RES}")

    print(f"[done] {COMPILER_WORKDIR_ROOT}")


if __name__ == "__main__":
    main()
