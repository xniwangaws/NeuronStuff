"""Jim Burtoft's SDXL Tile-4K pipeline on Trn2.

Workflow:
1) txt2img base @ 1024×1024 with traced 1K UNet/VAE/text-encoders
2) PIL bicubic upscale -> 4096×4096
3) Split into overlapping 1024×1024 tiles
4) Per-tile img2img refinement using same 1K UNet (no recompile)
5) Gaussian-weighted blend tiles back into final 4K image
"""
import os, sys, time, copy, math, json
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

MODEL_ID = "/home/ubuntu/sdxl-base"
COMPILE_DIR = "/home/ubuntu/sdxl/compile_dir_1024"
OUT_DIR = "/home/ubuntu/work_jim_tile4k"
DTYPE = torch.bfloat16
PROMPT = "An astronaut riding a green horse"
SEED = 42
TILE = int(os.environ.get("TILE", "1024"))
OVERLAP = int(os.environ.get("OVERLAP", "256"))
TARGET = int(os.environ.get("TARGET_RES", "4096"))
STRENGTH = float(os.environ.get("STRENGTH", "0.3"))
N_STEPS = int(os.environ.get("STEPS", "30"))     # base steps
REFINE_STEPS = max(2, int(round(STRENGTH * N_STEPS)))


# ---- attention patch (matches trace) -----------------------------------

def custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale


def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        probs = scores.softmax(dim=1).permute(0, 2, 1)
    else:
        scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        probs = scores.softmax(dim=-1)
    return probs


Attention.get_attention_scores = get_attention_scores_neuron


# ---- wrappers (match trace_sdxl_res.py) --------------------------------

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out = self.unet(
            sample, timestep, encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None,
                **kwargs):
        # NEFF was traced with timestep as 0-d scalar; reshape unconditionally.
        if torch.is_tensor(timestep):
            ts = timestep.float().reshape(())
        else:
            ts = torch.tensor(float(timestep))
        sample = self.unetwrap(
            sample,
            ts,
            encoder_hidden_states,
            added_cond_kwargs["text_embeds"],
            added_cond_kwargs["time_ids"],
        )[0]
        return UNet2DConditionOutput(sample=sample)


class TraceableTextEncoder(nn.Module):
    def __init__(self, te, kind):
        super().__init__()
        self.text_encoder = te
        self.kind = kind

    def forward(self, ids):
        out = self.text_encoder(ids, output_hidden_states=True)
        if self.kind == "clip_g":
            return (out[0], out[1], *out[2])
        return (out[1], out[0], *out[2])


class NeuronTextEncoder(nn.Module):
    def __init__(self, traced, original, kind):
        super().__init__()
        self.traced = traced
        self.config = original.config
        self.dtype = original.dtype
        self.device = original.device
        self.kind = kind  # "clip_l" or "clip_g"

    def forward(self, ids, output_hidden_states=True, **kwargs):
        out = self.traced(ids)
        # diffusers SDXL encode_prompt sets pooled_prompt_embeds = prompt_embeds[0]
        # when prompt_embeds[0].ndim == 2. Only CLIP-G should provide pooled.
        # For CLIP-L, return last_hidden_state at index 0 (3D) so the check fails.
        if self.kind == "clip_l":
            text_embeds_field = out[1]  # 3D last_hidden_state
        else:
            text_embeds_field = out[0]  # 2D projected embed (1280-d)
        return CLIPTextModelOutput(
            text_embeds=text_embeds_field,
            last_hidden_state=out[1],
            hidden_states=out[2:],
        )


class NeuronVAEDecoder(nn.Module):
    """Replicates pipe.vae.decode behavior using traced decoder + post_quant_conv."""
    def __init__(self, decoder, post_quant_conv, vae_config):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.config = vae_config


def load_pipeline():
    print("[load] base pipeline (CPU bf16)")
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

    # Replace UNet
    use_dp = os.environ.get("USE_DP", "1") == "1"
    if use_dp:
        print("[load] traced UNet -> DataParallel(0,1) (splits batch=2 -> 1+1)")
        unet_traced = torch.jit.load(f"{COMPILE_DIR}/unet/model.pt")
        wrap = UNetWrap(pipe.unet)
        nu = NeuronUNet(wrap)
        nu.unetwrap = torch_neuronx.DataParallel(unet_traced, [0, 1], set_dynamic_batching=True)
        pipe.unet = nu
    else:
        print("[load] traced UNet -> single core")
        unet_traced = torch.jit.load(f"{COMPILE_DIR}/unet/model.pt")
        wrap = UNetWrap(pipe.unet)
        nu = NeuronUNet(wrap)
        nu.unetwrap = unet_traced
        pipe.unet = nu

    # Replace text encoders
    print("[load] traced text_encoder/_2 -> single core")
    te1 = torch.jit.load(f"{COMPILE_DIR}/text_encoder/model.pt")
    te2 = torch.jit.load(f"{COMPILE_DIR}/text_encoder_2/model.pt")
    pipe.text_encoder = NeuronTextEncoder(te1, pipe.text_encoder, "clip_l")
    pipe.text_encoder_2 = NeuronTextEncoder(te2, pipe.text_encoder_2, "clip_g")

    # Replace VAE decoder + post_quant_conv with traced versions (still on default core)
    print("[load] traced VAE decoder")
    vae_dec = torch.jit.load(f"{COMPILE_DIR}/vae_decoder/model.pt")
    vae_pqc = torch.jit.load(f"{COMPILE_DIR}/vae_post_quant_conv/model.pt")
    pipe.vae.decoder = vae_dec
    pipe.vae.post_quant_conv = vae_pqc

    return pipe


def make_gaussian_mask(h, w, sigma_frac=0.35):
    yy = np.linspace(-1, 1, h, dtype=np.float32)
    xx = np.linspace(-1, 1, w, dtype=np.float32)
    g_y = np.exp(-0.5 * (yy / sigma_frac) ** 2)
    g_x = np.exp(-0.5 * (xx / sigma_frac) ** 2)
    return np.outer(g_y, g_x)


def tile_grid(target, tile, overlap):
    """Return list of (y, x) top-left corners covering target."""
    stride = tile - overlap
    coords = []
    ys = list(range(0, target - tile + 1, stride))
    if ys[-1] + tile < target:
        ys.append(target - tile)
    xs = list(range(0, target - tile + 1, stride))
    if xs[-1] + tile < target:
        xs.append(target - tile)
    for y in ys:
        for x in xs:
            coords.append((y, x))
    return coords


def upscale_pil(pil_img, target):
    return pil_img.resize((target, target), Image.BICUBIC)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(SEED)

    pipe = load_pipeline()
    pipe.set_progress_bar_config(disable=True)

    # img2img pipeline reuses the same components (no new modules to load)
    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    img2img.set_progress_bar_config(disable=True)

    metrics = {"runs": [], "tile": TILE, "overlap": OVERLAP, "target": TARGET,
               "strength": STRENGTH, "n_steps": N_STEPS,
               "refine_steps_per_tile": REFINE_STEPS, "prompt": PROMPT, "seed": SEED}

    NUM_RUNS = int(os.environ.get("NUM_RUNS", "4"))  # 1 cold + 3 warm
    for run_idx in range(NUM_RUNS):
        run = {"idx": run_idx}
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        g = torch.Generator(device="cpu").manual_seed(SEED)

        t0 = time.time()
        # 1) txt2img base 1024
        ts = time.time()
        base_img = pipe(prompt=PROMPT, height=1024, width=1024,
                        num_inference_steps=N_STEPS, generator=g).images[0]
        run["txt2img_s"] = time.time() - ts
        if run_idx == 0:
            base_img.save(f"{OUT_DIR}/base_1024.png")

        # 2) Upscale
        ts = time.time()
        big = upscale_pil(base_img, TARGET)
        run["upscale_s"] = time.time() - ts
        if run_idx == 0:
            big.save(f"{OUT_DIR}/upscaled_{TARGET}.png")

        # 3) Tile grid
        coords = tile_grid(TARGET, TILE, OVERLAP)
        run["num_tiles"] = len(coords)

        accum = np.zeros((TARGET, TARGET, 3), dtype=np.float32)
        weight = np.zeros((TARGET, TARGET, 1), dtype=np.float32)
        gmask = make_gaussian_mask(TILE, TILE)[..., None]  # (TILE, TILE, 1)

        # 4) per-tile img2img
        ts = time.time()
        per_tile = []
        big_arr = np.array(big)
        for (y, x) in coords:
            tile_pil = Image.fromarray(big_arr[y:y+TILE, x:x+TILE])
            tt0 = time.time()
            g_tile = torch.Generator(device="cpu").manual_seed(SEED + y * 13 + x)
            out = img2img(prompt=PROMPT, image=tile_pil,
                          strength=STRENGTH,
                          num_inference_steps=N_STEPS,
                          generator=g_tile).images[0]
            per_tile.append(time.time() - tt0)
            arr = np.array(out, dtype=np.float32)
            accum[y:y+TILE, x:x+TILE, :] += arr * gmask
            weight[y:y+TILE, x:x+TILE, :] += gmask

        run["tile_loop_s"] = time.time() - ts
        run["per_tile_s_mean"] = float(np.mean(per_tile))
        run["per_tile_s_min"] = float(np.min(per_tile))
        run["per_tile_s_max"] = float(np.max(per_tile))

        # 5) blend
        ts = time.time()
        blended = (accum / np.maximum(weight, 1e-6)).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(blended)
        run["blend_s"] = time.time() - ts

        run["total_s"] = time.time() - t0

        if run_idx == 0:
            out_img.save(f"{OUT_DIR}/final_{TARGET}_cold.png")
        else:
            out_img.save(f"{OUT_DIR}/final_{TARGET}_warm{run_idx}.png")

        print(f"[run {run_idx}] total={run['total_s']:.2f}s "
              f"txt2img={run['txt2img_s']:.2f}s upscale={run['upscale_s']:.2f}s "
              f"tiles={run['num_tiles']} tile_loop={run['tile_loop_s']:.2f}s "
              f"per_tile={run['per_tile_s_mean']:.2f}s blend={run['blend_s']:.2f}s")

        metrics["runs"].append(run)

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] metrics -> {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
