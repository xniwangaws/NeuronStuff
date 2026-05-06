"""Phase bisect — compile ONE sub-scope across all 16 transformer blocks,
run 1 inference on neuron, save image for offline PSNR.

Usage:
  python phase_bisect.py --scope attn1 --out /tmp/bisect_trial_1.png
  --scope {attn1, attn2, ff, proj, block}
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

os.environ["TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS"] = "0"
os.environ.setdefault("NEURON_CC_FLAGS", "--auto-cast=matmult")

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_bisect/modules")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["attn1", "attn2", "ff", "proj", "block", "t2d", "xblock", "none"], required=True)
    p.add_argument("--output_image", required=True)
    p.add_argument("--num_inferences", type=int, default=2)
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    return p.parse_args()


def find_all_transformer2d(unet):
    """Return list of (name, Transformer2DModel) found in unet.down_blocks, mid_block, up_blocks."""
    out = []
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    for name, m in unet.named_modules():
        if isinstance(m, Transformer2DModel):
            out.append((name, m))
    return out


def main():
    args = parse_args()
    torch.manual_seed(123)
    device = torch.device("neuron")
    tdtype = torch.bfloat16

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    from utils.wavelet_color import wavelet_color_fix
    from de_mod_lora_attr import replace_lora_modules_in_unet

    print(f"[bisect] scope={args.scope} out={args.output_image}", flush=True)
    print("[load] S3Diff...", flush=True)
    t0 = time.perf_counter()
    s3args = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        sd_path="/home/ubuntu/s3diff/models/sd-turbo",
        pretrained_path="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl",
    )
    net_sr = S3Diff(sd_path=s3args.sd_path, pretrained_path=s3args.pretrained_path,
                    lora_rank_unet=32, lora_rank_vae=16, args=s3args)
    net_sr.set_eval()
    unet_replaced = replace_lora_modules_in_unet(net_sr.unet, adapter_name="default")
    vae_replaced = replace_lora_modules_in_unet(net_sr.vae, adapter_name="vae_skip")
    print(f"[R] replaced UNet LoRA: {len(unet_replaced)}, VAE LoRA: {len(vae_replaced)}", flush=True)

    net_sr.to(device=device, dtype=tdtype)
    if hasattr(net_sr, "W"):
        net_sr.W.data = net_sr.W.data.to(dtype=tdtype)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                setattr(net_sr.sched, attr, t.to(device=device, dtype=tdtype))
        if hasattr(net_sr.sched, "timesteps") and isinstance(net_sr.sched.timesteps, torch.Tensor):
            net_sr.sched.timesteps = net_sr.sched.timesteps.to(device)
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)

    from my_utils import devices as _d
    _d.device = device
    _d.get_optimal_device = lambda: device
    _d.get_optimal_device_name = lambda: device.type
    from my_utils.vaehook import VAEHook
    net_sr.vae.encoder.forward = VAEHook(
        net_sr.vae.encoder, s3args.vae_encoder_tiled_size,
        is_decoder=False, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False)
    net_sr.vae.decoder.forward = VAEHook(
        net_sr.vae.decoder, s3args.vae_decoder_tiled_size,
        is_decoder=True, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False)
    print(f"[load] total {time.perf_counter() - t0:.1f}s", flush=True)

    # === compile scope ===
    compile_opts = dict(backend="neuron", dynamic=False, fullgraph=False)

    t2d_list = find_all_transformer2d(net_sr.unet)
    print(f"[bisect] found {len(t2d_list)} Transformer2DModel instances", flush=True)

    wrapped_count = 0
    wrapped_names = []
    if args.scope == "xblock":
        # Reproduce the R4 regression: wrap CrossAttnDownBlock2D / CrossAttnUpBlock2D.
        for i, blk in enumerate(net_sr.unet.down_blocks):
            if "CrossAttn" in type(blk).__name__:
                blk.forward = torch.compile(blk.forward, **compile_opts)
                wrapped_count += 1
                wrapped_names.append(f"down_blocks[{i}] ({type(blk).__name__})")
        if "CrossAttn" in type(net_sr.unet.mid_block).__name__:
            net_sr.unet.mid_block.forward = torch.compile(net_sr.unet.mid_block.forward, **compile_opts)
            wrapped_count += 1
            wrapped_names.append(f"mid_block ({type(net_sr.unet.mid_block).__name__})")
        for i, blk in enumerate(net_sr.unet.up_blocks):
            if "CrossAttn" in type(blk).__name__:
                blk.forward = torch.compile(blk.forward, **compile_opts)
                wrapped_count += 1
                wrapped_names.append(f"up_blocks[{i}] ({type(blk).__name__})")
    for (t2d_name, t2d) in t2d_list:
        # t2d.transformer_blocks is ModuleList of BasicTransformerBlock
        for bi, blk in enumerate(t2d.transformer_blocks):
            if args.scope == "attn1":
                blk.attn1.forward = torch.compile(blk.attn1.forward, **compile_opts)
                wrapped_count += 1
                wrapped_names.append(f"{t2d_name}.transformer_blocks[{bi}].attn1")
            elif args.scope == "attn2":
                if getattr(blk, "attn2", None) is not None:
                    blk.attn2.forward = torch.compile(blk.attn2.forward, **compile_opts)
                    wrapped_count += 1
                    wrapped_names.append(f"{t2d_name}.transformer_blocks[{bi}].attn2")
            elif args.scope == "ff":
                blk.ff.forward = torch.compile(blk.ff.forward, **compile_opts)
                wrapped_count += 1
                wrapped_names.append(f"{t2d_name}.transformer_blocks[{bi}].ff")
            elif args.scope == "block":
                blk.forward = torch.compile(blk.forward, **compile_opts)
                wrapped_count += 1
                wrapped_names.append(f"{t2d_name}.transformer_blocks[{bi}]")
        if args.scope == "proj":
            t2d.proj_in.forward = torch.compile(t2d.proj_in.forward, **compile_opts)
            t2d.proj_out.forward = torch.compile(t2d.proj_out.forward, **compile_opts)
            wrapped_count += 2
            wrapped_names.append(f"{t2d_name}.proj_in/out")
        if args.scope == "t2d":
            t2d.forward = torch.compile(t2d.forward, **compile_opts)
            wrapped_count += 1
            wrapped_names.append(f"{t2d_name} (Transformer2DModel)")
    print(f"[bisect] scope={args.scope} wrapped_count={wrapped_count}", flush=True)
    if len(wrapped_names) <= 6:
        for nm in wrapped_names:
            print(f"  - {nm}", flush=True)
    else:
        for nm in wrapped_names[:3]:
            print(f"  - {nm}", flush=True)
        print(f"  ... ({len(wrapped_names) - 3} more)", flush=True)

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model("/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    net_de = net_de.to(device=device, dtype=tdtype).eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)

    per_run_s = []
    sr_saved = None
    for i in range(args.num_inferences):
        torch.manual_seed(123)
        t0 = time.perf_counter()
        with torch.no_grad():
            deg = net_de(im_lr)
            ori_h, ori_w = im_lr.shape[2:]
            im_resize = F.interpolate(im_lr, size=(ori_h * 4, ori_w * 4),
                                      mode="bilinear", align_corners=False).contiguous()
            im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
            resize_h, resize_w = im_resize_norm.shape[2:]
            pad_h = math.ceil(resize_h / 64) * 64 - resize_h
            pad_w = math.ceil(resize_w / 64) * 64 - resize_w
            im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")
            x_tgt = net_sr(im_resize_norm, deg,
                           pos_prompt=[s3args.pos_prompt],
                           neg_prompt=[s3args.neg_prompt])
            x_tgt = x_tgt[:, :, :resize_h, :resize_w]
        dt = time.perf_counter() - t0
        per_run_s.append(dt)
        tag = "COLD" if i == 0 else f"warm#{i}"
        print(f"[run {i+1} {tag}] {dt:.2f}s", flush=True)
        if i == 0 and args.output_image:
            out_img = (x_tgt * 0.5 + 0.5).cpu().float()
            out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
            im_resize_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
            try:
                out_pil = wavelet_color_fix(out_pil, im_resize_pil)
            except Exception:
                pass
            out_pil.save(args.output_image)
            sr_saved = args.output_image
            print(f"[saved] {sr_saved}", flush=True)

    # PSNR vs the pure-eager Phase R reference
    ref_path = "/tmp/phase_r_cat.png"
    if os.path.exists(ref_path) and sr_saved and os.path.exists(sr_saved):
        ref = np.asarray(Image.open(ref_path).convert("RGB")).astype(np.float32)
        cur = np.asarray(Image.open(sr_saved).convert("RGB")).astype(np.float32)
        if ref.shape == cur.shape:
            mse = float(((ref - cur) ** 2).mean())
            if mse > 0:
                psnr = 20 * math.log10(255.0 / math.sqrt(mse))
            else:
                psnr = float("inf")
            print(f"[PSNR] vs {ref_path}: {psnr:.2f} dB (mse={mse:.3f})", flush=True)
        else:
            print(f"[PSNR] shape mismatch ref={ref.shape} cur={cur.shape}", flush=True)
    print()
    print(f"=== BISECT scope={args.scope} ===")
    for i, d in enumerate(per_run_s):
        print(f"  run {i+1}: {d:.2f}s")


if __name__ == "__main__":
    main()
