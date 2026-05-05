"""Phase E1: S3Diff eager smoke on Neuron.

Load all components on Neuron eager, run single inference on cat_LQ_256,
compare PSNR vs CPU eager reference.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Suppress dynamo errors for torch.compile fallback (not used in E1 but harmless)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "neuron"], required=True)
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--output_image", required=True)
    p.add_argument("--num_inferences", type=int, default=2,
                   help="1st = warmup/first forward, 2nd = steady")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[config] device={device}", flush=True)

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    from utils.wavelet_color import wavelet_color_fix

    print("[load] S3Diff...", flush=True)
    t0 = time.perf_counter()
    s3args = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    net_sr = S3Diff(
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
        lora_rank_unet=32, lora_rank_vae=16, args=s3args,
    )
    net_sr.set_eval()

    # Pick dtype
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    tdtype = dtype_map[args.dtype]
    print(f"[config] dtype={args.dtype} ({tdtype})", flush=True)

    # Move to target device — use nn.Module.to(device, dtype).
    t1 = time.perf_counter()
    net_sr.to(device=device, dtype=tdtype)
    # Keep timesteps/input_ids as int — only cast floating tensors
    if hasattr(net_sr, "W"):
        net_sr.W.data = net_sr.W.data.to(dtype=tdtype)
    # Scheduler float tensors — cast to matching dtype so sched.step output stays bf16
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                setattr(net_sr.sched, attr, t.to(device=device, dtype=tdtype))
    # timesteps keeps long
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)
    # Stray tensor attrs that are not registered as Parameter/buffer
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "timesteps", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(net_sr.sched, attr, t.to(device))
    # Sanity: find tensors NOT on target device type
    target_type = device.type  # "neuron" or "cpu"
    stray = []
    for n, p in net_sr.named_parameters():
        if p.device.type != target_type:
            stray.append(f"param {n}: {p.device}")
    for n, b in net_sr.named_buffers():
        if b.device.type != target_type:
            stray.append(f"buffer {n}: {b.device}")
    print(f"[check] stray tensors (wrong device): {len(stray)}", flush=True)
    if stray:
        print(f"[check] first 10 stray: {stray[:10]}", flush=True)
    # Fix VAEHook moving the encoder back to the wrong device.
    # vaehook.py does self.net.to(devices.get_optimal_device()) when to_gpu=True,
    # and the devices module hardcodes cpu when cuda is not available.
    # Either re-wrap with to_gpu=False, or patch the device module.
    from my_utils import devices as _d
    _d.device = device
    _d.get_optimal_device = lambda: device
    _d.get_optimal_device_name = lambda: device.type
    # Re-wrap VAEHook with to_gpu=False so it doesn't try to move submodules
    from my_utils.vaehook import VAEHook
    net_sr.vae.encoder.forward = VAEHook(
        net_sr.vae.encoder, s3args.vae_encoder_tiled_size,
        is_decoder=False, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False)
    net_sr.vae.decoder.forward = VAEHook(
        net_sr.vae.decoder, s3args.vae_decoder_tiled_size,
        is_decoder=True, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False)
    print(f"[patch] VAEHook re-wrapped with to_gpu=False", flush=True)
    load_s = time.perf_counter() - t0
    to_device_s = time.perf_counter() - t1
    print(f"[load] S3Diff init={t1-t0:.1f}s  to({device})={to_device_s:.1f}s  total={load_s:.1f}s", flush=True)

    print("[load] DEResNet...", flush=True)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.to(device=device, dtype=tdtype).eval()

    # Load + prepare LQ — cast to selected dtype
    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)
    print(f"[input] shape {tuple(im_lr.shape)} device {im_lr.device}", flush=True)

    # Run inferences
    per_run_s = []
    sr_saved = None
    for i in range(args.num_inferences):
        torch.manual_seed(args.seed)  # keep seed fixed across runs
        t0 = time.perf_counter()
        with torch.no_grad():
            deg = net_de(im_lr)
            ori_h, ori_w = im_lr.shape[2:]
            im_resize = F.interpolate(im_lr, size=(ori_h*4, ori_w*4),
                                      mode="bilinear", align_corners=False).contiguous()
            im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
            resize_h, resize_w = im_resize_norm.shape[2:]
            pad_h = math.ceil(resize_h / 64) * 64 - resize_h
            pad_w = math.ceil(resize_w / 64) * 64 - resize_w
            im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")

            # S3Diff forward: it does everything internally (tokenize, text encode, VAE encode,
            # de_mod compute + inject, UNet, scheduler, VAE decode).
            x_tgt = net_sr(im_resize_norm, deg,
                           pos_prompt=[s3args.pos_prompt],
                           neg_prompt=[s3args.neg_prompt])
            x_tgt = x_tgt[:, :, :resize_h, :resize_w]
        dt = time.perf_counter() - t0
        per_run_s.append(dt)
        print(f"[run {i+1}] {dt:.2f}s", flush=True)

        if i == 0:  # save first run's output
            out_img = (x_tgt * 0.5 + 0.5).cpu().float()
            out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
            im_resize_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
            try:
                out_pil = wavelet_color_fix(out_pil, im_resize_pil)
            except Exception as e:
                print(f"[warn] wavelet_color_fix failed: {e}", flush=True)
            out_pil.save(args.output_image)
            sr_saved = args.output_image
            print(f"[saved] {sr_saved}", flush=True)

    print()
    print(f"=== RESULT ({args.device}) ===")
    print(f"load_time_s: {load_s:.2f}")
    print(f"first_inference_s: {per_run_s[0]:.2f}")
    if len(per_run_s) > 1:
        print(f"second_inference_s: {per_run_s[1]:.2f}")
    print(f"sr_image: {sr_saved}")


if __name__ == "__main__":
    main()
