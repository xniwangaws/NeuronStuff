"""Phase R smoke test — full S3Diff pipeline with DeModLoRA modules.

Eager-mode bf16 on Neuron, same as phase_e1 but with peft LoRA replaced by
DeModLoRA-attr variants. This is the cheapest way to verify correctness AND
get a first latency data point before attempting torch_neuronx.trace.

Diff from phase_e1_smoke.py:
  - After loading S3Diff (and before .to(device, dtype)), call
    replace_lora_modules_in_unet(net_sr.unet) — swaps 257 peft LoraLayers
    for our DeModLoRA variants.
  - Do the same for VAE encoder (31 sites), but decoder has no LoRA.

If PSNR and latency match phase_e1_smoke bf16 baseline (8.60s / 43.26 dB),
the replacement is neutral and we can proceed to torch_neuronx.trace.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "neuron"], required=True)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--output_image", required=True)
    p.add_argument("--num_inferences", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    tdtype = dtype_map[args.dtype]
    print(f"[config] device={device} dtype={args.dtype}", flush=True)

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
    load_s3_s = time.perf_counter() - t0

    # --- R4 core: replace LoRA modules ---
    t_rep = time.perf_counter()
    from de_mod_lora_attr import replace_lora_modules_in_unet
    unet_replaced = replace_lora_modules_in_unet(net_sr.unet, adapter_name="default")
    vae_replaced = replace_lora_modules_in_unet(net_sr.vae, adapter_name="vae_skip")
    print(f"[R4] replaced UNet LoRA: {len(unet_replaced)} modules, "
          f"VAE LoRA: {len(vae_replaced)} modules ({time.perf_counter()-t_rep:.1f}s)", flush=True)

    # Move everything to device + dtype
    t1 = time.perf_counter()
    net_sr.to(device=device, dtype=tdtype)
    if hasattr(net_sr, "W"):
        net_sr.W.data = net_sr.W.data.to(dtype=tdtype)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                setattr(net_sr.sched, attr, t.to(device=device, dtype=tdtype))
        # Also move the scheduler's integer timesteps tensor (used in previous_timestep)
        if hasattr(net_sr.sched, "timesteps") and isinstance(net_sr.sched.timesteps, torch.Tensor):
            net_sr.sched.timesteps = net_sr.sched.timesteps.to(device)
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)
    # input_ids stays on text_encoder.device (repo s3diff_tile.py already patched)

    # vaehook.py fix: re-wrap with to_gpu=False (carried over from phase_e1_smoke.py)
    from my_utils import devices as _d
    _d.device = device
    _d.get_optimal_device = lambda: device
    _d.get_optimal_device_name = lambda: device.type
    from my_utils.vaehook import VAEHook
    net_sr.vae.encoder.forward = VAEHook(
        net_sr.vae.encoder, s3args.vae_encoder_tiled_size,
        is_decoder=False, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False,
    )
    net_sr.vae.decoder.forward = VAEHook(
        net_sr.vae.decoder, s3args.vae_decoder_tiled_size,
        is_decoder=True, fast_decoder=False, fast_encoder=False,
        color_fix=False, to_gpu=False,
    )

    to_device_s = time.perf_counter() - t1
    total_load = time.perf_counter() - t0
    print(f"[load] S3Diff init={load_s3_s:.1f}s  replace+to({device})={total_load - load_s3_s:.1f}s  total={total_load:.1f}s", flush=True)

    print("[load] DEResNet...", flush=True)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.to(device=device, dtype=tdtype).eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)
    print(f"[input] shape {tuple(im_lr.shape)} device {im_lr.device}", flush=True)

    # Need to upcast x_denoised before vae.decode (same as phase_e1_smoke).
    # Monkey-patch s3diff_tile.py's line 385 if needed — actually the repo is
    # already patched to do `.to(next(self.vae.parameters()).dtype)` before vae.decode.

    per_run_s = []
    sr_saved = None
    for i in range(args.num_inferences):
        torch.manual_seed(args.seed)
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
        print(f"[run {i+1}] {dt:.2f}s", flush=True)

        if i == 0 and args.output_image:
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
    print(f"=== Phase R RESULT ({args.device}/{args.dtype}) ===")
    print(f"load_time_s:      {total_load:.2f}")
    print(f"first_inference_s: {per_run_s[0]:.2f}")
    if len(per_run_s) > 1:
        warms = per_run_s[1:]
        print(f"warm_mean_s:      {sum(warms)/len(warms):.2f}  (runs {warms})")
    print(f"sr_image:         {sr_saved}")


if __name__ == "__main__":
    main()
