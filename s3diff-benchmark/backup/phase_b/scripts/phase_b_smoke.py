"""Phase B smoke test — S3Diff pipeline with NeuronS3DiffUNet (full custom).

Compared to Phase R (which monkey-patches peft LoRA into DeModLoRA), this uses
the **full custom UNet assembly** — our own class tree mirroring diffusers'
block hierarchy, with DeModLoRA modules baked in.

Benefit: trace-ready structure. Once `torch_neuronx.trace` API is available
in the Neuron SDK, this UNet can be traced with de_mod as explicit input.

Run:
  python phase_b_smoke.py --device neuron --dtype bf16 --output_image /tmp/out.png
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
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
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
    tdtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
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
    net_sr = S3Diff(sd_path=s3args.sd_path, pretrained_path=s3args.pretrained_path,
                    lora_rank_unet=32, lora_rank_vae=16, args=s3args)
    net_sr.set_eval()

    # Build custom UNet + copy weights
    print("[build] NeuronS3DiffUNet + copy weights...", flush=True)
    from s3diff_unet import NeuronS3DiffUNet
    # Import copy_weights from the test module (relative path)
    sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/tests")
    from test_full_custom_unet import copy_weights
    custom_unet = NeuronS3DiffUNet(lora_rank=32, lora_scaling=0.25)
    custom_unet.eval()
    copy_weights(custom_unet, net_sr.unet)

    # Wrap: make our custom UNet callable from S3Diff's forward via adapter
    from custom_unet_adapter import wrap_custom_unet
    # Also replace VAE encoder LoRA (not UNet, that uses our custom now)
    from de_mod_lora_attr import replace_lora_modules_in_unet
    vae_replaced = replace_lora_modules_in_unet(net_sr.vae, adapter_name="vae_skip")
    print(f"[build] VAE LoRA replaced: {len(vae_replaced)}", flush=True)

    # Move ref_unet (for de_mod harvesting) + our custom UNet
    print(f"[move] to {device} {args.dtype}...", flush=True)
    net_sr.to(device=device, dtype=tdtype)
    custom_unet.to(device=device, dtype=tdtype)
    # Now wrap ref_unet's forward to redirect to custom_unet
    wrap_custom_unet(custom_unet, net_sr.unet)

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

    load_s = time.perf_counter() - t0
    print(f"[load] total {load_s:.1f}s", flush=True)

    print("[load] DEResNet...", flush=True)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.to(device=device, dtype=tdtype).eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)
    print(f"[input] shape {tuple(im_lr.shape)}", flush=True)

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

        if i == 0:
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
    print(f"=== Phase B RESULT ({args.device}/{args.dtype}) ===")
    print(f"load_time_s:      {load_s:.2f}")
    print(f"first_inference_s: {per_run_s[0]:.2f}")
    if len(per_run_s) > 1:
        warms = per_run_s[1:]
        print(f"warm_mean_s:      {sum(warms)/len(warms):.2f}")
    print(f"sr_image:         {sr_saved}")


if __name__ == "__main__":
    main()
