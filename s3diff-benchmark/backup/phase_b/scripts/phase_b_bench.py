"""Phase B 5-image bench — S3Diff with full custom UNet + optional NKI attn1."""
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
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/tests")

LQ_DIR = "/home/ubuntu/workspace/s3diff_multi/lq"
CPU_REF_DIR = "/home/ubuntu/workspace/s3diff_multi/cpu_ref"
OUT_DIR = "/home/ubuntu/workspace/s3diff_nxdi/data/phase_b_outputs"
IMAGES = ["cat", "bus", "bird", "butterfly", "woman"]


def psnr_np(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def build_pipeline():
    import argparse as _ap
    s3args = _ap.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        sd_path="/home/ubuntu/s3diff/models/sd-turbo",
        pretrained_path="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl",
    )
    from s3diff_tile import S3Diff
    net_sr = S3Diff(sd_path=s3args.sd_path, pretrained_path=s3args.pretrained_path,
                    lora_rank_unet=32, lora_rank_vae=16, args=s3args)
    net_sr.set_eval()

    from s3diff_unet import NeuronS3DiffUNet
    from test_full_custom_unet import copy_weights
    from custom_unet_adapter import wrap_custom_unet
    from de_mod_lora_attr import replace_lora_modules_in_unet

    custom_unet = NeuronS3DiffUNet(lora_rank=32, lora_scaling=0.25)
    custom_unet.eval()
    copy_weights(custom_unet, net_sr.unet)
    replace_lora_modules_in_unet(net_sr.vae, adapter_name="vae_skip")

    device = torch.device("neuron")
    dtype = torch.bfloat16
    net_sr.to(device=device, dtype=dtype)
    custom_unet.to(device=device, dtype=dtype)
    wrap_custom_unet(custom_unet, net_sr.unet)

    if hasattr(net_sr, "W"):
        net_sr.W.data = net_sr.W.data.to(dtype=dtype)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                setattr(net_sr.sched, attr, t.to(device=device, dtype=dtype))
        if hasattr(net_sr.sched, "timesteps") and isinstance(net_sr.sched.timesteps, torch.Tensor):
            net_sr.sched.timesteps = net_sr.sched.timesteps.to(device)
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)

    from my_utils import devices as _d
    _d.device = device; _d.get_optimal_device = lambda: device
    _d.get_optimal_device_name = lambda: device.type
    from my_utils.vaehook import VAEHook
    net_sr.vae.encoder.forward = VAEHook(
        net_sr.vae.encoder, s3args.vae_encoder_tiled_size, is_decoder=False,
        fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False)
    net_sr.vae.decoder.forward = VAEHook(
        net_sr.vae.decoder, s3args.vae_decoder_tiled_size, is_decoder=True,
        fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False)

    from de_net import DEResNet
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model("/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    net_de = net_de.to(device=device, dtype=dtype).eval()
    return net_sr, net_de, s3args, device, dtype


def run_one_image(net_sr, net_de, s3args, device, dtype, lq_path, out_path, num_inf=3, seed=123):
    from utils.wavelet_color import wavelet_color_fix
    im = Image.open(lq_path).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=dtype)

    times = []
    saved = None
    for i in range(num_inf):
        torch.manual_seed(seed)
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
                           pos_prompt=[s3args.pos_prompt], neg_prompt=[s3args.neg_prompt])
            x_tgt = x_tgt[:, :, :resize_h, :resize_w]
        dt = time.perf_counter() - t0
        times.append(dt)
        if i == 0:
            out_img = (x_tgt * 0.5 + 0.5).cpu().float()
            out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
            im_resize_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
            try:
                out_pil = wavelet_color_fix(out_pil, im_resize_pil)
            except Exception:
                pass
            out_pil.save(out_path)
            saved = out_path
    return times, saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[build] Phase B pipeline...", flush=True)
    t0 = time.perf_counter()
    net_sr, net_de, s3args, device, dtype = build_pipeline()
    print(f"[build] {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for name in IMAGES:
        lq_path = f"{LQ_DIR}/{name}_LQ_256.png"
        ref_path = f"{CPU_REF_DIR}/{name}_cpu_fp32.png"
        out_path = f"{OUT_DIR}/{name}_phase_b.png"
        if not os.path.exists(lq_path) or not os.path.exists(ref_path):
            print(f"[skip] {name}: missing files"); continue
        print(f"\n[run] {name}...", flush=True)
        times, saved = run_one_image(net_sr, net_de, s3args, device, dtype, lq_path, out_path)
        ref = np.asarray(Image.open(ref_path).convert("RGB")).astype(np.float32)
        new = np.asarray(Image.open(saved).convert("RGB")).astype(np.float32)
        p = psnr_np(ref, new)
        cold = times[0]
        warm = sum(times[1:]) / len(times[1:]) if len(times) > 1 else float("nan")
        print(f"[result] {name}: cold={cold:.2f}s warm={warm:.2f}s PSNR={p:.2f} dB", flush=True)
        results.append({"name": name, "cold": cold, "warm": warm, "psnr": p})

    print("\n" + "="*70)
    print(f"{'image':<12} {'cold':>8} {'warm':>8} {'PSNR':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<12} {r['cold']:>7.2f}s {r['warm']:>7.2f}s {r['psnr']:>9.2f} dB")
    print("-"*70)
    if results:
        wm = [r["warm"] for r in results]
        ps = [r["psnr"] for r in results]
        print(f"{'mean':<12} {'-':>8} {np.mean(wm):>7.2f}s {np.mean(ps):>9.2f} dB")


if __name__ == "__main__":
    main()
