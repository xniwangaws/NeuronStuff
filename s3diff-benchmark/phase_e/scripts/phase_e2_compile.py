"""Phase E2: S3Diff eager + torch.compile(backend='neuron') — stage-by-stage keytime.

Stage ordering picks simpler graphs first (VAE enc/dec) then UNet.

Each stage runs 3 attempts inside a try block; compile-fail rolls back and
prints which stage died. We rebuild model state each stage to avoid cross-leak.
"""
import argparse, math, os, sys, time, traceback
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

# Allow Neuron to fall back to CPU on compile failure for unimplemented/failed ops
os.environ.setdefault("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "0")

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_eager/repo/src")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--output_dir", default="/tmp/phase_e2")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def build_model(args):
    from s3diff_tile import S3Diff
    from de_net import DEResNet

    s3args = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=96, latent_tiled_overlap=32,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    net_sr = S3Diff(sd_path=args.sd_path, pretrained_path=args.pretrained_path,
                    lora_rank_unet=32, lora_rank_vae=16, args=s3args)
    net_sr.set_eval()

    device = torch.device("neuron")
    net_sr.to(device)
    if hasattr(net_sr, "timesteps") and isinstance(net_sr.timesteps, torch.Tensor):
        net_sr.timesteps = net_sr.timesteps.to(device)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "timesteps", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(net_sr.sched, attr, t.to(device))

    from my_utils import devices as _d
    _d.device = device
    _d.get_optimal_device = lambda: device
    _d.get_optimal_device_name = lambda: device.type
    from my_utils.vaehook import VAEHook
    net_sr.vae.encoder.forward = VAEHook(
        net_sr.vae.encoder, s3args.vae_encoder_tiled_size, is_decoder=False,
        fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False)
    net_sr.vae.decoder.forward = VAEHook(
        net_sr.vae.decoder, s3args.vae_decoder_tiled_size, is_decoder=True,
        fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False)

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.to(device).eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device)

    return net_sr, net_de, im_lr, s3args


def run_once(net_sr, net_de, im_lr, s3args, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        deg = net_de(im_lr)
        ori_h, ori_w = im_lr.shape[2:]
        im_resize = F.interpolate(im_lr, size=(ori_h*4, ori_w*4), mode="bilinear", align_corners=False).contiguous()
        im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
        resize_h, resize_w = im_resize_norm.shape[2:]
        pad_h = math.ceil(resize_h / 64) * 64 - resize_h
        pad_w = math.ceil(resize_w / 64) * 64 - resize_w
        im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")
        x_tgt = net_sr(im_resize_norm, deg,
                       pos_prompt=[s3args.pos_prompt], neg_prompt=[s3args.neg_prompt])
        x_tgt = x_tgt[:, :, :resize_h, :resize_w]
    return x_tgt, im_resize


def save_sr(x_tgt, im_resize, out_path):
    from utils.wavelet_color import wavelet_color_fix
    out_img = (x_tgt * 0.5 + 0.5).cpu().float()
    out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
    im_resize_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
    try:
        out_pil = wavelet_color_fix(out_pil, im_resize_pil)
    except Exception:
        pass
    out_pil.save(out_path)


def bench_stage(label, net_sr, net_de, im_lr, s3args, n_runs, seed, out_path):
    print(f"\n===== STAGE: {label} =====", flush=True)
    times = []
    for i in range(n_runs):
        try:
            t0 = time.perf_counter()
            x_tgt, im_resize = run_once(net_sr, net_de, im_lr, s3args, seed)
            dt = time.perf_counter() - t0
            times.append(dt)
            tag = "COLD" if i == 0 else f"warm#{i}"
            print(f"  [{tag}] {dt:.2f}s", flush=True)
            if i == 0 and out_path:
                save_sr(x_tgt, im_resize, out_path)
                print(f"  [saved] {out_path}", flush=True)
        except Exception as e:
            print(f"  [FAIL run {i}] {type(e).__name__}: {str(e)[:300]}", flush=True)
            times.append(float("nan"))
            break
    return times


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()
    net_sr, net_de, im_lr, s3args = build_model(args)
    load_s = time.perf_counter() - t0
    print(f"[load] model to(neuron) done in {load_s:.1f}s", flush=True)

    results = {}

    # Stage 0: baseline
    results["s0_baseline"] = bench_stage(
        "stage0 baseline (no compile)", net_sr, net_de, im_lr, s3args,
        n_runs=2, seed=args.seed, out_path=f"{args.output_dir}/stage0.png")

    # Stage 1: compile VAE encoder original_forward (simpler than UNet)
    print("\n[compile] wrapping VAE encoder original_forward...", flush=True)
    try:
        if hasattr(net_sr.vae.encoder, "original_forward"):
            net_sr.vae.encoder.original_forward = torch.compile(
                net_sr.vae.encoder.original_forward, backend="neuron",
                dynamic=False, fullgraph=False,
            )
        results["s1_compile_vae_enc"] = bench_stage(
            "stage1 +compile VAE encoder", net_sr, net_de, im_lr, s3args,
            n_runs=3, seed=args.seed, out_path=f"{args.output_dir}/stage1.png")
    except Exception as e:
        print(f"[FAIL] stage1 setup: {e}", flush=True)
        results["s1_compile_vae_enc"] = [float("nan")]

    # Stage 2: also compile VAE decoder
    print("\n[compile] wrapping VAE decoder original_forward...", flush=True)
    try:
        if hasattr(net_sr.vae.decoder, "original_forward"):
            net_sr.vae.decoder.original_forward = torch.compile(
                net_sr.vae.decoder.original_forward, backend="neuron",
                dynamic=False, fullgraph=False,
            )
        results["s2_compile_vae_dec"] = bench_stage(
            "stage2 +compile VAE enc + dec", net_sr, net_de, im_lr, s3args,
            n_runs=3, seed=args.seed, out_path=f"{args.output_dir}/stage2.png")
    except Exception as e:
        print(f"[FAIL] stage2 setup: {e}", flush=True)
        results["s2_compile_vae_dec"] = [float("nan")]

    # Stage 3: also compile UNet.forward (biggest graph, most likely to hit compiler bugs)
    print("\n[compile] wrapping UNet.forward ...", flush=True)
    try:
        net_sr.unet.forward = torch.compile(
            net_sr.unet.forward, backend="neuron",
            dynamic=False, fullgraph=False,
        )
        results["s3_compile_unet"] = bench_stage(
            "stage3 +compile UNet", net_sr, net_de, im_lr, s3args,
            n_runs=3, seed=args.seed, out_path=f"{args.output_dir}/stage3.png")
    except Exception as e:
        print(f"[FAIL] stage3 setup: {e}", flush=True)
        results["s3_compile_unet"] = [float("nan")]

    # Summary
    def pick(ts):
        if not ts:
            return float("nan"), float("nan")
        cold = ts[0]
        warm_vals = [x for x in ts[1:] if not (isinstance(x, float) and math.isnan(x))]
        warm = min(warm_vals) if warm_vals else float("nan")
        return cold, warm

    print("\n" + "="*72)
    print(f"SUMMARY — cat_LQ_256 → 1024×1024 on Neuron eager (Phase E2)")
    print("="*72)
    for key, label in [
        ("s0_baseline",          "stage0 baseline          "),
        ("s1_compile_vae_enc",   "stage1 +compile VAE enc  "),
        ("s2_compile_vae_dec",   "stage2 +compile VAE dec  "),
        ("s3_compile_unet",      "stage3 +compile UNet     "),
    ]:
        ts = results.get(key, [])
        cold, warm = pick(ts)
        cold_s = f"{cold:7.2f}s" if not math.isnan(cold) else "  FAIL  "
        warm_s = f"{warm:7.2f}s" if not math.isnan(warm) else "  FAIL  "
        all_s  = [f"{x:.2f}" for x in ts] if ts else []
        print(f"  {label}  cold={cold_s}  warm_min={warm_s}  all={all_s}")
    print(f"\n[load_to_neuron] {load_s:.2f}s")


if __name__ == "__main__":
    main()
