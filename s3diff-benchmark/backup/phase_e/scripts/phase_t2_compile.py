"""Phase Task-2: Uniform-pad tiles + torch.compile VAE decoder inner block.

Strategy:
- Padded vaehook.py (applied in repo copy) guarantees ALL tiles feed the task
  queue at identical shape (tile_size + 2*pad, same). This should collapse the
  3-shape NEFF families observed in baseline profiling.
- Then try torch.compile(backend='neuron') on the decoder up_blocks[0..3] and
  mid_block modules. The task queue in vaehook.py already calls these modules'
  submodules (resnets[j], upsamplers[0]) directly, so compiling the submodules
  transparently intercepts the path.

Measurements: cold / warm1 / warm2 latencies; PSNR vs CPU fp32 reference.
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

os.environ.setdefault("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "0")

sys.path.insert(0, "/home/ubuntu/workspace/s3diff_task2/repo")
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_task2/repo/src")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--output_dir", default="/tmp/phase_t2")
    p.add_argument("--cpu_ref", default="/tmp/cpu_sr.png")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mode", choices=["pad_only", "pad_compile_resnets",
                                       "pad_compile_submod", "pad_compile_upblock"],
                   default="pad_only",
                   help="pad_only: no compile (verify correctness). "
                        "pad_compile_resnets: compile each resnet+upsampler inside decoder. "
                        "pad_compile_submod: compile decoder up_blocks[i].forward. "
                        "pad_compile_upblock: compile decoder.forward entirely.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
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

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    tdtype = dtype_map[args.dtype]
    device = torch.device("neuron")
    net_sr.to(device=device, dtype=tdtype)
    if hasattr(net_sr, "W"):
        net_sr.W.data = net_sr.W.data.to(dtype=tdtype)
    if hasattr(net_sr, "sched"):
        for attr in ["alphas_cumprod", "betas", "alphas", "sigmas"]:
            t = getattr(net_sr.sched, attr, None)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                setattr(net_sr.sched, attr, t.to(device=device, dtype=tdtype))
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
    net_de.to(device=device, dtype=tdtype).eval()

    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)

    return net_sr, net_de, im_lr, s3args


def apply_compile(net_sr, mode):
    if mode == "pad_only":
        return 0
    dec = net_sr.vae.decoder
    count = 0
    if mode == "pad_compile_resnets":
        # compile each resnet and upsampler inside decoder: smallest granularity.
        for blk in [dec.mid_block, *dec.up_blocks]:
            if hasattr(blk, "resnets"):
                for i, r in enumerate(blk.resnets):
                    r.forward = torch.compile(
                        r.forward, backend="neuron", dynamic=False, fullgraph=False)
                    count += 1
            if hasattr(blk, "upsamplers") and blk.upsamplers is not None:
                for u in blk.upsamplers:
                    u.forward = torch.compile(
                        u.forward, backend="neuron", dynamic=False, fullgraph=False)
                    count += 1
        # also conv_in / conv_out
        dec.conv_in.forward = torch.compile(
            dec.conv_in.forward, backend="neuron", dynamic=False, fullgraph=False)
        count += 1
        dec.conv_out.forward = torch.compile(
            dec.conv_out.forward, backend="neuron", dynamic=False, fullgraph=False)
        count += 1
    elif mode == "pad_compile_submod":
        # compile each up_block entirely — but task queue calls submodules inside,
        # so wrapping .forward won't be invoked. Still, instrumenting for completeness.
        for b in dec.up_blocks:
            b.forward = torch.compile(b.forward, backend="neuron",
                                      dynamic=False, fullgraph=False)
            count += 1
    elif mode == "pad_compile_upblock":
        dec.original_forward = torch.compile(
            dec.original_forward, backend="neuron",
            dynamic=False, fullgraph=False)
        count += 1
    return count


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


def psnr_vs(ref_path, out_path):
    try:
        a = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(Image.open(out_path).convert("RGB"), dtype=np.float32) / 255.0
        if a.shape != b.shape:
            return float("nan"), f"shape mismatch {a.shape} vs {b.shape}"
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return float("inf"), ""
        return 10.0 * np.log10(1.0 / mse), ""
    except Exception as e:
        return float("nan"), str(e)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()
    net_sr, net_de, im_lr, s3args = build_model(args)
    load_s = time.perf_counter() - t0
    print(f"[load] model to(neuron) done in {load_s:.1f}s", flush=True)

    n_compiled = apply_compile(net_sr, args.mode)
    print(f"[compile] mode={args.mode}  wrapped={n_compiled} module forward(s)", flush=True)

    n_runs = 3
    results = []
    for i in range(n_runs):
        try:
            t1 = time.perf_counter()
            x_tgt, im_resize = run_once(net_sr, net_de, im_lr, s3args, args.seed)
            dt = time.perf_counter() - t1
            tag = "COLD" if i == 0 else f"warm#{i}"
            print(f"[run {i+1} {tag}] {dt:.2f}s", flush=True)
            results.append(dt)
            out_path = f"{args.output_dir}/{args.mode}_{i}.png"
            save_sr(x_tgt, im_resize, out_path)
            if i == 0:
                psnr, err = psnr_vs(args.cpu_ref, out_path)
                print(f"[psnr] run{i+1} vs CPU ref: {psnr:.3f} dB  {err}", flush=True)
        except Exception as e:
            print(f"[FAIL run {i+1}] {type(e).__name__}: {str(e)[:500]}", flush=True)
            traceback.print_exc()
            results.append(float("nan"))
            break

    # Final summary
    print("\n" + "="*60)
    print(f"RESULT mode={args.mode} dtype={args.dtype}")
    print("="*60)
    for i, t in enumerate(results):
        tag = "cold" if i == 0 else f"warm{i}"
        s = f"{t:.2f}s" if not math.isnan(t) else "FAIL"
        print(f"  {tag}: {s}")
    # last PSNR pass: compare last successful run to CPU ref
    for i in reversed(range(len(results))):
        if not math.isnan(results[i]):
            out_path = f"{args.output_dir}/{args.mode}_{i}.png"
            psnr, err = psnr_vs(args.cpu_ref, out_path)
            print(f"  psnr_final(vs_cpu_ref) : {psnr:.3f} dB {err}")
            break


if __name__ == "__main__":
    main()
