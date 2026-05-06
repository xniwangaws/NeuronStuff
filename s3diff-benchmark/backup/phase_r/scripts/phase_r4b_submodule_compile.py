"""Phase R4b: compile UNet sub-components (down_blocks, mid_block, up_blocks) separately.

Hypothesis: full UNet compile triggers `vector::reserve` overflow in the compiler
due to graph size (257 LoRA sites * 3 einsum args per site). Splitting into
smaller subgraphs keeps each compile-unit bounded.

Sub-modules to compile:
  - unet.conv_in                                   (single DeModLoRAConv2d)
  - unet.time_embedding                            (small MLP, no LoRA)
  - unet.down_blocks[i]  for i in 0,1,2,3          (4 compiles)
  - unet.mid_block                                 (1 compile)
  - unet.up_blocks[i]    for i in 0,1,2,3          (4 compiles)
  - unet.conv_out                                  (single DeModLoRAConv2d)

Total: 11 compile units. Each should fit in the compiler budget comfortably.
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
sys.path.insert(0, "/home/ubuntu/workspace/s3diff_nxdi/modules")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lq_image", default="/home/ubuntu/s3diff/smoke_in/cat_LQ_256.png")
    p.add_argument("--output_image", default="/tmp/phase_r4b_cat.png")
    p.add_argument("--num_inferences", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--scope", choices=["all", "down", "mid", "up", "convs", "none"], default="all",
                   help="which submodules to compile")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("neuron")
    tdtype = torch.bfloat16

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    from utils.wavelet_color import wavelet_color_fix
    from de_mod_lora_attr import replace_lora_modules_in_unet

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
    print(f"[R4b] replaced UNet LoRA: {len(unet_replaced)}, VAE LoRA: {len(vae_replaced)}", flush=True)

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

    # R4b: compile sub-modules
    compile_opts = dict(backend="neuron", dynamic=False, fullgraph=False)
    compiled = []
    if args.scope in ("all", "convs"):
        net_sr.unet.conv_in.forward = torch.compile(net_sr.unet.conv_in.forward, **compile_opts)
        compiled.append("conv_in")
        net_sr.unet.conv_out.forward = torch.compile(net_sr.unet.conv_out.forward, **compile_opts)
        compiled.append("conv_out")
    if args.scope in ("all", "down"):
        for i, blk in enumerate(net_sr.unet.down_blocks):
            blk.forward = torch.compile(blk.forward, **compile_opts)
            compiled.append(f"down_blocks[{i}]")
    if args.scope in ("all", "mid"):
        net_sr.unet.mid_block.forward = torch.compile(net_sr.unet.mid_block.forward, **compile_opts)
        compiled.append("mid_block")
    if args.scope in ("all", "up"):
        for i, blk in enumerate(net_sr.unet.up_blocks):
            blk.forward = torch.compile(blk.forward, **compile_opts)
            compiled.append(f"up_blocks[{i}]")
    print(f"[compile] wrapped {len(compiled)} submodules: {compiled}", flush=True)

    # DE net + input
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model("/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    net_de = net_de.to(device=device, dtype=tdtype).eval()
    im = Image.open(args.lq_image).convert("RGB").resize((256, 256), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0).to(device=device, dtype=tdtype)

    per_run_s = []
    sr_saved = None
    for i in range(args.num_inferences):
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        try:
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
        except Exception as e:
            print(f"[run {i+1} FAIL] {type(e).__name__}: {str(e)[:400]}", flush=True)
            per_run_s.append(float("nan"))
            break

    print()
    print(f"=== Phase R4b scope={args.scope} RESULT ===")
    for i, d in enumerate(per_run_s):
        print(f"  run {i+1}: {d}")
    print(f"sr_image: {sr_saved}")


if __name__ == "__main__":
    main()
