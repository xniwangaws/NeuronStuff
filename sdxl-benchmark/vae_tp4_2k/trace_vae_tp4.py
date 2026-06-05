"""Trace + compile + validate + benchmark SDXL VAE decoder at 2K TP=4.

Same single-process ModelBuilder pattern as unet_tp4/trace_unet_tp4_mb.py.
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/work")

# LNC=1 -> 8 logical cores per trn2.3xlarge chip. TP=4 fits.
os.environ.setdefault("NEURON_RT_NUM_CORES", "8")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--latent_h", type=int, default=256)
    ap.add_argument("--latent_w", type=int, default=256)
    ap.add_argument("--workdir", default="/home/ubuntu/work/compile_vae_tp4_2k")
    ap.add_argument("--vae_path", default="/home/ubuntu/sdxl-base")
    ap.add_argument("--out_dir", default="/home/ubuntu/work")
    args = ap.parse_args()
    tp = args.tp
    H, W = args.latent_h, args.latent_w
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== VAE decoder TP={tp} latent={H}x{W} (image {H*8}x{W*8}) ===", flush=True)

    # ---- Example input
    example_inputs = (torch.zeros(1, 4, H, W, dtype=torch.bfloat16),)

    # ---- Load full VAE on CPU once for master state dict
    from diffusers import AutoencoderKL

    print("Loading SDXL VAE on CPU bf16...", flush=True)
    t0 = time.time()
    vae_cpu = AutoencoderKL.from_pretrained(
        args.vae_path, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
    )
    vae_cpu.eval()
    # Master state dict from VAEDecoderTraceWrapper view (post_quant_conv + decoder)
    master_sd = {}
    for k, v in vae_cpu.post_quant_conv.state_dict().items():
        master_sd[f"post_quant_conv.{k}"] = v.detach().clone()
    for k, v in vae_cpu.decoder.state_dict().items():
        master_sd[f"decoder.{k}"] = v.detach().clone()
    print(
        f"  loaded in {time.time()-t0:.1f}s, decoder+pqc params = "
        f"{sum(v.numel()*v.element_size() for v in master_sd.values())/1e6:.1f} MB",
        flush=True,
    )

    # ---- Build sharded VAE structure under NxDParallelState
    from neuronx_distributed.trace.parallel_context import NxDParallelState
    from neuronx_distributed.trace.model_builder_v2 import ModelBuilder
    from neuronx_distributed.trace.functions import shard_checkpoint

    with NxDParallelState(world_size=tp, tensor_model_parallel_size=tp):
        from neuron_vae_decoder_tp4 import shard_vae_decoder_in_place, VAEDecoderTraceWrapper

        print("Constructing sharded VAE decoder structure...", flush=True)
        vae = AutoencoderKL.from_pretrained(
            args.vae_path, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
        )
        vae.eval()
        stats = shard_vae_decoder_in_place(vae, tp=tp, dtype=torch.bfloat16)
        print(
            f"  sharded {stats['linear_replaced']} Linears, "
            f"{stats['conv_replaced']} Convs",
            flush=True,
        )
        print(
            f"  skipped: lin_small={stats['linear_skipped_small']} "
            f"lin_dim={stats['linear_skipped_dim']} "
            f"conv_small={stats['conv_skipped_small']} "
            f"conv_grouped={stats['conv_skipped_grouped']} "
            f"conv_dim={stats['conv_skipped_dim']}",
            flush=True,
        )
        wrapped = VAEDecoderTraceWrapper(vae)
        wrapped.eval()

        sharded_state_keys = set(dict(wrapped.named_parameters()).keys())
        master_keys = set(master_sd.keys())
        missing = master_keys - sharded_state_keys
        extra = sharded_state_keys - master_keys
        print(f"  state-dict alignment: {len(missing)} missing, {len(extra)} extra", flush=True)
        if missing:
            print(f"    sample missing: {list(missing)[:5]}", flush=True)
        if extra:
            print(f"    sample extra: {list(extra)[:5]}", flush=True)

        # ---- Build & trace
        print("Building ModelBuilder + tracing...", flush=True)
        t0 = time.time()
        builder = ModelBuilder(wrapped)
        builder.trace(args=example_inputs, tag="vae_decoder_tp4_2k")
        t_trace = time.time() - t0
        print(f"  trace done in {t_trace:.1f}s", flush=True)

        # ---- Compile with Luo's flag
        print("Compiling NEFF (with --tiled-inst-limit 10000000)...", flush=True)
        t0 = time.time()
        compiler_args = (
            "--model-type=unet-inference "
            "--lnc=1 "
            "-O1 "
            "--internal-hlo2tensorizer-options='--tiled-inst-limit 10000000' "
            "--verbose=DEBUG"
        )
        nxd_model = builder.compile(
            compiler_workdir=args.workdir,
            compiler_args=compiler_args,
        )
        t_compile = time.time() - t0
        print(f"  compile done in {t_compile/60:.1f} min", flush=True)

        # ---- Shard master state dict
        print("Sharding master state dict across TP ranks...", flush=True)
        t0 = time.time()
        sharded = shard_checkpoint(master_sd, wrapped)
        per_rank_size_mb = sum(t.numel() * t.element_size() for t in sharded[0].values()) / 1e6
        print(
            f"  per-rank state dict: {per_rank_size_mb:.1f} MB (took {time.time()-t0:.1f}s)",
            flush=True,
        )

        # ---- Load weights onto Neuron
        print("Loading weights onto Neuron device...", flush=True)
        t0 = time.time()
        nxd_model.set_weights(sharded)
        nxd_model.to_neuron()
        print(f"  to_neuron done in {time.time()-t0:.1f}s", flush=True)

        # ---- 1 cold + 9 warm forward passes
        torch.manual_seed(0)
        latent_test = torch.randn(1, 4, H, W, dtype=torch.bfloat16)
        print("Running 1 cold + 9 warm forward passes...", flush=True)
        with torch.no_grad():
            t0 = time.time()
            out = nxd_model(latent_test)
            t_cold = time.time() - t0
            if isinstance(out, (list, tuple)):
                out_t = out[0]
            elif isinstance(out, dict):
                out_t = list(out.values())[0]
            else:
                out_t = out
            print(f"  cold: {t_cold*1000:.1f}ms  out_shape={tuple(out_t.shape)}", flush=True)

            warm = []
            for _ in range(9):
                t1 = time.time()
                _ = nxd_model(latent_test)
                warm.append(time.time() - t1)
            warm_avg = sum(warm) / len(warm)
            warm_min = min(warm)
            print(
                f"  warm avg: {warm_avg*1000:.1f}ms  min: {warm_min*1000:.1f}ms  "
                f"list={[f'{x*1000:.1f}' for x in warm]}",
                flush=True,
            )

        neuron_out = out_t.detach().to(torch.float32).cpu().clone()

    # ---- CPU reference outside parallel state
    print("\nRunning CPU bf16 reference forward...", flush=True)
    t0 = time.time()
    cpu_wrapper = type("W", (), {})()
    cpu_wrapper.post_quant_conv = vae_cpu.post_quant_conv
    cpu_wrapper.decoder = vae_cpu.decoder
    with torch.no_grad():
        z = vae_cpu.post_quant_conv(latent_test)
        cpu_out = vae_cpu.decoder(z)
    print(f"  CPU forward done in {time.time()-t0:.1f}s, shape={tuple(cpu_out.shape)}", flush=True)
    cpu_out_f = cpu_out.detach().to(torch.float32).cpu()

    cos = torch.nn.functional.cosine_similarity(
        neuron_out.flatten(), cpu_out_f.flatten(), dim=0
    ).item()
    diff = (neuron_out - cpu_out_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_abs_max = cpu_out_f.abs().max().item()
    print(
        f"\n=== Numerical validation ===\n"
        f"  cos_sim: {cos:.6f}\n"
        f"  max_abs: {max_abs:.4f}\n"
        f"  mean_abs: {mean_abs:.4f}\n"
        f"  ref_abs_max: {ref_abs_max:.4f}",
        flush=True,
    )

    verdict = "PASS" if cos > 0.99 else "FAIL"

    # ---- Save report
    report = {
        "tp": tp,
        "latent_shape": [1, 4, H, W],
        "image_shape": [1, 3, H * 8, W * 8],
        "trace_s": t_trace,
        "compile_min": t_compile / 60,
        "cold_ms": t_cold * 1000,
        "warm_ms_avg": warm_avg * 1000,
        "warm_ms_min": warm_min * 1000,
        "warm_ms_list": [w * 1000 for w in warm],
        "out_shape": list(out_t.shape),
        "per_rank_state_mb": per_rank_size_mb,
        "cos": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_abs_max": ref_abs_max,
        "verdict": verdict,
        "linear_sharded": stats["linear_replaced"],
        "conv_sharded": stats["conv_replaced"],
        "compiler_args": compiler_args,
    }
    out_path = os.path.join(args.out_dir, "vae_tp4_2k_bench.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
