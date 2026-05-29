"""Validate TP=4 SDXL UNet numerically against CPU bf16 reference."""
import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/work_unet_tp4")

os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--latent_h", type=int, default=16)
    ap.add_argument("--latent_w", type=int, default=16)
    ap.add_argument("--workdir", default="/home/ubuntu/work_unet_tp4/compile_tp4_smoke")
    ap.add_argument("--unet_path", default="/home/ubuntu/sdxl-base/unet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="/home/ubuntu/work_unet_tp4/UNET_TP4_VALIDATION.json")
    args = ap.parse_args()
    tp = args.tp
    H, W = args.latent_h, args.latent_w

    print(f"=== UNet TP={tp} validation: latent={H}x{W} ===", flush=True)

    # ---- Deterministic inputs
    torch.manual_seed(args.seed)
    sample = torch.randn(1, 4, H, W, dtype=torch.bfloat16)
    timestep = torch.tensor([999], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.bfloat16)
    text_embeds = torch.randn(1, 1280, dtype=torch.bfloat16)
    time_ids = torch.tensor([[float(H*8), float(W*8), 0., 0., float(H*8), float(W*8)]], dtype=torch.bfloat16)
    example_inputs = (sample, timestep, encoder_hidden_states, text_embeds, time_ids)

    # ---- CPU reference
    print("Running CPU bf16 reference forward...", flush=True)
    from diffusers import UNet2DConditionModel
    cpu_unet = UNet2DConditionModel.from_pretrained(
        args.unet_path, variant="fp16", torch_dtype=torch.bfloat16
    )
    cpu_unet.eval()
    with torch.no_grad():
        t0 = time.time()
        ref = cpu_unet(
            sample=sample, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]
        t_cpu = time.time() - t0
    print(f"  CPU forward: {t_cpu:.1f}s, out_shape={tuple(ref.shape)}", flush=True)
    ref_cpu = ref.float().clone()
    master_sd = {k: v.detach().clone() for k, v in cpu_unet.state_dict().items()}
    del cpu_unet, ref
    gc.collect()

    # ---- Build Neuron TP=4 model and load
    print("Building Neuron TP=4 model...", flush=True)
    from neuronx_distributed.trace.parallel_context import NxDParallelState
    from neuronx_distributed.trace.model_builder_v2 import ModelBuilder
    from neuronx_distributed.trace.functions import shard_checkpoint

    with NxDParallelState(world_size=tp, tensor_model_parallel_size=tp):
        from neuron_unet import shard_unet_in_place, UNetTraceWrapper
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_path, variant="fp16", torch_dtype=torch.bfloat16
        )
        unet.eval()
        stats = shard_unet_in_place(unet, tp=tp, dtype=torch.bfloat16)
        wrapped = UNetTraceWrapper(unet)
        wrapped.eval()
        print(f"  sharded {stats['linear_replaced']}L, {stats['conv_replaced']}C", flush=True)

        builder = ModelBuilder(wrapped)
        # Use same example shape as compile
        example_zero = (
            torch.zeros(1, 4, H, W, dtype=torch.bfloat16),
            torch.tensor([999], dtype=torch.long),
            torch.zeros(1, 77, 2048, dtype=torch.bfloat16),
            torch.zeros(1, 1280, dtype=torch.bfloat16),
            torch.zeros(1, 6, dtype=torch.bfloat16),
        )
        print("  trace + compile (or load from cache)...", flush=True)
        t0 = time.time()
        builder.trace(args=example_zero, tag="unet_tp4")
        nxd_model = builder.compile(
            compiler_workdir=args.workdir,
            compiler_args="--model-type=unet-inference --lnc=2 -O1",
        )
        t_compile = time.time() - t0
        print(f"  compile: {t_compile/60:.2f} min (cache hit fast)", flush=True)

        # Shard + load weights
        prefixed = {f"unet.{k}": v for k, v in master_sd.items()}
        sharded = shard_checkpoint(prefixed, wrapped)
        per_rank_gb = sum(t.numel() * t.element_size() for t in sharded[0].values()) / 1e9
        print(f"  per-rank state: {per_rank_gb:.2f} GB", flush=True)
        nxd_model.set_weights(sharded)
        nxd_model.to_neuron()

        # Forward on real inputs
        with torch.no_grad():
            t0 = time.time()
            out = nxd_model(*example_inputs)
            t_cold = time.time() - t0
            if isinstance(out, (list, tuple)):
                out_t = out[0]
            elif isinstance(out, dict):
                out_t = list(out.values())[0]
            else:
                out_t = out
        print(f"  Neuron cold: {t_cold*1000:.1f}ms  out_shape={tuple(out_t.shape)}", flush=True)

        # Warm
        warm = []
        with torch.no_grad():
            for _ in range(5):
                t0 = time.time()
                _ = nxd_model(*example_inputs)
                warm.append(time.time() - t0)
        warm_avg = sum(warm) / len(warm) * 1000
        print(f"  Neuron warm avg: {warm_avg:.1f}ms", flush=True)

        # Numerical
        n = out_t.float().reshape(-1)
        c = ref_cpu.reshape(-1)
        cos = F.cosine_similarity(n, c, dim=0).item()
        max_abs = (n - c).abs().max().item()
        mean_abs = (n - c).abs().mean().item()
        ref_max = c.abs().max().item()
        rel = max_abs / max(ref_max, 1e-6)
        print(f"  cos={cos:.6f}  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  rel={rel:.4f}", flush=True)

        verdict = "PASS" if cos >= 0.99 else "FAIL"
        print(f"  VERDICT: {verdict} (cos>=0.99 required)", flush=True)

        report = {
            "tp": tp,
            "latent_shape": [1, 4, H, W],
            "compile_min": t_compile / 60,
            "cold_ms": t_cold * 1000,
            "warm_ms_avg": warm_avg,
            "warm_ms_list": [w * 1000 for w in warm],
            "out_shape": list(out_t.shape),
            "per_rank_state_gb": per_rank_gb,
            "cos": cos,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_max_abs": rel,
            "ref_abs_max": ref_max,
            "verdict": verdict,
            "linear_sharded": stats["linear_replaced"],
            "conv_sharded": stats["conv_replaced"],
        }
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
