"""Trace SDXL UNet TP=4 -> single NEFF using ModelBuilder pattern (proven by VAE).

Same pattern as work_b/test_vae_2k_tp.py: single-process compile, ModelBuilder
+ NxDParallelState + shard_checkpoint.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/work_unet_tp4")

os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--latent_h", type=int, default=256)
    ap.add_argument("--latent_w", type=int, default=256)
    ap.add_argument("--workdir", default="/home/ubuntu/work_unet_tp4/compile_tp4")
    ap.add_argument("--unet_path", default="/home/ubuntu/sdxl-base/unet")
    ap.add_argument("--bench_only", action="store_true")
    args = ap.parse_args()
    tp = args.tp
    H, W = args.latent_h, args.latent_w
    os.makedirs(args.workdir, exist_ok=True)

    print(f"=== UNet TP={tp} latent={H}x{W} ===", flush=True)

    # ---- Example inputs
    sample = torch.zeros(1, 4, H, W, dtype=torch.bfloat16)
    timestep = torch.tensor([999], dtype=torch.long)
    encoder_hidden_states = torch.zeros(1, 77, 2048, dtype=torch.bfloat16)
    text_embeds = torch.zeros(1, 1280, dtype=torch.bfloat16)
    time_ids = torch.zeros(1, 6, dtype=torch.bfloat16)
    example_inputs = (sample, timestep, encoder_hidden_states, text_embeds, time_ids)

    # ---- Load UNet on CPU and remember state dict (master, unsharded)
    from diffusers import UNet2DConditionModel
    print("Loading SDXL UNet on CPU bf16...", flush=True)
    t0 = time.time()
    unet_cpu = UNet2DConditionModel.from_pretrained(
        args.unet_path, variant="fp16", torch_dtype=torch.bfloat16
    )
    unet_cpu.eval()
    master_sd = {k: v.detach().clone() for k, v in unet_cpu.state_dict().items()}
    del unet_cpu
    gc.collect()
    print(f"  loaded in {time.time()-t0:.1f}s, {sum(v.numel()*v.element_size() for v in master_sd.values())/1e9:.2f} GB", flush=True)

    # ---- Build sharded UNet inside NxDParallelState context
    from neuronx_distributed.trace.parallel_context import NxDParallelState
    from neuronx_distributed.trace.model_builder_v2 import ModelBuilder
    from neuronx_distributed.trace.functions import shard_checkpoint

    with NxDParallelState(world_size=tp, tensor_model_parallel_size=tp):
        from neuron_unet import shard_unet_in_place, UNetTraceWrapper

        print("Constructing sharded UNet structure...", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_path, variant="fp16", torch_dtype=torch.bfloat16
        )
        unet.eval()
        stats = shard_unet_in_place(unet, tp=tp, dtype=torch.bfloat16)
        print(f"  sharded {stats['linear_replaced']} Linears, {stats['conv_replaced']} Convs", flush=True)
        wrapped = UNetTraceWrapper(unet)
        wrapped.eval()

        # Verify forward on CPU (validates structure compiles)
        # NOTE: sharded forward needs distributed comms, skip — only verify the
        # state dict alignment.
        sharded_state_keys = set(dict(wrapped.named_parameters()).keys())
        master_keys_prefixed = {f"unet.{k}" for k in master_sd.keys()}
        missing = master_keys_prefixed - sharded_state_keys
        extra = sharded_state_keys - master_keys_prefixed
        print(f"  state-dict alignment: {len(missing)} missing, {len(extra)} extra", flush=True)

        # ---- Build & trace
        print("Building ModelBuilder + tracing...", flush=True)
        t0 = time.time()
        builder = ModelBuilder(wrapped)
        builder.trace(args=example_inputs, tag="unet_tp4")
        t_trace = time.time() - t0
        print(f"  trace done in {t_trace:.1f}s", flush=True)

        # ---- Compile
        print("Compiling NEFF...", flush=True)
        t0 = time.time()
        compiler_args = "--model-type=unet-inference --lnc=2 -O1"
        nxd_model = builder.compile(
            compiler_workdir=args.workdir,
            compiler_args=compiler_args,
        )
        t_compile = time.time() - t0
        print(f"  compile done in {t_compile/60:.1f} min", flush=True)

        # ---- Shard the master state dict
        print("Sharding master state dict across TP ranks...", flush=True)
        prefixed_master = {f"unet.{k}": v for k, v in master_sd.items()}
        t0 = time.time()
        sharded = shard_checkpoint(prefixed_master, wrapped)
        per_rank_size_gb = sum(t.numel() * t.element_size() for t in sharded[0].values()) / 1e9
        print(f"  per-rank state dict: {per_rank_size_gb:.2f} GB (took {time.time()-t0:.1f}s)", flush=True)

        # ---- Load weights onto Neuron
        print("Loading weights onto Neuron device...", flush=True)
        t0 = time.time()
        nxd_model.set_weights(sharded)
        nxd_model.to_neuron()
        print(f"  to_neuron done in {time.time()-t0:.1f}s", flush=True)

        # ---- Cold + warm latency
        print("Running 1 cold + 5 warm forward passes...", flush=True)
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
            print(f"  cold: {t_cold*1000:.1f}ms  out_shape={tuple(out_t.shape)}", flush=True)

            warm = []
            for _ in range(5):
                t0 = time.time()
                _ = nxd_model(*example_inputs)
                warm.append(time.time() - t0)
            print(f"  warm avg: {sum(warm)/len(warm)*1000:.1f}ms  list={[f'{x*1000:.1f}' for x in warm]}", flush=True)

        # ---- Save report
        report = {
            "tp": tp,
            "latent_shape": [1, 4, H, W],
            "trace_s": t_trace,
            "compile_min": t_compile / 60,
            "cold_ms": t_cold * 1000,
            "warm_ms_avg": sum(warm) / len(warm) * 1000,
            "warm_ms_list": [w * 1000 for w in warm],
            "out_shape": list(out_t.shape),
            "per_rank_state_gb": per_rank_size_gb,
            "linear_sharded": stats["linear_replaced"],
            "conv_sharded": stats["conv_replaced"],
        }
        with open("/home/ubuntu/work_unet_tp4/trace_result.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Wrote trace_result.json", flush=True)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
