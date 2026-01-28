#!/usr/bin/env python3
"""
Step 2: Profile NEFF files
Use neuron-explorer capture to profile NEFF files generated in Step 1
"""

import os
import glob
import json
import subprocess

OUTPUT_DIR = "/tmp/mlp_benchmark_neffs"


def profile_neff(neff_path: str) -> dict:
    """Profile NEFF using neuron-explorer capture"""

    basename = os.path.basename(neff_path).replace(".neff", "")
    ntff_path = os.path.join(OUTPUT_DIR, f"{basename}.ntff")

    # Run neuron-explorer capture
    cmd = f"neuron-explorer capture -n {neff_path} -s {ntff_path} --profile-nth-exec=2"
    print(f"  Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Capture failed: {result.stderr[:500]}")
        return None

    # Find generated ntff file (may have _exec_2 suffix)
    actual_ntff = ntff_path.replace(".ntff", "_exec_2.ntff")
    if not os.path.exists(actual_ntff):
        actual_ntff = ntff_path

    if not os.path.exists(actual_ntff):
        print(f"  NTFF not found")
        return None

    print(f"  Generated: {actual_ntff}")

    # Use neuron-profile view to generate JSON report
    json_path = os.path.join(OUTPUT_DIR, f"{basename}.json")
    view_cmd = f"neuron-profile view -n {neff_path} -s {actual_ntff} --output-format=json --output-file={json_path}"
    subprocess.run(view_cmd, shell=True, capture_output=True, text=True)

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        if 'summary' in data and len(data['summary']) > 0:
            summary = data['summary'][0]
            return {
                "total_time_us": summary.get('total_time', 0) * 1e6,
                "hardware_flops": summary.get('hardware_flops', 0),
                "tensor_engine_time_us": summary.get('tensor_engine_active_time', 0) * 1e6,
                "dma_time_us": summary.get('dma_active_time', 0) * 1e6,
                "vector_engine_time_us": summary.get('vector_engine_active_time', 0) * 1e6,
                "mfu_percent": summary.get('mfu_estimated_percent', 0),
            }

    return None


def main():
    print("=" * 80)
    print("Step 2: Profile NEFF files")
    print("=" * 80)

    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: {OUTPUT_DIR} not found. Run step1 first.")
        return

    # Find all NEFF files
    neff_files = glob.glob(os.path.join(OUTPUT_DIR, "*.neff"))
    if not neff_files:
        print(f"No NEFF files found in {OUTPUT_DIR}")
        return

    print(f"Found {len(neff_files)} NEFF files")

    results = []

    for neff_path in sorted(neff_files):
        basename = os.path.basename(neff_path).replace(".neff", "")
        parts = basename.split("_")
        kernel_name = parts[0]

        # Parse config
        config_parts = "_".join(parts[1:])
        config_str = config_parts.replace("b", "b=").replace("_s", ", s=").replace("_h", ", h=").replace("_i", ", i=")

        print(f"\n[{kernel_name}] {config_str}")
        print(f"  NEFF: {neff_path}")

        profile = profile_neff(neff_path)

        if profile:
            print(f"  Total kernel time: {profile['total_time_us']:.2f} μs")
            print(f"  Tensor engine: {profile['tensor_engine_time_us']:.2f} μs")
            print(f"  DMA: {profile['dma_time_us']:.2f} μs")
            print(f"  Vector engine: {profile['vector_engine_time_us']:.2f} μs")
            print(f"  MFU: {profile['mfu_percent']:.2f}%")
            print(f"  Hardware FLOPS: {profile['hardware_flops']:,}")

        results.append({
            "name": kernel_name,
            "config": config_str,
            "profile": profile,
        })

    # Print summary table
    print("\n" + "=" * 100)
    print("Performance Summary")
    print("=" * 100)
    print(f"{'Kernel':<12} {'Config':<35} {'Total (μs)':<12} {'Tensor (μs)':<12} {'DMA (μs)':<10} {'MFU (%)':<10}")
    print("-" * 100)

    for r in results:
        if r.get("profile"):
            p = r["profile"]
            print(f"{r['name']:<12} {r['config']:<35} {p['total_time_us']:<12.2f} {p['tensor_engine_time_us']:<12.2f} {p['dma_time_us']:<10.2f} {p['mfu_percent']:<10.2f}")
        else:
            print(f"{r['name']:<12} {r['config']:<35} {'FAILED':<12}")

    # Compare same configs
    print("\n" + "=" * 100)
    print("Comparison Analysis")
    print("=" * 100)

    # Group by config
    config_results = {}
    for r in results:
        config = r['config']
        if config not in config_results:
            config_results[config] = {}
        config_results[config][r['name']] = r.get('profile')

    for config, kernels in config_results.items():
        nkilib = kernels.get('nkilib')
        neuronxcc = kernels.get('neuronxcc')

        if nkilib and neuronxcc:
            t1 = nkilib['total_time_us']
            t2 = neuronxcc['total_time_us']
            ratio = t2 / t1 if t1 > 0 else 0

            if ratio > 1:
                print(f"{config}: nkilib ({t1:.2f} μs) is {ratio:.2f}x faster than neuronxcc ({t2:.2f} μs)")
            else:
                print(f"{config}: neuronxcc ({t2:.2f} μs) is {1/ratio:.2f}x faster than nkilib ({t1:.2f} μs)")


if __name__ == "__main__":
    main()
