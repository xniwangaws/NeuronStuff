#!/usr/bin/env python3
"""
Step 1: Generate NEFF files
Run nkilib and neuronxcc MLP kernels, save NEFF files to output directory
"""

import os
import glob
import shutil

# Set environment variables
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import torch
import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel
from neuronxcc.nki.language import nc

# nkilib kernel (requires nkilib_standalone to be installed)
from nkilib_standalone.nkilib.core.mlp.mlp import mlp as nkilib_mlp


OUTPUT_DIR = "/tmp/mlp_benchmark_neffs"


def find_latest_neff(pattern="*.neff"):
    """Find the most recently generated NEFF file"""
    neff_files = glob.glob(pattern)
    if not neff_files:
        return None
    return max(neff_files, key=os.path.getctime)


def generate_nkilib_neff(batch: int, seqlen: int, hidden: int, intermediate: int, output_name: str):
    """Generate NEFF for nkilib MLP kernel"""
    print(f"\n[nkilib] batch={batch}, seqlen={seqlen}, hidden={hidden}, intermediate={intermediate}")

    device = xm.xla_device()

    # Clear previous NEFF files
    for f in glob.glob("*.neff"):
        os.remove(f)

    # Create inputs
    hidden_tensor = torch.randn(batch, seqlen, hidden, dtype=torch.bfloat16).to(device)
    gate_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    up_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    down_w = (torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02).to(device)

    # Execute kernel (triggers compilation)
    print("  Compiling and executing...")
    output = nkilib_mlp(
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_w,
        up_proj_weights_tensor=up_w,
        down_proj_weights_tensor=down_w,
    )
    xm.mark_step()
    xm.wait_device_ops()

    # Find generated NEFF
    neff_path = find_latest_neff()
    if not neff_path:
        print("  NEFF not found!")
        return None

    # Copy to output directory
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.neff")
    shutil.copy(neff_path, output_path)
    print(f"  Saved NEFF: {output_path}")

    return output_path


def generate_neuronxcc_neff(batch: int, seqlen: int, hidden: int, intermediate: int, output_name: str):
    """Generate NEFF for neuronxcc mlp_isa_kernel"""
    print(f"\n[neuronxcc] batch={batch}, seqlen={seqlen}, hidden={hidden}, intermediate={intermediate}")

    device = xm.xla_device()

    # Clear previous NEFF files
    for f in glob.glob("*.neff"):
        os.remove(f)

    # Create inputs
    hidden_tensor = torch.randn(batch, seqlen, hidden, dtype=torch.bfloat16).to(device)
    ln_w = torch.ones(1, hidden, dtype=torch.bfloat16).to(device)
    gate_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    up_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    down_w = (torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02).to(device)
    output_tensor = torch.zeros(batch, seqlen, hidden, dtype=torch.bfloat16, device=device)

    logical_nc_config = 2
    grid = (nc(logical_nc_config),)
    _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

    # Execute kernel (triggers compilation)
    print("  Compiling and executing...")
    _mlp_fwd_call[grid](
        hidden_tensor,
        ln_w,
        gate_w,
        up_w,
        down_w,
        output_tensor,
        kernel_name="MLP",
        fused_rmsnorm=False,
        skip_gamma=False,
        eps=1e-5,
    )
    xm.mark_step()
    xm.wait_device_ops()

    # Find generated NEFF
    neff_path = find_latest_neff()
    if not neff_path:
        print("  NEFF not found!")
        return None

    # Copy to output directory
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.neff")
    shutil.copy(neff_path, output_path)
    print(f"  Saved NEFF: {output_path}")

    return output_path


def main():
    print("=" * 80)
    print("Step 1: Generate NEFF files")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [
        (1, 128, 1024, 512),
        (1, 256, 2048, 1024),
    ]

    neff_files = []

    for batch, seqlen, hidden, intermediate in configs:
        config_suffix = f"b{batch}_s{seqlen}_h{hidden}_i{intermediate}"

        # Generate nkilib NEFF
        nkilib_neff = generate_nkilib_neff(
            batch, seqlen, hidden, intermediate,
            f"nkilib_{config_suffix}"
        )
        if nkilib_neff:
            neff_files.append(("nkilib", f"b={batch}, s={seqlen}, h={hidden}, i={intermediate}", nkilib_neff))

        # Generate neuronxcc NEFF
        neuronxcc_neff = generate_neuronxcc_neff(
            batch, seqlen, hidden, intermediate,
            f"neuronxcc_{config_suffix}"
        )
        if neuronxcc_neff:
            neff_files.append(("neuronxcc", f"b={batch}, s={seqlen}, h={hidden}, i={intermediate}", neuronxcc_neff))

    # Print summary
    print("\n" + "=" * 80)
    print("Generated NEFF files")
    print("=" * 80)
    for kernel, config, path in neff_files:
        print(f"  {kernel:<12} {config:<35} {path}")

    print(f"\nAll NEFF files saved to: {OUTPUT_DIR}")
    print("\nNext step: Run benchmark_mlp_step2_profile.py to profile these NEFF files")


if __name__ == "__main__":
    main()
