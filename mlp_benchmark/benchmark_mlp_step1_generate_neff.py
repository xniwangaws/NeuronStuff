#!/usr/bin/env python3
"""
Step 1: 生成 NEFF 文件
运行 nkilib 和 neuronxcc 的 MLP kernel，保存 NEFF 文件到指定目录
"""

import os
import glob
import shutil

# 设置环境变量
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import torch
import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel
from neuronxcc.nki.language import nc

# nkilib kernel (需要安装 nkilib_standalone)
from nkilib_standalone.nkilib.core.mlp.mlp import mlp as nkilib_mlp


OUTPUT_DIR = "/tmp/mlp_benchmark_neffs"


def find_latest_neff(pattern="*.neff"):
    """找到最新生成的 NEFF 文件"""
    neff_files = glob.glob(pattern)
    if not neff_files:
        return None
    return max(neff_files, key=os.path.getctime)


def generate_nkilib_neff(batch: int, seqlen: int, hidden: int, intermediate: int, output_name: str):
    """Generate NEFF for nkilib MLP kernel"""
    print(f"\n[nkilib] batch={batch}, seqlen={seqlen}, hidden={hidden}, intermediate={intermediate}")

    device = xm.xla_device()

    # 清除之前的 NEFF 文件
    for f in glob.glob("*.neff"):
        os.remove(f)

    # 创建输入
    hidden_tensor = torch.randn(batch, seqlen, hidden, dtype=torch.bfloat16).to(device)
    gate_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    up_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    down_w = (torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02).to(device)

    # 执行 kernel（触发编译）
    print("  Compiling and executing...")
    output = nkilib_mlp(
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_w,
        up_proj_weights_tensor=up_w,
        down_proj_weights_tensor=down_w,
    )
    xm.mark_step()
    xm.wait_device_ops()

    # 找到生成的 NEFF
    neff_path = find_latest_neff()
    if not neff_path:
        print("  NEFF not found!")
        return None

    # 复制到输出目录
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.neff")
    shutil.copy(neff_path, output_path)
    print(f"  Saved NEFF: {output_path}")

    return output_path


def generate_neuronxcc_neff(batch: int, seqlen: int, hidden: int, intermediate: int, output_name: str):
    """Generate NEFF for neuronxcc mlp_isa_kernel"""
    print(f"\n[neuronxcc] batch={batch}, seqlen={seqlen}, hidden={hidden}, intermediate={intermediate}")

    device = xm.xla_device()

    # 清除之前的 NEFF 文件
    for f in glob.glob("*.neff"):
        os.remove(f)

    # 创建输入
    hidden_tensor = torch.randn(batch, seqlen, hidden, dtype=torch.bfloat16).to(device)
    ln_w = torch.ones(1, hidden, dtype=torch.bfloat16).to(device)
    gate_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    up_w = (torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02).to(device)
    down_w = (torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02).to(device)
    output_tensor = torch.zeros(batch, seqlen, hidden, dtype=torch.bfloat16, device=device)

    logical_nc_config = 2
    grid = (nc(logical_nc_config),)
    _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

    # 执行 kernel（触发编译）
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

    # 找到生成的 NEFF
    neff_path = find_latest_neff()
    if not neff_path:
        print("  NEFF not found!")
        return None

    # 复制到输出目录
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.neff")
    shutil.copy(neff_path, output_path)
    print(f"  Saved NEFF: {output_path}")

    return output_path


def main():
    print("=" * 80)
    print("Step 1: 生成 NEFF 文件")
    print("=" * 80)

    # 创建输出目录
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
    print("生成的 NEFF 文件")
    print("=" * 80)
    for kernel, config, path in neff_files:
        print(f"  {kernel:<12} {config:<35} {path}")

    print(f"\n所有 NEFF 文件保存在: {OUTPUT_DIR}")
    print("\n下一步: 运行 benchmark_mlp_step2_profile.py 来 profile 这些 NEFF 文件")


if __name__ == "__main__":
    main()
