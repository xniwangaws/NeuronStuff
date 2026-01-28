# MLP Kernel Benchmark

公平对比 nkilib MLP kernel 和 neuronxcc mlp_isa_kernel 的性能。

## 方法

使用 `NEURON_FRAMEWORK_DEBUG=1` + `neuron-explorer capture` 方法测量实际 kernel 执行时间。

## 使用方法

```bash
# 激活 Neuron 环境
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Step 1: 生成 NEFF 文件
python3 benchmark_mlp_step1_generate_neff.py

# Step 2: Profile NEFF 文件 (需要单独运行，独占 Neuron device)
python3 benchmark_mlp_step2_profile.py
```

## 结果

| Kernel | Config | Total (μs) | Tensor Eng (μs) | DMA (μs) |
|--------|--------|------------|-----------------|----------|
| neuronxcc | b=1, s=128, h=1024, i=512 | 39.36 | 14.66 | 19.80 |
| nkilib | b=1, s=128, h=1024, i=512 | 43.22 | 15.22 | 21.08 |
| neuronxcc | b=1, s=256, h=2048, i=1024 | 67.21 | 36.64 | 40.35 |
| nkilib | b=1, s=256, h=2048, i=1024 | 85.26 | 57.09 | 53.12 |

**对比:**
- 小配置: neuronxcc 比 nkilib 快 1.10x
- 大配置: neuronxcc 比 nkilib 快 1.27x

## 文件说明

- `benchmark_mlp_step1_generate_neff.py` - 生成 NEFF 文件
- `benchmark_mlp_step2_profile.py` - Profile NEFF 文件
- `benchmark_results.txt` - 详细结果和分析

## 依赖

- AWS Neuron SDK
- PyTorch NeuronX
- nkilib_standalone (需要安装)

## 关键参数

`--profile-nth-exec=2`: 执行 NEFF 2 次，只 profile 第 2 次，避免 warmup 开销影响结果。
