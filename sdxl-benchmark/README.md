# SDXL-base-1.0 多设备 benchmark 报告 (Neuron vs L4)

_[English version: README.en.md](README.en.md)_

> Prompt: `"An astronaut riding a green horse"`, guidance 7.5 (SDXL 默认), 50 step, batch=1, seeds 42–51 (共 10 个)

## 1. 设备与价格 (AWS on-demand, 2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2 (LNC=2, 4 logical cores) | 96 GB HBM | **$2.235** | ap-southeast-4 (墨尔本) |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron 物理上跑在 trn2.48xlarge, SDXL 1K 跑 DataParallel=2 (2/4 logical cores = 1/2 Trainium2 芯片), 按 1/2 芯片刊例计价 ($1.1175/hr)。2K 用 TP=4 整芯片 ($2.235/hr)。

## 2. 1024² 端到端耗时 + $/image

| 设备 | 精度 | Mean (s) | Peak HBM/VRAM | Pass | **$/image** |
|---|---|---:|---|---:|---:|
| **Neuron trn2.3xl** | **BF16 + NKI flash-attn DP=2** | **11.14** | ~24 GB | 10/10 | **$0.0035** |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.0073 |
| **L4 g6.4xlarge** | **FP8+torch.compile** | **12.68** | 6.87 GB | 10/10 | **$0.0047** |

**核心结论 (1K)**:
- **Neuron 比 L4 (FP8+compile) 快 1.14×, 便宜 26%** ($0.0035 vs $0.0047)
- L4 BF16 慢 1.77× / 贵 110% vs Neuron

## 3. 2048² 端到端耗时 + $/image

| 设备 | 精度 | Mean (s) | Peak HBM/VRAM | Pass | **$/image** |
|---|---|---:|---|---:|---:|
| Neuron trn2.3xl | BF16 TP=4 UNet + CPU VAE | 213.9 | ~40 GB | 10/10 | $0.133 |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.0350 |
| **L4 g6.4xlarge** | **FP8+torch.compile** | **74.85** | 6.88 GB | 10/10 | **$0.0275** |

**核心结论 (2K)**:
- **L4 FP8+compile 在 2K 反而更优**: 比 Neuron 快 2.86×, 便宜 79%
- SDXL UNet 太小 (2.6B), TP=4 通信开销超过并行收益; UNet 在 256×256 spatial 的 self-attention 是瓶颈
- Neuron VAE seg 优化可降至 ~150s, 但仍慢于 L4

## 4. 4096² 可行性

| 设备 | 状态 | 备注 |
|---|---|---|
| **Neuron trn2** | **编译失败** | NCC_EOOM002: peak HBM 57GB > 24GB/core 限制 (其中 DMA spill 44GB) |
| L4 g6.4xlarge | BF16 | 619s, 9.91 GB peak (1 seed sample) |
| **L4 g6.4xlarge** | **FP8+torch.compile** | **550.21s** / $0.2022 (3 seeds sample) |

**Neuron 4K 阻塞**:
- 整体 UNet 编译: 57GB HBM > 24GB/core
- 分段编译: seg2/seg3 含 attention 失败 (320GB/48GB), seg1+seg4 PASS
- 即使加 NKI flash attention, transpose DMA spill 仍超
- **当前 SDK 2.29 + Trn2 24GB/core 是硬件限制**

## 5. 同 prompt / seed 的生图对比 (seed 42)

### 5.1 1024²

| **Neuron trn2 BF16 DP=2** | L4 BF16 |
|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) |

### 5.2 2048²

| Neuron trn2 BF16 TP=4 (213.9s) | L4 FP8+compile (74.85s) |
|:---:|:---:|
| 图像在原 ap-southeast-4 实例上生成,实例已释放 | ![](astronaut_bench/results/sdxl_astro_l4_fp8_compile_2048/seed42_astro.png) |

### 5.3 4096²

| Neuron | L4 FP8+compile (550s) |
|:---:|:---:|
| 编译失败 (HBM 24GB 限制) | ![](astronaut_bench/results/sdxl_astro_l4_fp8_compile_4096/seed42_astro.png) |

## 6. 硬件 / 软件配置

**Neuron (trn2.3xlarge 等效)**
- AMI: Neuron DLAMI (Ubuntu 24.04, SDK 2.29)
- venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- 1K: torch_neuronx.trace + DataParallel=2 + NKI flash-attn (whn09 fork)
- 2K: UNet TP=4 (Jim flags bypass NCC_EVRF007) + CPU VAE float32

**L4 g6.4xlarge**: DLAMI PyTorch 2.9 / torch 2.9.1+cu128 / diffusers 0.38 / FP8 via `torchao.Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`

## 7. 结论

1. **1K**: Neuron 略胜 — 速度快 14%, 便宜 26% vs L4 FP8+compile
2. **2K**: L4 FP8+compile 大胜 — 快 2.86×, 便宜 79% (SDXL 模型小, Neuron 并行优势不明显)
3. **4K**: 仅 L4 可用 (550s); Neuron 受 24GB/core HBM 限制无法编译
4. **场景建议**:
   - 1K: 优选 Neuron (成本最低, 速度接近)
   - 2K: 优选 L4 FP8+compile (快/便宜全胜)
   - 4K: 仅 L4 可行

## 8. 复现步骤 (Exact Reproduction Steps)

### 8.1 Neuron 复现

**实例规格**:
```
- AMI:       ami-0721447583d4d85fe (Deep Learning AMI Neuron PyTorch 2.9, Ubuntu 24.04, 2026-05-02)
- Instance:  trn2.48xlarge (or trn2.3xlarge for production - LNC=2, 4 logical cores)
- Region:    us-east-2 (Ohio) - capacity block required for trn2
- Disk:      500 GB gp3
- Key pair:  neuron-bench-us-east-2
```

**环境准备**:
```bash
# Activate Neuron venv (pre-installed in DLAMI)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install diffusers (not in default venv)
pip install diffusers transformers accelerate

# Download SDXL model
mkdir -p /home/ubuntu/models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0',
                  local_dir='/home/ubuntu/models/sdxl-base-1.0',
                  ignore_patterns=['*.onnx*', '*.bin', '*.msgpack'])
"
ln -sfn /home/ubuntu/models/sdxl-base-1.0 /home/ubuntu/models/sdxl-base

# Clone this repo
git clone https://github.com/xniwangaws/NeuronStuff.git
cd NeuronStuff/sdxl-benchmark
```

**1K Neuron (DataParallel=2, NKI flash-attn)**:
```bash
cd astronaut_bench
# Compile UNet 1K (one-time, ~10-20min)
SDXL_RES=1024 python3 trace_sdxl_res.py

# Run 10 seeds
python3 sdxl_whn09_fixed.py
# Output: /home/ubuntu/sdxl_astro_results/seed{42..51}.png
# Expected: ~11.14s/image steady state, peak ~24GB HBM
```

**2K Neuron (TP=1 + DataParallel + custom_badbmm attention)**:
```bash
cd astronaut_bench
mkdir -p /home/ubuntu/sdxl

# Compile (~30-60min for full UNet at 2K, uses --auto-cast matmult --optlevel 1)
SDXL_RES=2048 python3 trace_sdxl_res.py

# Run inference: edit bench_neuron_astro.py to set RES=2048, then:
python3 bench_neuron_astro.py
# Expected: ~213.9s/image, 10 seeds, peak ~40GB HBM
```

**4K Neuron — 不可行 (Compilation Fails)**:
```
Error: NCC_EOOM002 — Peak HBM 57.268 GB > 24 GB/core limit
Breakdown:
  - 11.190 GB I/O tensors
  - 43.803 GB DMA ring spills (21.565 GB transpose alone)
  - 4.661 GB scratchpad

Even with TP=4, NKI attention_cte, -O1, and --inst-count-limit=100M,
the compile passes HLO gen + dep_reduction + isa_gen but fails final
hbm_usage check after ~2.5 hours.

Trn2's 24GB/core HBM is the hard limit at 4K (65,536-token attention).
```

### 8.2 L4 复现

**实例规格**:
```
- AMI:       Deep Learning AMI Neuron PyTorch 2.9 (or PyTorch 2.10 with cu130)
- Instance:  g6.4xlarge
- Region:    sa-east-1 (or any with L4 capacity)
- Disk:      200 GB gp3
- Key pair:  neuron-bench-sa-east-1
```

**环境准备**:
```bash
source /opt/pytorch/bin/activate
pip install torchao diffusers transformers
```

**1K / 2K / 4K L4 BF16**:
```bash
cd astronaut_bench
python3 bench_gpu_astro.py --resolution 1024  # ~19.75s
python3 bench_gpu_astro.py --resolution 2048  # ~95.19s
python3 bench_gpu_astro.py --resolution 4096  # ~619s (1 seed, requires 22GB VRAM)
```

**1K / 2K / 4K L4 FP8+torch.compile (推荐)**:
```bash
cd astronaut_bench
python3 bench_gpu_astro_fp8_compile.py --resolution 1024  # ~12.68s, peak 6.87GB
python3 bench_gpu_astro_fp8_compile.py --resolution 2048  # ~74.85s, peak 6.88GB
python3 bench_gpu_astro_fp8_compile.py --resolution 4096  # ~550.21s, peak 7.01GB
```

### 8.3 关键文件

| 文件 | 用途 |
|------|------|
| `astronaut_bench/trace_sdxl_res.py` | Neuron trace 脚本 (参数化 RES=1024/2048/4096) |
| `astronaut_bench/sdxl_whn09_fixed.py` | Neuron 1K 推理 (NKI flash-attn DP=2) |
| `astronaut_bench/bench_neuron_astro.py` | Neuron 2K 推理 (TP=1 DP=4) |
| `astronaut_bench/bench_gpu_astro.py` | L4/H100 BF16 推理 |
| `astronaut_bench/bench_gpu_astro_fp8_compile.py` | L4/H100 FP8 + torch.compile (推荐路径) |

### 8.4 关键编译参数 (Neuron)

```python
# trace_sdxl_res.py 用的极简配置:
COMPILER_ARGS = ["--auto-cast", "matmult", "--optlevel", "1"]
DTYPE = torch.bfloat16
UNET_DEVICE_IDS = [0, 1, 2, 3]  # DataParallel across 4 logical cores

# Custom attention (诱导 neuronx-cc 更好的 lowering):
def custom_badbmm(a, b, scale):
    return torch.bmm(a, b) * scale

def get_attention_scores_neuron(self, query, key, attn_mask):
    if query.size() == key.size():
        attention_scores = custom_badbmm(key, query.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = custom_badbmm(query, key.transpose(-1, -2), self.scale)
        attention_probs = attention_scores.softmax(dim=-1)
    return attention_probs

Attention.get_attention_scores = get_attention_scores_neuron  # monkey-patch diffusers
```

### 8.5 4K 失败的尝试 (供参考, 都不成功)

| 尝试 | 配置 | 结果 |
|------|------|------|
| 整体 UNet TP=1 batch=2 | `torch_neuronx.trace` 直接 trace | NCC_EXSP001: 49GB > 24GB |
| 整体 UNet TP=4 + DataParallel | + Jim flags `--inst-count-limit=15M` | NCC_EOOM002: 57GB > 24GB |
| 分段编译 (per-block NEFF) | seg by seg, no NKI | seg1+seg4 PASS; seg2/seg3 FAIL (320GB/48GB) |
| 分段编译 + NKI attention_cte | swap AttnProcessor2_0 → NKI flash | walrus 跑 6-7 小时后仍 hbm_usage FAIL |
| NxDI-style TP=4 (ColumnParallelLinear) | `parallel_model_trace` + NKI | 同样 hbm_usage 限制 |

**结论**: 4K 在当前 SDK 2.29 + Trn2 24GB/core 上是硬件 + 编译器的根本限制。所有路径都死在 walrus 的 `hbm_usage.cpp:131 Assertion: TotalDRAMUsage <= HBMLimit`.
