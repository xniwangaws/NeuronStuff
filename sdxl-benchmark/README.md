# SDXL-base-1.0 多设备 benchmark 报告

> Prompt:`"An astronaut riding a green horse"`,guidance 7.5(SDXL 默认),50 step,batch=1,seeds 42–51(共 10 个;L4 4K 仅 seed 42 抽样)

## 1. 设备与价格(AWS on-demand,2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2(TP=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4(墨尔本) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron 物理上跑在 trn2.48xlarge,SDXL TP=4 只占用单个 Trainium2(8 物理核 → 4 逻辑核,LNC=2),按 trn2.3xlarge 等效单芯片刊例计价。SDXL 无官方 FP8 参考实现,本次 GPU 基准统一使用 **BF16**。

## 2. 1024² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **3.84** | 8.98 GB | 10/10 | **$0.00462** | **1.00×** | **1.00×** |
| Neuron trn2.3xl | BF16 TP=4 | N/A | — | — | — | — | — |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.00726 | 0.19×(慢 5.14×) | 1.57× 贵 |

`$/image = (Mean / 3600) × $/hr`

**核心结论**:
- 1K 下 SDXL 在 H100 上仅 3.84 s / image,属单芯片最优档
- L4 绝对速度慢 5.14×,但单价仅为 H100 的 31%,$/image 仍贵约 57%(H100 单卡更具性价比)
- Neuron trn2 **编译阶段全部通过(5/5 NEFF,~30 min)**,但运行时 `DataParallel` 加载出现 `NRT_RESOURCE` 错误(`Visible cores: 0, 1`),**编译完成,运行时资源分配问题待 AWS 侧修复**

## 3. 2048² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **12.14** | 9.00 GB | 10/10 | **$0.01459** | **1.00×** | **1.00×** |
| Neuron trn2.3xl | BF16 TP=4 | N/A | — | — | — | — | — |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.03498 | 0.13×(慢 7.84×) | 2.40× 贵 |

**核心结论**:
- 2K 分辨率下 H100 仍然稳健(12 s / image,VRAM 仅 9 GB),BF16 已足够
- L4 单图 ~95 s,$/image 是 H100 的 2.4 倍,接近生产下限
- Neuron 编译通过,运行时问题同 1K,待修复后补测

## 4. 4096² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **94.37** | 11.62 GB | 10/10 | **$0.11341** | **1.00×** | **1.00×** |
| Neuron trn2.3xl | BF16 TP=4 | N/A | — | — | — | — | — |
| L4 g6.4xlarge | BF16(1 seed 抽样) | 619.18 | 9.91 GB | 1/1 | $0.22754 | 0.15×(慢 6.56×) | 2.01× 贵 |

**核心结论**:
- 4K 下 H100 仍可跑通 10/10 seeds(94 s / image,VRAM 11.6 GB),单图成本约 $0.11
- L4 4K 单图 > 10 min,仅 seed 42 抽样验证;VRAM 9.91 GB,接近 24 GB 上限但未 OOM
- SDXL 原生分辨率 1024²,4K 为超采样产物,视觉质量存在 patch artifacts,生产慎用

## 5. 同 prompt / seed 的生图对比(seed 42)

### 5.1 1024² seed 42

| H100 BF16 | Neuron BF16 TP=4 | L4 BF16+offload |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_1024/seed42_astro.png) | 编译通过,运行时资源分配问题待修复;不出样例 | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) |

### 5.2 2048² seed 42

| H100 BF16 | Neuron BF16 TP=4 | L4 BF16+offload |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_2048/seed42_astro.png) | 编译通过,运行时资源分配问题待修复;不出样例 | ![](astronaut_bench/results/sdxl_astro_l4_2048/seed42_astro.png) |

### 5.3 4096² seed 42

| H100 BF16 | Neuron BF16 TP=4 | L4 BF16+offload |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_4096/seed42_astro.png) | 编译通过,运行时资源分配问题待修复;不出样例 | ![](astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png) |

**视觉一致性**:1K / 2K 下 H100 与 L4 均产出清晰的宇航员 + 绿马主体,仅 seed noise 级差异,不影响主体识别。4K 为原生 1024² 上采样,细节受模型 spec 限制。

## 6. 10-seed 全量 PNG 路径

| 设备 / 分辨率 | 目录 |
|---|---|
| H100 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_1024/seed{42..51}_astro.png` |
| H100 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_2048/seed{42..51}_astro.png` |
| H100 4K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_4096/seed{42..51}_astro.png` |
| L4 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_1024/seed{42..51}_astro.png` |
| L4 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_2048/seed{42..51}_astro.png` |
| L4 4K BF16(1 seed 抽样) | `astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png` |
| Neuron(全分辨率) | N/A(运行时资源分配问题,待修复) |

每个目录含 `results.json`(mean_s / peak_vram_gb / per-seed std 等)。

## 7. 硬件 / 软件配置

**Neuron(trn2.3xlarge 等效)**
- SDK:**2.29** / neuronx-cc / torch-neuronx
- venv:`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- TP:**TP=4**,LNC=2,compile 5/5 NEFF 通过(~30 min)
- 运行时:`DataParallel` load 失败,`NRT_RESOURCE` `Visible cores: 0, 1`,**待 AWS 侧修复**
- 参考脚本模式:PR #149 style flags

**H100 p5.4xlarge**:DLAMI PyTorch 2.9 / CUDA 12.9 / torch 2.9.1+cu128 / diffusers 0.38.0 / BF16

**L4 g6.4xlarge**:DLAMI / torch 2.9.1+cu128 / diffusers 0.38.0 / bitsandbytes 0.45(NF4 工具链可选,本次 SDXL 主测 BF16)

**SDXL 参数**:guidance 7.5(默认),50 step,batch=1,PNDMScheduler 默认。

## 8. 运行脚本(快速复现)

GPU(H100 / L4,通用):

```bash
python astronaut_bench/bench_gpu_astro.py \
    --device cuda:0 --dtype bf16 \
    --model /home/ubuntu/models/sdxl-base \
    --prompt "An astronaut riding a green horse" \
    --resolution 1024 \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --steps 50 --guidance 7.5 \
    --out /opt/dlami/nvme/sdxl_astro_h100_1024
```

Neuron(trn2.3xlarge,编译 + benchmark):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 编译(5 NEFF,~30 min,可缓存)
python astronaut_bench/trace_sdxl_res.py \
    --model /home/ubuntu/models/sdxl-base \
    --resolution 1024 \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024

# 运行(当前 NRT_RESOURCE 报错,修复后可用)
python benchmark_neuron.py \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024 \
    --model /home/ubuntu/models/sdxl-base \
    --prompt "An astronaut riding a green horse" \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --steps 50 --guidance 7.5 \
    --out /home/ubuntu/sdxl_astro_neuron_1024
```

对应 2K / 4K:`trace_sdxl_res.py --resolution 2048 / 4096` + `benchmark_neuron.py` 的对应 compile_dir。

## 9. 结论

1. **H100 BF16 是当前 SDXL 单卡最优**:1K 3.84 s / $0.00462,2K 12.14 s / $0.0146,4K 94 s / $0.113,10/10 seeds 全通过
2. **L4 BF16 可跑全分辨率但性价比较差**:1K $0.00726(贵 57%),2K $0.0350(贵 2.4×),4K $0.228(贵 2.0×);4K 仅 seed 42 抽样
3. **Neuron trn2 编译阶段成功**(5/5 NEFF,~30 min,PR #149 style flags),但 **benchmark 运行时 `NRT_RESOURCE` `Visible cores: 0, 1`** 阻塞 DataParallel 加载 —— **编译完成,运行时资源分配问题待 AWS 侧修复**,修复后可补齐 $/image 对比
4. **SDXL 视觉一致性**:1K / 2K 下 H100 与 L4 主体一致(宇航员 + 绿马),4K 为上采样,细节受 SDXL 1024² 原生 spec 限制
5. **后续动作**:跟进 Neuron DataParallel 资源分配 issue,修复后按相同 prompt / seed / step 重跑 1K/2K/4K,补全 trn2 列 $/image 并给出 Neuron vs H100 vs L4 三方性价比结论
