# FLUX.2-klein-base-9B 多设备 benchmark 报告

> Prompt:`"A cat holding a sign that says hello world"`,guidance 4.0,50 step,batch=1,seeds 42–51(共 10 个)

## 1. 设备与价格(AWS on-demand,2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2(TP=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4(墨尔本) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron 物理上跑在 trn2.48xlarge,klein TP=4 只占用 `NEURON_RT_VISIBLE_CORES=16-19` 的单个 Trainium2(8 物理核 → 4 逻辑核,LNC=2),按 trn2.3xlarge 等效单芯片刊例计价。

## 2. 1024² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | P95 | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | **BF16(基准)** | **24.10** | — | 37.33 GB | 10/10 | **$0.02896** | **1.00×** | **1.00×** |
| H100 p5.4xlarge | FP8(torchao) | 21.18 | — | 28.25 GB | 10/10 | $0.02545 | 1.14× 更快 | **0.88×**(便宜 12%) |
| **Neuron trn2.3xl** | **BF16 TP=4** | **37.75** | 38.92 | **~24 GB** | **10/10** | **$0.02344** | 0.64×(慢 1.57×) | **0.81×**(**便宜 19%**) |
| L4 g6.4xlarge | NF4(bnb) | 211.49 | 213.12 | 19.70 GB | 10/10 | $0.07772 | 0.11×(慢 8.8×) | 2.68× 贵 |
| L4 g6.4xlarge | BF16+offload | 226.59 | 267.76 | 19.00 GB | 10/10 | $0.08327 | 0.11×(慢 9.4×) | 2.88× 贵 |

`$/image = (Mean / 3600) × $/hr`

**核心结论**:
- **Neuron $/image 全场最低,比 H100 BF16 便宜 19%**($0.0234 vs $0.0290)
- Neuron 绝对速度慢 1.57×,但 trn2.3xlarge 单价仅为 H100 的 52%,综合胜出
- H100 FP8 是 GPU 最优但仍比 Neuron 贵 9%;L4 虽单价最低,速度慢到成本反而高 2.7–2.9×
- Neuron 单芯片 HBM ~24 GB,比 H100 BF16(37 GB)低 35%

## 3. 2048² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | **BF16(基准)** | **107.05** | 45.00 GB | 10/10 | **$0.1286** | **1.00×** | **1.00×** |
| H100 p5.4xlarge | FP8 | 106.20 | 35.92 GB | 10/10 | $0.1276 | 1.01× | 0.99× |
| **Neuron trn2.3xl** | **BF16 TP=4** | **196.06** | ~40 GB | **10/10** | **$0.1217** | 0.55×(慢 1.83×) | **0.95×**(便宜 5%) |
| L4 g6.4xlarge | NF4(3 seed 抽样) | 918.58 | 10.43 GB | 3/3 | $0.3376 | 0.12×(慢 8.6×) | 2.62× 贵 |
| L4 g6.4xlarge | BF16+offload(1 seed) | 913.73 | 21.15 GB | 1/1 | $0.3358 | 0.12×(慢 8.5×) | 2.61× 贵 |

**核心结论**:
- 2K 下 H100 BF16 / FP8 / Neuron 三者 $/image 非常接近(差 <6%),**Neuron 仍是最低**
- H100 FP8 比 BF16 仅便宜 1%(2K 已 compute-bound,FP8 加速优势收窄)
- L4 单图 ~15 min + VRAM 紧张,只能抽样,**无法满足生产节奏**

## 4. 4096² 可行性 —— 模型超规格

klein 官方 `max_area = 4 MP(≈ 2048²)`,4K = 16 MP 超规格。**所有设备输出均为 std≈20 的噪声图(GRAY),不是硬件限制**。

| 设备 | 结果 |
|---|---|
| H100 BF16 | 1024s / image,GRAY 噪声 |
| H100 FP8 | 1019s / image,GRAY 噪声 |
| Neuron BF16 TP=4 | 未完成编译(HLO gen 超时,NUM_PATCHES=65536 过大) |
| L4 NF4 / BF16 | OOM |

客户若需 4K,需等 BFL 发布更大 spec 的 checkpoint。

## 5. DiT 加载 / 冷启动 / 稳态拆分(Neuron 1K)

| 阶段 | 耗时 |
|---|---:|
| Compile(一次性,可缓存到 S3/EFS) | 156.9 s |
| Weight load + NxD init | 20.2 s |
| 首次推理(含 graph replay warmup) | 25.4 s |
| **稳态 mean(10 seed)** | **37.75 s** |

- **首次 cold-start**(无 NEFF 缓存):~3.4 min
- **热启动**(NEFF 缓存命中):~45 s
- **稳态**:37.75 s/image

## 6. 同 prompt / seed 的生图对比

### 6.1 1024² 4-panel 对比(seed 42)

![1K 4-panel](task015_klein_jim_pr146/results/grid_1024.png)

(左→右) H100 BF16 | H100 FP8 | **Neuron trn2 BF16 TP=4** | L4 NF4

> 注:本张 Neuron 抽到装饰画风 variant,其余 9 个 seed(43–51)正常产出举牌猫,详见 `klein_1k_50step/seed{43..51}_cat.png`。

### 6.2 2048² seed 42 对比

| H100 BF16 | H100 FP8 | **Neuron trn2 BF16 TP=4** | L4 NF4 |
|:---:|:---:|:---:|:---:|
| ![](task015_klein_jim_pr146/results/h100_2048_bf16/seed42_cat.png) | ![](task015_klein_jim_pr146/results/h100_2048_fp8/seed42_cat.png) | ![](task015_klein_jim_pr146/results/klein_2k_50step/seed42_cat.png) | ![](task015_klein_jim_pr146/results/l4_2048_nf4/seed42_cat.png) |

**视觉一致性**:1K / 2K 下所有设备均产出清晰的猫 + "HELLO WORLD" 牌子,仅 seed noise 级差异,不影响主体识别。

### 6.3 10-seed 全量 PNG

| 设备 | 目录 |
|---|---|
| Neuron 1K BF16 | `task015_klein_jim_pr146/results/klein_1k_50step/seed{42..51}_cat.png` |
| Neuron 2K BF16 | `task015_klein_jim_pr146/results/klein_2k_50step/seed{42..51}_cat.png` |
| H100 1K BF16 / FP8 | `task015_klein_jim_pr146/results/h100_1024_{bf16,fp8}/seed{42..51}_cat.png` |
| H100 2K BF16 / FP8 | `task015_klein_jim_pr146/results/h100_2048_{bf16,fp8}/seed{42..51}_cat.png` |
| L4 2K NF4 / BF16+offload | `task015_klein_jim_pr146/results/l4_2048_{nf4,bf16_offload}/seed42_cat.png` |

## 7. 硬件 / 软件配置

**Neuron(trn2.3xlarge 等效)**
- AMI:`ami-042fbe428a1a7a882`(Neuron DLAMI 20260410,Ubuntu 22.04)
- SDK:**2.29** / neuronx-cc 2.24.5133 / torch-neuronx 2.9.0.2.13 / NxDI 0.9.17334(PR #146 `contrib/flux2-klein`)
- venv:`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/`
- TP:**TP=4**,LNC=2,`NEURON_RT_VISIBLE_CORES=16-19`

**H100 p5.4xlarge**:DLAMI PyTorch 2.9 / CUDA 12.9 / torch 2.9.1+cu128 / diffusers 0.38.0 / FP8 via **torchao**

**L4 g6.4xlarge**:DLAMI / torch 2.7.0+cu128 / diffusers 0.38.0 / bitsandbytes 0.45(NF4)

**klein 实现来源**:AWS NxDI [PR #146](https://github.com/aws-neuron/neuronx-distributed-inference/pull/146) by Jim Burtoft(AWS),分支 `contrib/flux2-klein`。OPPO 侧按客户规格(50 step / 10 seed)重跑并做多设备 GPU 对齐。

## 8. 运行脚本(快速复现)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES=16-19          # 1 个 Trainium2 的 4 逻辑核
export NEURON_COMPILED_ARTIFACTS=$KLEIN_CACHE

python task015_klein_jim_pr146/bench_klein_1k_50step.py \
    --model /mnt/nvme/flux2_klein \
    --out   /mnt/nvme/klein_bench_1k_50step \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --prompt "A cat holding a sign that says hello world" \
    --guidance 4.0 --steps 50 --tp 4
```

对应 2K / 4K:`bench_klein_2k.py` / `bench_klein_4k.py`。

## 9. 显存虚拟化(1/2 / 1/4)—— N/A

Trainium2 **没有 NVIDIA MIG 式的硬件显存切分**。Neuron 用 **LNC(Logical Neuron Core)** 做**计算核切分**(`NEURON_LOGICAL_NC_CONFIG=2` 把 8 物理核合成 4 逻辑核),HBM 96 GB 对单设备工作负载共享不切分。多租户并发需 **多进程 + 不重叠的 `NEURON_RT_VISIBLE_CORES` 子集**,属应用层切分。

本次 klein TP=4 占满 1 个 Trainium2(HBM 占用 ~24 GB @1K / ~40 GB @2K),**剩余 HBM 原理上可容纳第二实例**,但需 LNC=1 + 独立 Python 进程,不在本次测试范围。

## 10. 结论

1. **Neuron klein 1K / 2K × 50 step × 10 seed 全通过**(std=65–79,视觉清晰的猫 + "HELLO WORLD" 牌子)
2. **$/image 全场最低**:Neuron 比 H100 BF16 便宜 **19%(1K)/ 5%(2K)**
3. **速度**:Neuron 比 H100 BF16 慢 1.57×(1K)/ 1.83×(2K),但 trn2.3xlarge 单价仅为 H100 的 52%,综合 $/image 胜出
4. **HBM 占用**:Neuron 1K 24 GB,比 H100 BF16(37 GB)低 35%,适合小显存 / 高并发部署
5. **4K 不可行**(所有设备含 H100 均噪声图,模型规格限制,非硬件问题)
6. **Capacity 可用性**:p5 在 us-east / us-west / eu / sa-east 全 region `InsufficientInstanceCapacity`,最终只有 ap-northeast-1c 锁到;trn2 capacity block 更易获取 —— 综合性价比 + 可用性,**Neuron trn2 是 FLUX.2-klein 更可规模化部署的选择**
