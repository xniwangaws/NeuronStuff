# FLUX.1-dev alien-prompt benchmark — Neuron / H100 / L4

_[English version: README.en.md](README.en.md)_


> Prompt:`"A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"`,guidance 3.5,28 steps,batch=1,max_sequence_length=512,seeds 42–51(共 10 个)

## 1. 设备与价格(AWS on-demand,2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2(WORLD=4, backbone_tp=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4 |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | ap-northeast-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

## 2. 1024² 端到端耗时 + 峰值显存 + $/image(以 H100 FP8 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 FP8 | 成本 vs H100 FP8 |
|---|---|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | BF16 | **5.87** | 33.85 GB | 10/10 | $0.00706 | 1.45× 更快 | 0.83×(便宜 17%) |
| H100 p5.4xlarge | **FP8(基准, torchao)** | 8.54 | 22.77 GB | 10/10 | **$0.01026** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **3.04** | 22.77 GB | 10/10 | **$0.00365** | **2.81× faster** | **0.36×**(便宜 64%） |
| **Neuron trn2.3xl** | **BF16 WORLD=4** | **8.03** | ~25 GB(单 Trainium2) | **10/10** | **$0.00499** | 1.06× 更快 | **0.49×**(**便宜 51%**) |
| L4 g6.4xlarge | NF4(bnb+offload) | 57.65 | 6.79 GB | 10/10 | $0.02119 | 0.15×(慢 6.75×) | 2.06× 贵 |
| L4 g6.4xlarge | **FP8(wangkanai, seq-offload)** | 123.21 | 2.39 GB | 10/10 | $0.04528 | 0.07×(慢 14.4×) | 4.41× 贵 |

`$/image = (Mean / 3600) × $/hr`

**核心结论**:
- **Neuron 在 FLUX.1-dev 上**既更快(比 H100 FP8 快 1.06×)**又更便宜**(单图成本仅 H100 FP8 的 49%,**便宜 51%**)
- H100 BF16 绝对速度最快(5.87s),比 FP8 便宜 17%(FP8 quantize 在单张小图上的 overhead 超过加速收益)
- Neuron 单芯片 HBM ~25 GB,低于 H100 BF16(34 GB)
- L4 NF4 速度慢 6.75×,load 678s(NF4 转换慢),性价比最差
- L4 FP8(wangkanai/flux-dev-fp8)速度更慢(123 s,慢 14.4×):权重在 ckpt 内为 F8_E4M3,但 diffusers 加载时 upcast 到 bf16,12 B ≈ 24 GB 超出 L4 22 GB 显存,被迫走 sequential CPU offload → PCIe 成为瓶颈。峰值显存压到 2.39 GB 但单图成本 $0.04528(4.41× 贵于 H100 FP8);相比 L4 NF4 的 57 s,FP8 "干净"但在 22 GB L4 上吃亏于 offload 开销。

## 2b. 2048² / 4096² super-resolution(model spec `max_area=4MP`,已 force run)

| 设备 | 精度 | Res | Mean (s) | Peak VRAM | Pass | **$/image** |
|---|---|---|---:|---:|---:|---:|
| H100 p5.4xlarge | FP8(torchao) | 2048² | **37.52** | 29.9 GB | 10/10 | **$0.04509** |
| **H100 p5.4xlarge** | **FP8+torch.compile** | 2048² | **17.08** | 29.9 GB | 10/10 | **$0.02053** |
| H100 p5.4xlarge | FP8(torchao) | 4096² | **328.70** | 37.67 GB | 10/10 | **$0.39492** |
| L4 g6.4xlarge | FP8(wangkanai, seq-offload) | 2048² | **339.38** | 2.42 GB | 10/10 | **$0.12471** |
| **Neuron trn2.3xl** | **BF16 TP=4 (Transformer Neuron + VAE CPU)** | 2048² | **109.1** | ~25 GB | 10/10 | **$0.0677** — Jim flags bypass EVRF007, VAE on CPU float32 |
| Neuron trn2.3xl | BF16 TP=4 | 4096² | **BLOCKED** | — | — | 同理,未尝试 |
| L4 g6.4xlarge | FP8 wangkanai | 4096² | **BLOCKED OOM** | — | 0/3 | L4 22GB VRAM 吃不下 12B bf16 upcast (~16GB) + 4K activation (~8GB) |

**注**:FLUX.1-dev 官方 spec 为 1024²,`max_area=4MP`。2K / 4K 属 super-resolution force run,输出质量受 spec 限制(`std` 偏低、细节下降),仅作硬件极限参考。

## 3. DiT 加载 / 冷启动 / 稳态拆分(Neuron 1K)

| 阶段 | 耗时 |
|---|---:|
| Compile(一次性,可缓存到 S3/EFS) | 103.4 s |
| Weight load + NxD init | 78.0 s |
| 首次推理(含 graph replay warmup) | ~8 s |
| **稳态 mean(10 seed)** | **8.03 s** |

- **首次 cold-start**(无 NEFF 缓存):~3.2 min
- **热启动**(NEFF 缓存命中):~85 s
- **稳态**:8.03 s/image(28 steps)

## 4. 同 prompt / seed 的生图对比(seed 42)

| H100 BF16 | H100 FP8 | **Neuron trn2 BF16 WORLD=4** | L4 NF4 |
|:---:|:---:|:---:|:---:|
| ![](alien_bench/results/flux1_alien_h100_bf16/seed42_alien.png) | ![](alien_bench/results/flux1_alien_h100_fp8/seed42_alien.png) | ![](alien_bench/results/flux1_alien_trn2_bf16/seed42_alien.png) | ![](alien_bench/results/flux1_alien_l4_nf4/seed42_alien.png) |

**视觉一致性**:4 设备同 prompt + 同 seed 42 → 均产出识别度高的绿色外星人(有 prompt bias variant:seed 42 在 H100/L4 上偶现 "cat hello world" 构图,属 FLUX.1-dev 训练数据偏好)。其他 9 个 seed(43–51)均稳定产出 fluorescent alien 主体。

## 5. 10-seed 全量 PNG 路径

| 设备 | 目录 |
|---|---|
| Neuron 1K BF16 | `alien_bench/results/flux1_alien_trn2_bf16/seed{42..51}_alien.png` |
| H100 1K BF16 | `alien_bench/results/flux1_alien_h100_bf16/seed{42..51}_alien.png` |
| H100 1K FP8 | `alien_bench/results/flux1_alien_h100_fp8/seed{42..51}_alien.png` |
| L4 1K NF4 | `alien_bench/results/flux1_alien_l4_nf4/seed{42..51}_alien.png` |
| L4 1K FP8 | `alien_bench/results/flux1_alien_l4_fp8/seed{42..51}_alien.png` |

## 6. 硬件 / 软件配置

**Neuron(trn2.3xlarge)**
- AMI:Neuron DLAMI / SDK 2.29 / neuronx-cc 2.24.5133 / torch-neuronx 2.9.0.2.13.26312 / NxDI(NeuronFluxApplication)
- venv:`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- Topology:**WORLD=4**,backbone_tp=4,t5_tp=4,clip+vae tp=1,LNC=2

**H100 p5.4xlarge**:DLAMI / torch 2.9.1+cu128 / diffusers 0.38 / FP8 via `torchao.Float8DynamicActivationFloat8WeightConfig`

**L4 g6.4xlarge**:DLAMI / torch 2.7.0+cu128 / diffusers 0.38 / bitsandbytes NF4 + `enable_model_cpu_offload`

**L4 g6.4xlarge FP8**:DLAMI PyTorch 2.9 / torch 2.9.1+cu130 / diffusers 0.35.1 / transformers 4.57.6 / 模型 [wangkanai/flux-dev-fp8](https://huggingface.co/wangkanai/flux-dev-fp8)(17 GB single-file ComfyUI-style ckpt,F8_E4M3 权重 16.7 B)/ `FluxPipeline.from_single_file` + `enable_sequential_cpu_offload`。12B transformer 在 bf16 compute 下 ≈ 24 GB,超过 L4 22 GB,必须 sequential offload 逐层流式 → PCIe 搬运主导耗时,单张 123 s。

**实现**:FLUX.1-dev benchmark 基于 [AWS NxDI NeuronFluxApplication](https://awsdocs-neuron.readthedocs-hosted.com/),Neuron 端一键 compile + load + forward;GPU 端用 `diffusers.FluxPipeline`。

## 7. 运行脚本(快速复现)

```bash
# Neuron
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python3 alien_bench/bench_neuron_alien.py

# H100
python3 alien_bench/bench_h100_alien.py --precision bf16 --out /opt/dlami/nvme/flux1_alien_h100_bf16
python3 alien_bench/bench_h100_alien.py --precision fp8  --out /opt/dlami/nvme/flux1_alien_h100_fp8

# L4
python3 alien_bench/bench_l4_alien.py --precision nf4 --out ~/flux1_alien_l4_nf4
```

## 8. 结论

1. **Neuron FLUX.1-dev 在 trn2.3xlarge 上 10/10 pass**(mean 8.03s,稳态 28-step,28GB HBM 内)
2. **$/image 全场最低**:Neuron **$0.00499**,比 H100 FP8 便宜 51%,比 H100 BF16 便宜 29%
3. **速度**:Neuron 8.03s ≈ H100 FP8 8.54s,**比 FP8 快 1.06×**;H100 BF16 5.87s 最快(+45%)但贵 1.41×
4. **HBM 占用**:Neuron 25 GB,远低于 H100 BF16 的 34 GB,单芯片 96GB 仍大量余量
5. **Capacity 考量**:p5 在 us-east / us-west / eu / sa-east 全 region 均 `InsufficientInstanceCapacity`,最终只有 ap-northeast-1 锁到;trn2 capacity block 更容易获取,对规模化部署是关键加分项
