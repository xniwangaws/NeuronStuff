# Task 015 — FLUX.2-klein-base-9B Neuron (via AWS PR #146)

## 来源

本目录的 Neuron klein 端口 **不是 OPPO 自研**,而是基于 AWS 官方 NxDI contribution:

**[aws-neuron/neuronx-distributed-inference PR #146](https://github.com/aws-neuron/neuronx-distributed-inference/pull/146)**
- 作者: **Jim Burtoft (AWS)**
- 分支: `contrib/flux2-klein`
- 模型: Black Forest Labs [`flux2-klein-base-9B`](https://huggingface.co/black-forest-labs/flux2-klein-base-9B)
- 架构: 9B DiT + **Qwen3-8B** Text Encoder + Flux2 VAE

## OPPO 做了什么

1. **Run & verify**: 在我方 `trn2.48xlarge` 环境 (SDK 2.29 DLAMI 20260410) 上跑通 PR #146 klein
   port,确认 end-to-end 图像质量与延迟符合预期。
2. **OPPO-spec benchmark**: Jim 原本 benchmark 是 30 step,本任务按 OPPO 客户要求改为
   **50 step × 10 seed × cat-hello-world prompt**,覆盖 1K + 2K 两个分辨率。
3. **GPU baseline 对齐**: 把 klein 的 Neuron 结果接入主 REPORT §13 与 dev 在相同 prompt/step/seed 下的 H100 / L4 数据对比。
4. **Source 尊重原作者**: Jim 会把源码推到他自己 PR #146,本目录 **不 copy 他的 `application.py` /
   `modeling_flux2_klein.py`**,避免和官方 PR 冲突。如需复现,请 checkout 他的 branch。

## 文件

| 文件 | 作用 |
|---|---|
| `README.md` | 本文档 |
| `bench_klein_1k_50step.py` | OPPO-spec 10-seed benchmark runner for 1K |
| `bench_klein_2k.py` | 同上 2K |
| `bench_klein_4k.py` | 4K 编译尝试 (klein 官方 max_area=4MP 超规格,仅供参考) |
| `bench_klein_10seed.py` | Jim 原版 10-seed (30 step) reference |
| `results/klein_1k_50step/` | 10 seed × 1K × cat 输出 PNG + `results.json` |
| `results/klein_2k_50step/` | 10 seed × 2K × cat 输出 PNG + `results.json` |

## 结果 (1K cat prompt,50 step,10 seed,TP=4,BF16)

```
mean_s : 37.75
std_s  : 1.04
min_s  : 35.43
max_s  : 38.92
pass   : 10/10
```

## 结果 (2K cat prompt,50 step,10 seed,TP=4,BF16)

```
mean_s  : 196.06
std_s   : 1.52
passes  : 10/10
compile : 887.7s (一次性)
```

## 复现

**Step 1** — checkout Jim 的 PR branch:
```bash
git clone https://github.com/aws-neuron/neuronx-distributed-inference
cd neuronx-distributed-inference
gh pr checkout 146
# 或: git fetch origin pull/146/head:contrib-flux2-klein && git checkout contrib-flux2-klein
```

**Step 2** — 下载 weights:
```bash
huggingface-cli download black-forest-labs/flux2-klein-base-9B \
    --local-dir /mnt/nvme/flux2_klein_weights
```

**Step 3** — Compile + run (在 Jim 提供的 `src/neuronx_distributed_inference/models/flux2_klein/` 下):
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_LOGICAL_NC_CONFIG=2
python bench_klein_1k_50step.py   # 从本目录拷到 PR146 src 目录
```

## 与 FLUX.2-dev (32B) 对比

见 `/Users/xniwang/NeuronStuff/flux2/task014_dev_v3/README.md` 的对比表 —
dev 参数 3.6× klein,但 TP=8 摊薄后单步延迟相近 (36.95s vs 37.75s @ 1K 50-step)。
