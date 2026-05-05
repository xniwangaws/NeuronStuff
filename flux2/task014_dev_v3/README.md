# Task 014 — FLUX.2-dev 32B Neuron Port (v3, our own)

## 目的

本目录是 **OPPO 自研 FLUX.2-dev (32B)** Neuron 端口代码与 10-seed benchmark 数据。
与 `task015_klein_jim_pr146/` (klein 9B, AWS 官方 PR #146) 并列,验证大尺寸 FLUX.2 模型也能在 trn2 上跑通。

## 与 AWS PR #146 的关系

- **参考**: 本端口的 TP 拆分模式(fused `to_qkv_mlp_proj` 拆为独立 Q/K/V/gate/value `ColumnParallelLinear`)复用了
  Jim Burtoft 在 [AWS PR #146 `contrib/flux2-klein`](https://github.com/aws-neuron/neuronx-distributed-inference/pull/146)
  为 klein 建立的 NxDI pattern。
- **自研扩展**: dev (32B) 的架构差异由我们补齐:
  - **48** single-stream blocks (vs klein **24**)
  - hidden dim **6144** (vs klein **3840**),head_dim 128,heads 48
  - **Mistral-Small-3.2-24B** text encoder (vs klein 的 **Qwen3-8B**)
  - 4-axis RoPE `axes_dim=[32,32,32,32], theta=2000`
  - 所有 linear `bias=False`
  - Shared modulation (6 × hidden) 跨 block broadcast
- TP=8 (LNC=2), BF16, trn2.48xlarge

## 文件

| 文件 | 作用 |
|---|---|
| `neuron_flux2_dit_v3.py` | 自研 DiT scaffold (~734 lines):MMDiT double-block + single-block, 4-axis RoPE, TP=8 shards, NKI flash attention |
| `compile_dit_v3.py` | Compile harness:load HF weights → split fused linears → NxDI convert → `trace_model` to NEFF |
| `verify_converter.py` | CPU shape/smoke check (TP=1 reference) 对齐到 HF `Flux2Pipeline` 单步输出 |
| `neuron_flux2_pipeline.py` | End-to-end pipeline driver (md5 `21539ec5`, 与 task011 相同),组装 TE + DiT + VAE NEFF |
| `smoke_v3.py` | 单张 1024² cat prompt smoke 测试 |
| `bench_v3_10seed.py` | 10 seed × cat prompt @ 1K 50-step benchmark runner |
| `results/bench_v3.log` | 实际 benchmark 日志 (见下) |

## 结果 (1K cat prompt,50 step,10 seed)

```
=== SUMMARY ===
Samples : 10
mean_s  : 36.95
std_s   : 0.57
min_s   : 35.69
max_s   : 37.43
pass    : 10/10
```

**组件拆分** (per image, from log):
- Text encoder (CPU Mistral-3, 非 Neuron): 0.07s
- DiT denoise loop (50 step TP=8): **27.68s** (~553 ms/step)
- VAE decode (tiled 512² → 1024²): 9.63s
- **Total**: ~37s

## 与 klein 9B 对比 (同 1K 50-step cat 10-seed)

| 模型 | 参数量 | TP | Mean (s) | Pass |
|---|---:|---:|---:|---:|
| FLUX.2-klein (Jim PR #146) | 9B DiT + 8B TE | 4 | 37.75 | 10/10 |
| **FLUX.2-dev (本 task014)** | **32B DiT + 24B TE** | **8** | **36.95** | **10/10** |

dev 比 klein 参数多 3.6×,但 TP=8 摊薄后单步延迟反而相近 — 说明 Neuron 规模扩展线性良好。

## 复现

```bash
# On trn2.48xlarge (SDK 2.29 DLAMI 20260410)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_LOGICAL_NC_CONFIG=2

# 1. Compile (一次性, ~40 min)
python compile_dit_v3.py --resolution 1024 --tp 8 \
    --weights /home/ubuntu/flux2_weights \
    --out /mnt/nvme/neff/dit_tp8_1024_v3.pt

# 2. Smoke test
python smoke_v3.py

# 3. 10-seed bench
python bench_v3_10seed.py
```

## Known issues

- Text encoder 目前跑在 CPU (Mistral-3-24B HF weights,0.07s 已够快,未 port 到 Neuron)。
  如果后续 50-step 测试 TE 成为瓶颈 (e.g. batch>1),可把 TE NEFF 编译到 Neuron(参考 task006)。
- VAE 用 tiled decode (512² → 1024² 分 9 个 tile),对 1K 已够快。2K 需重编译 DiT NEFF 并验证 VAE scaling。
