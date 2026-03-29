# Neuron Patches

## Patch 1: NKI TopK 元素限制 (NIO-2589)

neuronx_distributed 在 TRN2 上强制使用 NKI topk kernel，大 batch 时超出 max8 指令 16384 元素限制导致编译失败。

**文件:** `neuronx_distributed/operators/topk.py`

**修复:** 将 NKI topk 替换为 XLA 版本：
```python
# 替换原有的 if/else 分支为：
topk_implementation = TopK.apply
```

影响所有模型，官方修复计划 Neuron 2.28+。

---

## Patch 2: Qwen2 缺失 --lnc 编译参数 (NIO-3231)

Qwen2 重写了 get_compiler_args() 但漏掉 --lnc flag，导致 LNC=1 时编译器静默使用 Trn2 默认的 LNC=2，运行时报错。

**文件:** `neuronx_distributed_inference/models/qwen2/modeling_qwen2.py`

**修复:** 在 get_compiler_args() return 前加一行：
```python
compiler_args += f --lnc={self.neuron_config.logical_nc_config}
```

只影响 Qwen2 系列，Qwen3 不受影响。修复后需清除编译缓存：`rm -rf /var/tmp/neuron-compile-cache`
