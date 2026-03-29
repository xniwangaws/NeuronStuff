# Neuron Patches

venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/`

## Patch 1: NKI TopK 元素限制 (NIO-2589)

neuronx_distributed 在 TRN2 上强制使用 NKI topk kernel，大 batch 时超出 max8 指令 16384 元素限制导致编译失败。

**文件:** `neuronx_distributed/operators/topk.py` 第 69-74 行

**修复:** 将第 69-74 行：
```python
can_use_nki_topk = dim in (-1, len(tensor.shape) - 1) and _is_nki_topk_available()
if can_use_nki_topk:
    stages = 1
    topk_implementation = nki_topk
else:
    topk_implementation = TopK.apply
```
替换为：
```python
topk_implementation = TopK.apply
```

影响所有模型，官方修复计划 Neuron 2.28+。

---

## Patch 2: Qwen2 缺失 --lnc 编译参数 (NIO-3231)

Qwen2 重写了 get_compiler_args() 但漏掉 --lnc flag，导致 LNC=1 时编译器静默使用 Trn2 默认的 LNC=2，运行时报错。

**文件:** `neuronx_distributed_inference/models/qwen2/modeling_qwen2.py` 第 237-242 行

**修复:** 在第 241 行（return 前）插入一行：
```python
compiler_args += f --lnc={self.neuron_config.logical_nc_config}
```

修复后第 237-243 行应为：
```python
def get_compiler_args(self):
    compiler_args = --enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1
    # Add flags for cc-overlap
    compiler_args +=  --tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma
    compiler_args +=  --internal-hlo2tensorizer-options=--verify-hlo=true
    compiler_args += f --lnc={self.neuron_config.logical_nc_config}
    return compiler_args
```

只影响 Qwen2 系列，Qwen3 不受影响。修复后需清除编译缓存：`rm -rf /var/tmp/neuron-compile-cache`
