# S3Diff on AWS Trainium2 — 生产代码 (Trial 6)

Ship-quality 代码, 对应根目录 `README.md` 的 Trial 6 方案.

- Warm 6.14s @ 1K, 60.26s @ 2K, 303.18s @ 4K
- PSNR 43.10 dB vs CPU fp32 @ 1K
- 架构: `torch.compile(backend="neuron")` 作用于 UNet 内 16 个 `Transformer2DModel`, 其他模块保持 eager. LoRA 层由 `DeModLoRALinearAttr` / `DeModLoRAConv2dAttr` 替换 peft 的 `LoraLayer` (代数等价 fold, max|diff| < 1e-14).

## 目录结构

```
src/
├── modules/                          # 自定义 nn.Module
│   ├── de_mod_lora.py                # DeModLoRALinear / DeModLoRAConv2d (forward-arg de_mod)
│   ├── de_mod_lora_attr.py           # *Attr 变体 (读 self.de_mod 属性), + replace_lora_modules_in_unet()
│   ├── s3diff_attention.py           # NeuronS3DiffAttention (可选 NKI attention_cte kernel)
│   ├── s3diff_transformer_block.py   # NeuronS3DiffBasicTransformerBlock
│   ├── s3diff_transformer_2d.py      # NeuronS3DiffTransformer2DModel
│   └── s3diff_resnet.py              # NeuronS3DiffResnetBlock2D
├── scripts/
│   ├── phase_bisect_hires.py         # ⭐ 生产脚本: 1K/2K/4K 都用这个
│   ├── phase_bisect.py               # bisect 各 compile scope (1K only)
│   ├── dump_unet_keys.py             # dump state_dict keys + LoRA targets
│   ├── extract_state_dicts.py        # 提取 UNet / VAE decoder state_dict 到 .pt
│   ├── convert_state_dict.py         # peft layout → Neuron layout (remove .default)
│   └── inspect_unet.py               # UNet 结构 + 参数量审计
├── tests/
│   ├── test_de_mod_lora.py           # unit test (random weight, max|diff| < 1e-14)
│   ├── test_de_mod_lora_load.py      # 从真实 state_dict 加载 verify
│   ├── test_blocks_r4.py             # Transformer2DModel + ResnetBlock block-level cos=1.0
│   ├── test_transformer_block.py     # BasicTransformerBlock cos=1.0
│   ├── test_full_unet_r4.py          # 全 UNet 替换 peft → DeMod, cos=1.0 vs diffusers
│   └── debug_transformer_block.py    # 分 stage (norm/attn1/attn2/ff) 定位 diff
└── data/
    ├── lora_targets.txt              # S3Diff target_modules_unet / vae 清单
    ├── unet_class_summary.txt        # UNet 含 1028 ModuleDict / 792 Linear / 261 Conv2d 等
    └── bisect_summary.md             # 7 trials 的 compile scope vs warm/PSNR 结果表
```

## 快速复现 (trn2.3xlarge, SDK 2.29 eager venv)

```bash
# 环境
source ~/workspace/native_venv/bin/activate

# 1K (输入 256×256 → 输出 1024×1024)
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path/to/LQ_256.png> --lq_size 256 \
    --output_image /tmp/sr_1k.png --num_inferences 3

# 2K (输入 512×512 → 输出 2048×2048)
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path/to/LQ_512.png> --lq_size 512 \
    --output_image /tmp/sr_2k.png --num_inferences 3

# 4K (输入 1024×1024 → 输出 4096×4096)
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path/to/LQ_1024.png> --lq_size 1024 \
    --output_image /tmp/sr_4k.png --num_inferences 2
```

`--scope t2d` = 编译 16 个 `Transformer2DModel.forward` (Trial 6 winner). 可选值:

| scope | 含义 | 1K warm |
|---|---|---|
| t2d | 编译 16 × Transformer2DModel | **6.14s** ⭐ |
| block | 编译 16 × BasicTransformerBlock | 6.60s |
| attn1 | 只编译 attn1 | 7.74s |
| ff | 只编译 FF | 7.78s |
| xblock | 编译 CrossAttnDown/Up block2D | **CRASH** (NCC_IRPX901, SDK bug) |

## 关键算法: DeModLoRA folded einsum

原 peft `my_lora_fwd` 每 LoRA site 做 `Linear → einsum → Linear → add`:
```python
_tmp = lora_A(dropout(x))                                  # Linear 1
_tmp = einsum('...lk,...kr->...lr', _tmp, self.de_mod)     # einsum
result = result + lora_B(_tmp) * scaling                    # Linear 2
```

折叠后 (Phase R C-1a 算法):
```python
a  = lora_A(x)                                              # Linear 1
bd = einsum('or,bkr->bok', lora_B.weight, de_mod)           # einsum (fold B+demod)
out_adapter = einsum('bok,...k->...o', bd, a)               # einsum (apply)
result = base(x) + out_adapter * scaling
```

**少一个 `Linear_dot` 节点**, 避免 neuronx-cc 的 `NCC_IRPX901` 编译 assert 在 `proj_in.lora_A` 上触发. 数值等价, `max|diff| = 1e-14` (fp64 floor).

## 依赖
- Python 3.12, torch 2.10, torch-neuronx 0.1.0+5e711e8 (eager SDK)
- neuronx-cc 2.24.8799.0+6f62ff7c
- diffusers 0.34.0, peft 0.19.1, transformers 4.57.3

## 相关资源
- 客户报告: `../README.md`
- 旧方案 (phase3 / phase_e / phase_r / phase_b) 存档于 `../backup/`
