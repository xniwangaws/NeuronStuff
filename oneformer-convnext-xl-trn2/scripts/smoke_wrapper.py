#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from torch import nn
from transformers import OneFormerConfig, OneFormerForUniversalSegmentation
from transformers.models.oneformer import modeling_oneformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import OneFormerInferenceCore
from neuron_port.ops import multi_scale_deformable_attention_bilinear


def make_config() -> OneFormerConfig:
    config = OneFormerConfig(
        text_encoder_vocab_size=99,
        hidden_size=64,
        num_queries=10,
        num_labels=4,
        encoder_feedforward_dim=32,
        dim_feedforward=64,
        encoder_layers=2,
        decoder_layers=2,
    )
    config.backbone_config.embed_dim = 16
    config.backbone_config.depths = [1, 1, 1, 1]
    config.backbone_config.hidden_size = 16
    config.backbone_config.num_channels = 3
    config.backbone_config.num_heads = [1, 1, 2, 2]
    config.backbone = None
    config.hidden_dim = 64
    config.mask_dim = 64
    config.conv_dim = 64
    config.text_encoder_width = 64
    config.task_seq_len = 77
    config.max_seq_len = 77
    config.text_encoder_context_length = 77
    config.text_encoder_n_ctx = 4
    config.is_training = False
    config.use_auxiliary_loss = False
    return config


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return nn.functional.cosine_similarity(
        actual.flatten(),
        expected.flatten(),
        dim=0,
    ).item()


def main() -> None:
    torch.manual_seed(0)
    model = OneFormerForUniversalSegmentation(make_config()).eval()
    wrapper = OneFormerInferenceCore(model).eval()
    pixel_values = torch.randn(1, 3, 256, 256)
    task_inputs = torch.randint(0, 99, (1, 77))

    with torch.no_grad():
        raw_outputs = wrapper(pixel_values, task_inputs)

    modeling_oneformer.multi_scale_deformable_attention = (
        multi_scale_deformable_attention_bilinear
    )
    with torch.no_grad():
        custom_outputs = wrapper(pixel_values, task_inputs)

    class_cosine = cosine(custom_outputs[0], raw_outputs[0])
    mask_cosine = cosine(custom_outputs[1], raw_outputs[1])
    class_max_abs = (
        custom_outputs[0] - raw_outputs[0]
    ).abs().max().item()
    mask_max_abs = (
        custom_outputs[1] - raw_outputs[1]
    ).abs().max().item()
    print(
        {
            "class_shape": list(raw_outputs[0].shape),
            "mask_shape": list(raw_outputs[1].shape),
            "class_cosine": class_cosine,
            "mask_cosine": mask_cosine,
            "class_max_abs": class_max_abs,
            "mask_max_abs": mask_max_abs,
        }
    )

    if class_max_abs > 1e-5 or mask_max_abs > 1e-5:
        raise ValueError("Custom attention path diverged from the raw path")


if __name__ == "__main__":
    main()
