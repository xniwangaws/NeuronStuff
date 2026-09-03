#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
from transformers import ConvNextConfig, OneFormerConfig
from transformers.models.oneformer.modeling_oneformer import (
    OneFormerForUniversalSegmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--converter-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def load_converter_module(path: str):
    spec = importlib.util.spec_from_file_location(
        "upstream_oneformer_converter",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load converter source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config() -> OneFormerConfig:
    backbone_config = ConvNextConfig(
        num_channels=3,
        patch_size=4,
        depths=[3, 3, 27, 3],
        hidden_sizes=[256, 512, 1024, 2048],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    id2label = {index: f"ade20k_{index}" for index in range(150)}
    label2id = {label: index for index, label in id2label.items()}
    return OneFormerConfig(
        backbone_config=backbone_config,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ignore_value=255,
        num_classes=150,
        num_queries=250,
        no_object_weight=0.1,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        contrastive_weight=0.5,
        contrastive_temperature=0.07,
        train_num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        init_std=0.02,
        init_xavier_std=1.0,
        layer_norm_eps=1e-5,
        is_training=False,
        use_auxiliary_loss=False,
        output_auxiliary_logits=False,
        strides=[4, 8, 16, 32],
        task_seq_len=77,
        max_seq_len=77,
        text_encoder_width=256,
        text_encoder_context_length=77,
        text_encoder_num_layers=6,
        text_encoder_vocab_size=49408,
        text_encoder_proj_layers=2,
        text_encoder_n_ctx=16,
        conv_dim=256,
        mask_dim=256,
        hidden_dim=256,
        norm="GN",
        encoder_layers=6,
        encoder_feedforward_dim=1024,
        decoder_layers=10,
        use_task_norm=True,
        num_attention_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pre_norm=False,
        enforce_input_proj=False,
        query_dec_layers=2,
        common_stride=4,
        id2label=id2label,
        label2id=label2id,
    )


class StateDictModel(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self._source_state_dict = state_dict

    def state_dict(self, *args, **kwargs):
        return self._source_state_dict.copy()


def make_converter_class(module):
    base_class = module.OriginalOneFormerCheckpointToOursConverter

    class ConvNextOneFormerConverter(base_class):
        def replace_convnext_backbone(
            self,
            dst_state_dict,
            src_state_dict,
            config,
        ) -> None:
            dst_prefix = "pixel_level_module.encoder"
            src_prefix = "backbone"
            renamed_keys = [
                (
                    f"{src_prefix}.downsample_layers.0.0.weight",
                    f"{dst_prefix}.embeddings.patch_embeddings.weight",
                ),
                (
                    f"{src_prefix}.downsample_layers.0.0.bias",
                    f"{dst_prefix}.embeddings.patch_embeddings.bias",
                ),
                (
                    f"{src_prefix}.downsample_layers.0.1.weight",
                    f"{dst_prefix}.embeddings.layernorm.weight",
                ),
                (
                    f"{src_prefix}.downsample_layers.0.1.bias",
                    f"{dst_prefix}.embeddings.layernorm.bias",
                ),
            ]

            for stage_index, depth in enumerate(config.backbone_config.depths):
                if stage_index > 0:
                    for component_index in range(2):
                        for suffix in ("weight", "bias"):
                            renamed_keys.append(
                                (
                                    (
                                        f"{src_prefix}.downsample_layers."
                                        f"{stage_index}.{component_index}.{suffix}"
                                    ),
                                    (
                                        f"{dst_prefix}.encoder.stages."
                                        f"{stage_index}.downsampling_layer."
                                        f"{component_index}.{suffix}"
                                    ),
                                )
                            )

                for block_index in range(depth):
                    src_block = (
                        f"{src_prefix}.stages.{stage_index}.{block_index}"
                    )
                    dst_block = (
                        f"{dst_prefix}.encoder.stages.{stage_index}."
                        f"layers.{block_index}"
                    )
                    renamed_keys.append(
                        (
                            f"{src_block}.gamma",
                            f"{dst_block}.layer_scale_parameter",
                        )
                    )
                    for src_name, dst_name in (
                        ("dwconv", "dwconv"),
                        ("norm", "layernorm"),
                        ("pwconv1", "pwconv1"),
                        ("pwconv2", "pwconv2"),
                    ):
                        for suffix in ("weight", "bias"):
                            renamed_keys.append(
                                (
                                    f"{src_block}.{src_name}.{suffix}",
                                    f"{dst_block}.{dst_name}.{suffix}",
                                )
                            )

                for suffix in ("weight", "bias"):
                    renamed_keys.append(
                        (
                            f"{src_prefix}.norm{stage_index}.{suffix}",
                            (
                                f"{dst_prefix}.hidden_states_norms."
                                f"stage{stage_index + 1}.{suffix}"
                            ),
                        )
                    )

            self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

        def replace_dinat_backbone(
            self,
            dst_state_dict,
            src_state_dict,
            config,
        ) -> None:
            self.replace_convnext_backbone(
                dst_state_dict,
                src_state_dict,
                config,
            )

        def convert_strict(self, oneformer):
            dst_state_dict = module.TrackedStateDict(oneformer.state_dict())
            src_state_dict = self.original_model.state_dict()
            self.replace_pixel_module(
                dst_state_dict,
                src_state_dict,
                is_swin=False,
            )
            self.replace_transformer_module(
                dst_state_dict,
                src_state_dict,
            )
            self.replace_task_mlp(dst_state_dict, src_state_dict)

            missed_keys = sorted(dst_state_dict.diff())
            leftover_keys = sorted(src_state_dict)
            allowed_leftover_prefixes = (
                "criterion.",
                "text_encoder.",
                "text_projector.",
                "prompt_ctx.",
            )
            unexpected_leftovers = [
                key
                for key in leftover_keys
                if not key.startswith(allowed_leftover_prefixes)
            ]
            if missed_keys:
                raise ValueError(
                    "Unmapped destination keys:\n" + "\n".join(missed_keys)
                )
            if unexpected_leftovers:
                raise ValueError(
                    "Unexpected source keys:\n"
                    + "\n".join(unexpected_leftovers)
                )
            oneformer.load_state_dict(dst_state_dict.copy())
            return oneformer, {
                "missed_destination_keys": missed_keys,
                "allowed_leftover_source_keys": leftover_keys,
                "unexpected_leftover_source_keys": unexpected_leftovers,
            }

    return ConvNextOneFormerConverter


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    converter_module = load_converter_module(args.converter_source)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    source_state_dict = checkpoint["model"]
    config = build_config()
    model = OneFormerForUniversalSegmentation(config).eval()

    converter_class = make_converter_class(converter_module)
    converter = converter_class(
        StateDictModel(source_state_dict),
        config,
    )
    model.model, conversion_report = converter.convert_strict(model.model)
    model.save_pretrained(output_dir, safe_serialization=False)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_dict_tensor_count": len(model.state_dict()),
        "checkpoint_iteration": checkpoint.get("iteration"),
        **conversion_report,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
