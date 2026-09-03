from pathlib import Path

import torch
from torch import Tensor, nn

from .ops import multi_scale_deformable_attention_bilinear


def _trace_friendly_reference_points(
    spatial_shapes: Tensor,
    valid_ratios: Tensor,
    device: torch.device,
) -> Tensor:
    """Static-shape equivalent of OneFormer's linspace-based implementation."""

    reference_points_list = []
    for level, (height, width) in enumerate(spatial_shapes):
        height_value = int(height)
        width_value = int(width)
        ref_y, ref_x = torch.meshgrid(
            torch.arange(
                height_value,
                dtype=valid_ratios.dtype,
                device=device,
            )
            + 0.5,
            torch.arange(
                width_value,
                dtype=valid_ratios.dtype,
                device=device,
            )
            + 0.5,
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / (
            valid_ratios[:, None, level, 1] * height_value
        )
        ref_x = ref_x.reshape(-1)[None] / (
            valid_ratios[:, None, level, 0] * width_value
        )
        reference_points_list.append(torch.stack((ref_x, ref_y), -1))

    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None] * valid_ratios[:, None]


def _trace_friendly_decoder_layer_forward(
    self: nn.Module,
    index: int,
    output: Tensor,
    multi_stage_features: list[Tensor],
    multi_stage_positional_embeddings: list[Tensor],
    attention_mask: Tensor | None = None,
    query_embeddings: Tensor | None = None,
    output_attentions: bool = False,
) -> tuple[Tensor, ...]:
    """Equivalent decoder layer without dynamic nonzero/index_put operations."""

    level_index = index % self.num_feature_levels
    if attention_mask is not None:
        all_masked = attention_mask.sum(-1) == attention_mask.shape[-1]
        attention_mask = attention_mask & ~all_masked.unsqueeze(-1)

    output, cross_attn_weights = self.cross_attn(
        output,
        multi_stage_features[level_index],
        memory_mask=attention_mask,
        memory_key_padding_mask=None,
        pos=multi_stage_positional_embeddings[level_index],
        query_pos=query_embeddings,
    )
    output, self_attn_weights = self.self_attn(
        output,
        output_mask=None,
        output_key_padding_mask=None,
        query_pos=query_embeddings,
    )
    output = self.ffn(output)

    outputs = (output,)
    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)
    return outputs


def patch_oneformer_for_neuron(modeling_oneformer: object) -> None:
    """Install fixed-shape, Neuron-trace-friendly OneFormer operations."""

    modeling_oneformer.multi_scale_deformable_attention = (
        multi_scale_deformable_attention_bilinear
    )
    modeling_oneformer.OneFormerPixelDecoderEncoderOnly.get_reference_points = (
        staticmethod(_trace_friendly_reference_points)
    )
    modeling_oneformer.OneFormerTransformerDecoderLayer.forward = (
        _trace_friendly_decoder_layer_forward
    )


class OneFormerInferenceCore(nn.Module):
    """Inference-only OneFormer core with a trace-friendly tensor tuple output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.core = model.model
        self.core.is_training = False
        self.core.config.is_training = False
        self.core.config.output_hidden_states = False
        self.core.config.output_attentions = False
        self.core.config.use_auxiliary_loss = False
        self.core.transformer_module.decoder.use_auxiliary_loss = False

    def forward(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pixel_output = self.core.pixel_level_module(
            pixel_values,
            output_hidden_states=False,
        )
        task_token = self.core.task_encoder(task_inputs.to(self.core.dtype))
        transformer_output = self.core.transformer_module(
            multi_scale_features=pixel_output.decoder_features,
            mask_features=pixel_output.decoder_last_feature,
            task_token=task_token,
            output_attentions=False,
        )
        return (
            transformer_output.prediction_class,
            transformer_output.prediction_masks,
        )


def load_oneformer(
    model_id: str,
    cache_dir: str | Path,
    use_custom_grid_sample: bool,
    local_files_only: bool = False,
) -> tuple[nn.Module, nn.Module]:
    from transformers import OneFormerForUniversalSegmentation
    from transformers.models.oneformer import modeling_oneformer

    if use_custom_grid_sample:
        patch_oneformer_for_neuron(modeling_oneformer)

    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        use_safetensors=False,
    )
    model.eval()
    wrapper = OneFormerInferenceCore(model)
    wrapper.eval()
    return model, wrapper
