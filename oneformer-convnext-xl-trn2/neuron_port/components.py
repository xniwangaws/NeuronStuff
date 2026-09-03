import torch
from torch import Tensor, nn


class BackboneCore(nn.Module):
    def __init__(self, oneformer_core: nn.Module):
        super().__init__()
        self.encoder = oneformer_core.pixel_level_module.encoder

    def forward(
        self,
        pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_maps = self.encoder(pixel_values).feature_maps
        return (
            feature_maps[0],
            feature_maps[1],
            feature_maps[2],
            feature_maps[3],
        )


class PixelDecoderCore(nn.Module):
    def __init__(self, oneformer_core: nn.Module):
        super().__init__()
        self.decoder = oneformer_core.pixel_level_module.decoder

    def forward(
        self,
        feature_4: Tensor,
        feature_8: Tensor,
        feature_16: Tensor,
        feature_32: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.decoder(
            [
                feature_4,
                feature_8,
                feature_16,
                feature_32,
            ],
            output_hidden_states=False,
        )
        return (
            output.multi_scale_features[0],
            output.multi_scale_features[1],
            output.multi_scale_features[2],
            output.mask_features,
        )


class PixelLevelCore(nn.Module):
    def __init__(self, oneformer_core: nn.Module):
        super().__init__()
        self.pixel_level_module = oneformer_core.pixel_level_module

    def forward(
        self,
        pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.pixel_level_module(
            pixel_values,
            output_hidden_states=False,
        )
        return (
            output.decoder_features[0],
            output.decoder_features[1],
            output.decoder_features[2],
            output.decoder_last_feature,
        )


class TaskEncoderCore(nn.Module):
    def __init__(self, oneformer_core: nn.Module):
        super().__init__()
        self.task_encoder = oneformer_core.task_encoder
        self.output_dtype = oneformer_core.dtype

    def forward(self, task_inputs: Tensor) -> Tensor:
        return self.task_encoder(task_inputs.to(self.output_dtype))


class TransformerCore(nn.Module):
    def __init__(self, oneformer_core: nn.Module):
        super().__init__()
        self.transformer_module = oneformer_core.transformer_module

    def forward(
        self,
        feature_16: Tensor,
        feature_32: Tensor,
        feature_64: Tensor,
        mask_features: Tensor,
        task_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = self.transformer_module(
            multi_scale_features=[
                feature_16,
                feature_32,
                feature_64,
            ],
            mask_features=mask_features,
            task_token=task_token,
            output_attentions=False,
        )
        return output.prediction_class, output.prediction_masks
