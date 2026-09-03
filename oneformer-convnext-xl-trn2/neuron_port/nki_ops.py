import torch
from torch import Tensor, nn


MSDA_SPATIAL_SHAPES = ((20, 20), (40, 40), (80, 80))
MSDA_LEVEL_START_INDEX = (0, 400, 2000)


def load_msda_nki_kernel():
    try:
        from nkilib.experimental.deformable_attention.ms_deformable_attention import (
            ms_deformable_attention,
        )
    except ImportError as error:
        raise RuntimeError(
            "The NKI Library MSDeformableAttention kernel is unavailable. "
            "Install nki_library with neuronx-cc>=2.26 and nki>=0.5."
        ) from error
    return ms_deformable_attention


def sanitize_far_oob_samples(
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Make far-outside samples safe for the NKI kernel.

    With align_corners=False, a sample has no non-zero bilinear contribution
    once x is outside (-0.5 / width, 1 + 0.5 / width), or y is outside the
    corresponding height range. Setting that sample's attention weight to
    zero and replacing its coordinate with 0.5 is exactly equivalent to zero
    padding, while avoiding unsupported far-outside indirect DMA addresses.
    """

    safe_locations = []
    safe_weights = []
    for level, (height, width) in enumerate(MSDA_SPATIAL_SHAPES):
        level_locations = sampling_locations[:, :, :, level, :, :]
        level_weights = attention_weights[:, :, :, level, :]
        x = level_locations[..., 0]
        y = level_locations[..., 1]
        valid = (
            (x > (-0.5 / width))
            & (x < (1.0 + 0.5 / width))
            & (y > (-0.5 / height))
            & (y < (1.0 + 0.5 / height))
        )
        safe_locations.append(
            torch.where(
                valid.unsqueeze(-1),
                level_locations,
                torch.full_like(level_locations, 0.5),
            )
        )
        safe_weights.append(
            level_weights * valid.to(level_weights.dtype)
        )
    return (
        torch.stack(safe_locations, dim=3),
        torch.stack(safe_weights, dim=3),
    )


class NkiMsdaOutputProjectionCore(nn.Module):
    def __init__(self, attention: nn.Module, lnc: int):
        super().__init__()
        self.output_proj = attention.output_proj
        self.lnc = lnc
        self.kernel = load_msda_nki_kernel()

    def forward(
        self,
        value: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        safe_locations, safe_weights = sanitize_far_oob_samples(
            sampling_locations,
            attention_weights,
        )
        attention_output = self.kernel[self.lnc](
            value.to(torch.bfloat16),
            MSDA_SPATIAL_SHAPES,
            MSDA_LEVEL_START_INDEX,
            safe_locations.to(torch.float32),
            safe_weights.to(torch.bfloat16),
            value_layout="BLNC",
            sampling_locations_layout="BQHLP2",
            align_corners=False,
            padding_mode="zeros",
        )
        return self.output_proj(attention_output.to(torch.float32))


class NkiFusedPixelDecoderEncoderLayerCore(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        reference_points: Tensor,
        lnc: int,
    ):
        super().__init__()
        attention = layer.self_attn
        self.value_proj = attention.value_proj
        self.sampling_offsets = attention.sampling_offsets
        self.attention_weights = attention.attention_weights
        self.output_proj = attention.output_proj
        self.self_attn_layer_norm = layer.self_attn_layer_norm
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2
        self.final_layer_norm = layer.final_layer_norm
        self.num_heads = attention.n_heads
        self.num_levels = attention.n_levels
        self.num_points = attention.n_points
        self.head_dim = attention.d_model // attention.n_heads
        self.lnc = lnc
        self.kernel = load_msda_nki_kernel()
        self.register_buffer("reference_points", reference_points)
        self.register_buffer(
            "offset_normalizer",
            torch.tensor(
                [[20, 20], [40, 40], [80, 80]],
                dtype=reference_points.dtype,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
    ) -> Tensor:
        query = hidden_states + position_embeddings
        batch_size, sequence_length, _ = hidden_states.shape
        value = self.value_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = torch.softmax(
            attention_weights,
            dim=-1,
        ).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        sampling_locations = (
            self.reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / self.offset_normalizer[None, None, None, :, None, :]
        )
        safe_locations, safe_weights = sanitize_far_oob_samples(
            sampling_locations,
            attention_weights,
        )
        attention_output = self.kernel[self.lnc](
            value.to(torch.bfloat16),
            MSDA_SPATIAL_SHAPES,
            MSDA_LEVEL_START_INDEX,
            safe_locations.to(torch.float32),
            safe_weights.to(torch.bfloat16),
            value_layout="BLNC",
            sampling_locations_layout="BQHLP2",
            align_corners=False,
            padding_mode="zeros",
        )
        attention_output = self.output_proj(
            attention_output.to(torch.float32)
        )
        hidden_states = self.self_attn_layer_norm(
            hidden_states + attention_output
        )
        residual = hidden_states
        hidden_states = torch.relu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return self.final_layer_norm(residual + hidden_states)


class NkiFusedPixelDecoderEncoderStackCore(nn.Module):
    """Run all pixel-decoder encoder layers in one Neuron graph."""

    def __init__(
        self,
        layers: nn.ModuleList,
        reference_points: Tensor,
        lnc: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NkiFusedPixelDecoderEncoderLayerCore(
                    layer,
                    reference_points,
                    lnc,
                )
                for layer in layers
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return hidden_states
