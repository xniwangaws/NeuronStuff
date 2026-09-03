import torch
from torch import Tensor, nn


def _gather_2d(input_tensor: Tensor, x_index: Tensor, y_index: Tensor) -> Tensor:
    batch_size, channels, height, width = input_tensor.shape
    output_height, output_width = x_index.shape[-2:]

    valid = (
        (x_index >= 0)
        & (x_index < width)
        & (y_index >= 0)
        & (y_index < height)
    )
    x_safe = x_index.clamp(0, width - 1).to(torch.int64)
    y_safe = y_index.clamp(0, height - 1).to(torch.int64)
    flat_index = (y_safe * width + x_safe).reshape(batch_size, 1, -1)
    flat_index = flat_index.expand(batch_size, channels, output_height * output_width)

    values = torch.gather(
        input_tensor.reshape(batch_size, channels, height * width),
        dim=2,
        index=flat_index,
    )
    values = values.reshape(batch_size, channels, output_height, output_width)
    return values * valid.unsqueeze(1).to(values.dtype)


def bilinear_grid_sample_2d(input_tensor: Tensor, grid: Tensor) -> Tensor:
    """Neuron-friendly equivalent of grid_sample(..., align_corners=False).

    The implementation is limited to the mode used by OneFormer:
    bilinear interpolation with zero padding.
    """

    _, _, height, width = input_tensor.shape
    grid_x = grid[..., 0]
    grid_y = grid[..., 1]

    source_x = ((grid_x + 1.0) * width - 1.0) * 0.5
    source_y = ((grid_y + 1.0) * height - 1.0) * 0.5

    x0 = torch.floor(source_x)
    y0 = torch.floor(source_y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    top_left = _gather_2d(input_tensor, x0, y0)
    top_right = _gather_2d(input_tensor, x1, y0)
    bottom_left = _gather_2d(input_tensor, x0, y1)
    bottom_right = _gather_2d(input_tensor, x1, y1)

    weight_top_left = (x1 - source_x) * (y1 - source_y)
    weight_top_right = (source_x - x0) * (y1 - source_y)
    weight_bottom_left = (x1 - source_x) * (source_y - y0)
    weight_bottom_right = (source_x - x0) * (source_y - y0)

    return (
        top_left * weight_top_left.unsqueeze(1)
        + top_right * weight_top_right.unsqueeze(1)
        + bottom_left * weight_bottom_left.unsqueeze(1)
        + bottom_right * weight_bottom_right.unsqueeze(1)
    )


def multi_scale_deformable_attention_bilinear(
    value: Tensor,
    value_spatial_shapes: Tensor | list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """OneFormer deformable attention with the custom bilinear sampler."""

    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split(
        [height * width for height, width in value_spatial_shapes],
        dim=1,
    )
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []

    for level_id, (height, width) in enumerate(value_spatial_shapes):
        value_level = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        sampling_grid_level = (
            sampling_grids[:, :, :, level_id]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampling_value_list.append(
            bilinear_grid_sample_2d(value_level, sampling_grid_level)
        )

    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )
    output = (
        (
            torch.stack(sampling_value_list, dim=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class RawGridSample(nn.Module):
    def forward(self, input_tensor: Tensor, grid: Tensor) -> Tensor:
        return nn.functional.grid_sample(
            input_tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class CustomGridSample(nn.Module):
    def forward(self, input_tensor: Tensor, grid: Tensor) -> Tensor:
        return bilinear_grid_sample_2d(input_tensor, grid)
