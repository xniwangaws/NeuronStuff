"""NKI prototypes for fused ConvNeXt inference blocks."""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import Tensor, nn

from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from nkilib.core.utils.tensor_view import TensorView


_CHANNELS = 256
_EXPANDED_CHANNELS = 1024
_HEIGHT = 160
_WIDTH = 160
_SPATIAL_SIZE = _HEIGHT * _WIDTH
_PARTITION_TILE = 128
_SPATIAL_TILE = 512
_CHANNEL_TILES = _CHANNELS // _PARTITION_TILE
_EXPANDED_TILES = _EXPANDED_CHANNELS // _PARTITION_TILE
_SPATIAL_TILES = _SPATIAL_SIZE // _SPATIAL_TILE
_DW_KERNEL = 7
_DW_PADDING = _DW_KERNEL // 2
_HEIGHT_TILE = 16
_HEIGHT_TILES = _HEIGHT // _HEIGHT_TILE
_HEIGHT_TILE_SPATIAL = _HEIGHT_TILE * _WIDTH
_PADDED_HEIGHT_TILE = _HEIGHT_TILE + 2 * _DW_PADDING
_PADDED_WIDTH = _WIDTH + 2 * _DW_PADDING
_PADDED_TILE_SPATIAL = _PADDED_HEIGHT_TILE * _PADDED_WIDTH


@nki.jit
def convnext_stage0_post_dw_fused(
    hidden_ref: nl.ndarray,
    residual_ref: nl.ndarray,
    norm_weight_ref: nl.ndarray,
    norm_bias_ref: nl.ndarray,
    pw1_weight_ref: nl.ndarray,
    pw1_bias_ref: nl.ndarray,
    pw2_weight_ref: nl.ndarray,
    pw2_bias_ref: nl.ndarray,
    layer_scale_ref: nl.ndarray,
) -> nl.ndarray:
    """Fuse the post-depthwise portion of a stage-0 ConvNeXt-XL block.

    Dimensions:
        C: 256 channels
        E: 1024 expanded channels
        S: 160 * 160 spatial positions

    Args:
        hidden_ref: [1, C, 160, 160] BF16 depthwise-convolution output.
        residual_ref: [1, C, 160, 160] BF16 block residual.
        norm_weight_ref: [2, 128, 1] FP32 LayerNorm gamma.
        norm_bias_ref: [2, 128, 1] FP32 LayerNorm beta.
        pw1_weight_ref: [8, 2, 128, 128] BF16 blocked transposed weight.
        pw1_bias_ref: [8, 128, 1] FP32 pointwise-1 bias.
        pw2_weight_ref: [2, 8, 128, 128] BF16 blocked transposed weight.
        pw2_bias_ref: [2, 128, 1] FP32 pointwise-2 bias.
        layer_scale_ref: [2, 128, 1] FP32 ConvNeXt layer scale.

    Returns:
        [1, C, 160, 160] BF16 block output.

    Notes:
        - Specialized for the highest-resolution ConvNeXt-XL stage.
        - LNC2 shards the 50 spatial tiles evenly across two physical cores.
        - Pointwise weights remain resident in SBUF across spatial tiles.
        - LayerNorm statistics and Tensor Engine accumulation use FP32.
    """

    kernel_assert(
        hidden_ref.shape == (1, _CHANNELS, _HEIGHT, _WIDTH),
        f"hidden_ref must be {(1, _CHANNELS, _HEIGHT, _WIDTH)}",
    )
    kernel_assert(
        residual_ref.shape == hidden_ref.shape,
        "residual_ref must match hidden_ref",
    )
    kernel_assert(
        norm_weight_ref.shape == (_CHANNEL_TILES, _PARTITION_TILE, 1),
        "invalid norm_weight_ref shape",
    )
    kernel_assert(
        pw1_weight_ref.shape
        == (
            _EXPANDED_TILES,
            _CHANNEL_TILES,
            _PARTITION_TILE,
            _PARTITION_TILE,
        ),
        "invalid pw1_weight_ref shape",
    )
    kernel_assert(
        pw2_weight_ref.shape
        == (
            _CHANNEL_TILES,
            _EXPANDED_TILES,
            _PARTITION_TILE,
            _PARTITION_TILE,
        ),
        "invalid pw2_weight_ref shape",
    )

    program_count = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    kernel_assert(program_count == 2, "this kernel requires an LNC2 launch")
    kernel_assert(
        _SPATIAL_TILES % program_count == 0,
        "spatial tiles must divide evenly across LNC programs",
    )
    tiles_per_program = _SPATIAL_TILES // program_count

    output_ref = nl.ndarray(
        hidden_ref.shape,
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    hidden_2d = hidden_ref.reshape((_CHANNELS, _SPATIAL_SIZE))
    residual_2d = residual_ref.reshape((_CHANNELS, _SPATIAL_SIZE))
    output_2d = output_ref.reshape((_CHANNELS, _SPATIAL_SIZE))

    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="convnext_stage0_post_dw")

    ones_fp32 = sbm.alloc_stack(
        (_PARTITION_TILE, 1),
        dtype=nl.float32,
        name="ones_fp32",
    )
    nisa.memset(dst=ones_fp32, value=1.0)

    norm_weights = []
    norm_biases = []
    layer_scales = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        norm_weight = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"norm_weight_{channel_tile}",
        )
        norm_bias = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"norm_bias_{channel_tile}",
        )
        layer_scale = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"layer_scale_{channel_tile}",
        )
        nisa.dma_copy(
            dst=norm_weight,
            src=norm_weight_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        nisa.dma_copy(
            dst=norm_bias,
            src=norm_bias_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        nisa.dma_copy(
            dst=layer_scale,
            src=layer_scale_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        norm_weights.append(norm_weight)
        norm_biases.append(norm_bias)
        layer_scales.append(layer_scale)

    pw1_weights = []
    pw1_biases = []
    for output_tile in nl.affine_range(_EXPANDED_TILES):
        weight_row = []
        for input_tile in nl.affine_range(_CHANNEL_TILES):
            weight_tile = sbm.alloc_stack(
                (_PARTITION_TILE, _PARTITION_TILE),
                dtype=nl.bfloat16,
                name=f"pw1_weight_{output_tile}_{input_tile}",
            )
            nisa.dma_copy(
                dst=weight_tile,
                src=pw1_weight_ref[
                    output_tile,
                    input_tile,
                    0:_PARTITION_TILE,
                    0:_PARTITION_TILE,
                ],
            )
            weight_row.append(weight_tile)
        bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"pw1_bias_{output_tile}",
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=pw1_bias_ref[output_tile, 0:_PARTITION_TILE, 0:1],
        )
        pw1_weights.append(weight_row)
        pw1_biases.append(bias_tile)

    pw2_weights = []
    pw2_biases = []
    for output_tile in nl.affine_range(_CHANNEL_TILES):
        weight_row = []
        for input_tile in nl.affine_range(_EXPANDED_TILES):
            weight_tile = sbm.alloc_stack(
                (_PARTITION_TILE, _PARTITION_TILE),
                dtype=nl.bfloat16,
                name=f"pw2_weight_{output_tile}_{input_tile}",
            )
            nisa.dma_copy(
                dst=weight_tile,
                src=pw2_weight_ref[
                    output_tile,
                    input_tile,
                    0:_PARTITION_TILE,
                    0:_PARTITION_TILE,
                ],
            )
            weight_row.append(weight_tile)
        bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"pw2_bias_{output_tile}",
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=pw2_bias_ref[output_tile, 0:_PARTITION_TILE, 0:1],
        )
        pw2_weights.append(weight_row)
        pw2_biases.append(bias_tile)

    for local_spatial_tile in nl.sequential_range(tiles_per_program):
        spatial_tile = program_id * tiles_per_program + local_spatial_tile
        spatial_start = spatial_tile * _SPATIAL_TILE
        spatial_end = spatial_start + _SPATIAL_TILE
        sbm.open_scope()

        input_tiles = []
        for channel_tile in nl.affine_range(_CHANNEL_TILES):
            channel_start = channel_tile * _PARTITION_TILE
            channel_end = channel_start + _PARTITION_TILE
            input_tile = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
            nisa.dma_copy(
                dst=input_tile,
                src=hidden_2d[
                    channel_start:channel_end,
                    spatial_start:spatial_end,
                ],
            )
            input_tiles.append(input_tile)

        sum_psum = nl.ndarray(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        square_sum_psum = nl.ndarray(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        input_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        squared_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        for channel_tile in nl.affine_range(_CHANNEL_TILES):
            nisa.tensor_copy(
                dst=input_fp32,
                src=input_tiles[channel_tile],
            )
            nisa.activation(
                dst=squared_fp32,
                data=input_fp32,
                op=nl.square,
            )
            nisa.nc_matmul(
                dst=sum_psum,
                stationary=ones_fp32,
                moving=input_fp32,
                accumulate=(channel_tile > 0),
            )
            nisa.nc_matmul(
                dst=square_sum_psum,
                stationary=ones_fp32,
                moving=squared_fp32,
                accumulate=(channel_tile > 0),
            )

        mean = sbm.alloc_stack(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        mean_square = sbm.alloc_stack(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        square_mean = sbm.alloc_stack(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        variance = sbm.alloc_stack(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        invstd = sbm.alloc_stack(
            (1, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        nisa.tensor_copy(dst=mean, src=sum_psum)
        nisa.tensor_copy(dst=mean_square, src=square_sum_psum)
        nisa.tensor_scalar(
            dst=mean,
            data=mean,
            op0=nl.multiply,
            operand0=(1.0 / _CHANNELS),
        )
        nisa.tensor_scalar(
            dst=mean_square,
            data=mean_square,
            op0=nl.multiply,
            operand0=(1.0 / _CHANNELS),
        )
        nisa.activation(dst=square_mean, data=mean, op=nl.square)
        nisa.tensor_tensor(
            dst=variance,
            data1=mean_square,
            data2=square_mean,
            op=nl.subtract,
        )
        nisa.tensor_scalar(
            dst=variance,
            data=variance,
            op0=nl.maximum,
            operand0=0.0,
        )
        nisa.tensor_scalar(
            dst=variance,
            data=variance,
            op0=nl.add,
            operand0=1e-6,
        )
        nisa.activation(dst=invstd, data=variance, op=nl.rsqrt)

        mean_broadcast = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        invstd_broadcast = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        stream_shuffle_broadcast(src=mean, dst=mean_broadcast)
        stream_shuffle_broadcast(src=invstd, dst=invstd_broadcast)

        normalized_tiles = []
        centered = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        normalized_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        for channel_tile in nl.affine_range(_CHANNEL_TILES):
            nisa.tensor_copy(
                dst=input_fp32,
                src=input_tiles[channel_tile],
            )
            nisa.tensor_tensor(
                dst=centered,
                data1=input_fp32,
                data2=mean_broadcast,
                op=nl.subtract,
            )
            nisa.tensor_tensor(
                dst=normalized_fp32,
                data1=centered,
                data2=invstd_broadcast,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=normalized_fp32,
                data1=normalized_fp32,
                data2=TensorView(norm_weights[channel_tile])
                .broadcast(dim=1, size=_SPATIAL_TILE)
                .get_view(),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=normalized_fp32,
                data1=normalized_fp32,
                data2=TensorView(norm_biases[channel_tile])
                .broadcast(dim=1, size=_SPATIAL_TILE)
                .get_view(),
                op=nl.add,
            )
            normalized_bf16 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
            nisa.tensor_copy(
                dst=normalized_bf16,
                src=normalized_fp32,
            )
            normalized_tiles.append(normalized_bf16)

        expanded_tiles = []
        for output_tile in nl.affine_range(_EXPANDED_TILES):
            expanded_psum = nl.ndarray(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            for input_tile in nl.affine_range(_CHANNEL_TILES):
                nisa.nc_matmul(
                    dst=expanded_psum,
                    stationary=pw1_weights[output_tile][input_tile],
                    moving=normalized_tiles[input_tile],
                    accumulate=(input_tile > 0),
                )
            expanded_fp32 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.float32,
            )
            expanded_bf16 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
            nisa.tensor_copy(dst=expanded_fp32, src=expanded_psum)
            nisa.tensor_tensor(
                dst=expanded_fp32,
                data1=expanded_fp32,
                data2=TensorView(pw1_biases[output_tile])
                .broadcast(dim=1, size=_SPATIAL_TILE)
                .get_view(),
                op=nl.add,
            )
            nisa.activation(
                dst=expanded_bf16,
                data=expanded_fp32,
                op=nl.gelu,
            )
            expanded_tiles.append(expanded_bf16)

        for output_tile in nl.affine_range(_CHANNEL_TILES):
            output_psum = nl.ndarray(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            for input_tile in nl.affine_range(_EXPANDED_TILES):
                nisa.nc_matmul(
                    dst=output_psum,
                    stationary=pw2_weights[output_tile][input_tile],
                    moving=expanded_tiles[input_tile],
                    accumulate=(input_tile > 0),
                )

            output_fp32 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.float32,
            )
            residual_bf16 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
            residual_fp32 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.float32,
            )
            output_bf16 = sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
            channel_start = output_tile * _PARTITION_TILE
            channel_end = channel_start + _PARTITION_TILE

            nisa.tensor_copy(dst=output_fp32, src=output_psum)
            nisa.tensor_tensor(
                dst=output_fp32,
                data1=output_fp32,
                data2=TensorView(pw2_biases[output_tile])
                .broadcast(dim=1, size=_SPATIAL_TILE)
                .get_view(),
                op=nl.add,
            )
            nisa.tensor_tensor(
                dst=output_fp32,
                data1=output_fp32,
                data2=TensorView(layer_scales[output_tile])
                .broadcast(dim=1, size=_SPATIAL_TILE)
                .get_view(),
                op=nl.multiply,
            )
            nisa.dma_copy(
                dst=residual_bf16,
                src=residual_2d[
                    channel_start:channel_end,
                    spatial_start:spatial_end,
                ],
            )
            nisa.tensor_copy(dst=residual_fp32, src=residual_bf16)
            nisa.tensor_tensor(
                dst=output_fp32,
                data1=output_fp32,
                data2=residual_fp32,
                op=nl.add,
            )
            nisa.tensor_copy(dst=output_bf16, src=output_fp32)
            nisa.dma_copy(
                dst=output_2d[
                    channel_start:channel_end,
                    spatial_start:spatial_end,
                ],
                src=output_bf16,
            )

        sbm.close_scope()

    sbm.close_scope()
    return output_ref


def _stage0_depthwise_height_tile(
    input_2d,
    output_2d,
    filter_tiles,
    bias_tiles,
    sbm,
    source_row_start,
    source_row_count,
    padded_row_start,
    output_spatial_start,
):
    """Compute one 16-row stage-0 depthwise tile entirely in SBUF."""

    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        channel_start = channel_tile * _PARTITION_TILE
        sbm.open_scope()
        padded_input = sbm.alloc_stack(
            (_PARTITION_TILE, _PADDED_TILE_SPATIAL),
            dtype=nl.bfloat16,
        )
        depthwise_output = sbm.alloc_stack(
            (_PARTITION_TILE, _HEIGHT_TILE_SPATIAL),
            dtype=nl.float32,
        )
        output_bf16 = sbm.alloc_stack(
            (_PARTITION_TILE, _HEIGHT_TILE_SPATIAL),
            dtype=nl.bfloat16,
        )

        nisa.memset(dst=padded_input, value=0)
        nisa.dma_copy(
            dst=padded_input.ap(
                pattern=[
                    [_PADDED_TILE_SPATIAL, _PARTITION_TILE],
                    [_PADDED_WIDTH, source_row_count],
                    [1, _WIDTH],
                ],
                offset=(
                    padded_row_start * _PADDED_WIDTH
                    + _DW_PADDING
                ),
            ),
            src=input_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [_WIDTH, source_row_count],
                    [1, _WIDTH],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + source_row_start * _WIDTH
                ),
            ),
        )

        nisa.memset(dst=depthwise_output, value=0)
        for kernel_row in nl.affine_range(_DW_KERNEL):
            for kernel_col in nl.affine_range(_DW_KERNEL):
                kernel_index = kernel_row * _DW_KERNEL + kernel_col
                nisa.scalar_tensor_tensor(
                    dst=depthwise_output.ap(
                        pattern=[
                            [_HEIGHT_TILE_SPATIAL, _PARTITION_TILE],
                            [_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=0,
                    ),
                    data=padded_input.ap(
                        pattern=[
                            [_PADDED_TILE_SPATIAL, _PARTITION_TILE],
                            [_PADDED_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=(
                            kernel_row * _PADDED_WIDTH
                            + kernel_col
                        ),
                    ),
                    op0=nl.multiply,
                    operand0=filter_tiles[channel_tile][
                        0:_PARTITION_TILE,
                        kernel_index : kernel_index + 1,
                    ],
                    op1=nl.add,
                    operand1=depthwise_output.ap(
                        pattern=[
                            [_HEIGHT_TILE_SPATIAL, _PARTITION_TILE],
                            [_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=0,
                    ),
                )

        nisa.tensor_tensor(
            dst=depthwise_output,
            data1=depthwise_output,
            data2=TensorView(bias_tiles[channel_tile])
            .broadcast(dim=1, size=_HEIGHT_TILE_SPATIAL)
            .get_view(),
            op=nl.add,
        )
        nisa.tensor_copy(dst=output_bf16, src=depthwise_output)
        nisa.dma_copy(
            dst=output_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [1, _HEIGHT_TILE_SPATIAL],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + output_spatial_start
                ),
            ),
            src=output_bf16,
        )
        sbm.close_scope()


@nki.jit
def convnext_stage0_depthwise_7x7(
    hidden_ref: nl.ndarray,
    filter_ref: nl.ndarray,
    bias_ref: nl.ndarray,
) -> nl.ndarray:
    """Specialized LNC2 7x7 depthwise convolution for stage-0."""

    kernel_assert(
        hidden_ref.shape == (1, _CHANNELS, _HEIGHT, _WIDTH),
        f"hidden_ref must be {(1, _CHANNELS, _HEIGHT, _WIDTH)}",
    )
    kernel_assert(
        filter_ref.shape
        == (_CHANNEL_TILES, _PARTITION_TILE, _DW_KERNEL * _DW_KERNEL),
        "invalid filter_ref shape",
    )
    kernel_assert(
        bias_ref.shape == (_CHANNEL_TILES, _PARTITION_TILE, 1),
        "invalid bias_ref shape",
    )

    program_count = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    kernel_assert(program_count == 2, "this kernel requires an LNC2 launch")
    kernel_assert(
        _HEIGHT_TILES == 10,
        "the edge/interior schedule assumes ten height tiles",
    )

    output_ref = nl.ndarray(
        hidden_ref.shape,
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    input_2d = hidden_ref.reshape((_CHANNELS, _SPATIAL_SIZE))
    output_2d = output_ref.reshape((_CHANNELS, _SPATIAL_SIZE))

    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="convnext_stage0_depthwise")
    filter_tiles = []
    bias_tiles = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        filter_tile = sbm.alloc_stack(
            (_PARTITION_TILE, _DW_KERNEL * _DW_KERNEL),
            dtype=nl.float32,
            name=f"dw_filter_{channel_tile}",
        )
        bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"dw_bias_{channel_tile}",
        )
        nisa.dma_copy(
            dst=filter_tile,
            src=filter_ref[
                channel_tile,
                0:_PARTITION_TILE,
                0 : _DW_KERNEL * _DW_KERNEL,
            ],
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=bias_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        filter_tiles.append(filter_tile)
        bias_tiles.append(bias_tile)

    # Core 0 handles the top edge tile and core 1 handles the bottom edge.
    edge_source_row = program_id * (_HEIGHT - (_HEIGHT_TILE + _DW_PADDING))
    edge_padded_row = (1 - program_id) * _DW_PADDING
    edge_output_spatial = (
        program_id * (_HEIGHT_TILES - 1) * _HEIGHT_TILE_SPATIAL
    )
    _stage0_depthwise_height_tile(
        input_2d,
        output_2d,
        filter_tiles,
        bias_tiles,
        sbm,
        source_row_start=edge_source_row,
        source_row_count=_HEIGHT_TILE + _DW_PADDING,
        padded_row_start=edge_padded_row,
        output_spatial_start=edge_output_spatial,
    )

    # The eight interior tiles need no boundary-dependent control flow.
    interior_tiles_per_program = (_HEIGHT_TILES - 2) // program_count
    for local_tile in nl.sequential_range(interior_tiles_per_program):
        height_tile = (
            1
            + program_id * interior_tiles_per_program
            + local_tile
        )
        source_row = height_tile * _HEIGHT_TILE - _DW_PADDING
        output_spatial = height_tile * _HEIGHT_TILE_SPATIAL
        _stage0_depthwise_height_tile(
            input_2d,
            output_2d,
            filter_tiles,
            bias_tiles,
            sbm,
            source_row_start=source_row,
            source_row_count=_PADDED_HEIGHT_TILE,
            padded_row_start=0,
            output_spatial_start=output_spatial,
        )

    sbm.close_scope()
    return output_ref


def _stage0_depthwise_height_tile_to_sbuf(
    input_2d,
    filter_tiles,
    bias_tiles,
    sbm,
    source_row_start,
    source_row_count,
    padded_row_start,
):
    """Compute a 16-row depthwise tile and keep both C-tiles in SBUF."""

    depthwise_tiles = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        depthwise_output = sbm.alloc_stack(
            (_PARTITION_TILE, _HEIGHT_TILE_SPATIAL),
            dtype=nl.float32,
        )
        depthwise_tiles.append(depthwise_output)

        channel_start = channel_tile * _PARTITION_TILE
        sbm.open_scope()
        padded_input = sbm.alloc_stack(
            (_PARTITION_TILE, _PADDED_TILE_SPATIAL),
            dtype=nl.float32,
        )
        nisa.memset(dst=padded_input, value=0)
        nisa.dma_copy(
            dst=padded_input.ap(
                pattern=[
                    [_PADDED_TILE_SPATIAL, _PARTITION_TILE],
                    [_PADDED_WIDTH, source_row_count],
                    [1, _WIDTH],
                ],
                offset=(
                    padded_row_start * _PADDED_WIDTH
                    + _DW_PADDING
                ),
            ),
            src=input_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [_WIDTH, source_row_count],
                    [1, _WIDTH],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + source_row_start * _WIDTH
                ),
            ),
        )

        nisa.memset(dst=depthwise_output, value=0)
        for kernel_row in nl.affine_range(_DW_KERNEL):
            for kernel_col in nl.affine_range(_DW_KERNEL):
                kernel_index = kernel_row * _DW_KERNEL + kernel_col
                nisa.scalar_tensor_tensor(
                    dst=depthwise_output.ap(
                        pattern=[
                            [_HEIGHT_TILE_SPATIAL, _PARTITION_TILE],
                            [_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=0,
                    ),
                    data=padded_input.ap(
                        pattern=[
                            [_PADDED_TILE_SPATIAL, _PARTITION_TILE],
                            [_PADDED_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=(
                            kernel_row * _PADDED_WIDTH
                            + kernel_col
                        ),
                    ),
                    op0=nl.multiply,
                    operand0=filter_tiles[channel_tile][
                        0:_PARTITION_TILE,
                        kernel_index : kernel_index + 1,
                    ],
                    op1=nl.add,
                    operand1=depthwise_output.ap(
                        pattern=[
                            [_HEIGHT_TILE_SPATIAL, _PARTITION_TILE],
                            [_WIDTH, _HEIGHT_TILE],
                            [1, _WIDTH],
                        ],
                        offset=0,
                    ),
                )

        nisa.tensor_tensor(
            dst=depthwise_output,
            data1=depthwise_output,
            data2=TensorView(bias_tiles[channel_tile])
            .broadcast(dim=1, size=_HEIGHT_TILE_SPATIAL)
            .get_view(),
            op=nl.add,
        )
        sbm.close_scope()

    return depthwise_tiles


def _stage0_post_spatial_tile(
    hidden_tiles,
    residual_2d,
    output_2d,
    norm_weights,
    norm_biases,
    pw1_weights,
    pw1_biases,
    pw2_weights,
    pw2_biases,
    layer_scales,
    ones_fp32,
    sbm,
    spatial_start,
):
    """Apply the fused ConvNeXt post-depthwise path to 512 positions."""

    sbm.open_scope()
    sum_psum = nl.ndarray(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    square_sum_psum = nl.ndarray(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    input_fp32 = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    squared_fp32 = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        nisa.tensor_copy(
            dst=input_fp32,
            src=hidden_tiles[channel_tile],
        )
        nisa.activation(
            dst=squared_fp32,
            data=input_fp32,
            op=nl.square,
        )
        nisa.nc_matmul(
            dst=sum_psum,
            stationary=ones_fp32,
            moving=input_fp32,
            accumulate=(channel_tile > 0),
        )
        nisa.nc_matmul(
            dst=square_sum_psum,
            stationary=ones_fp32,
            moving=squared_fp32,
            accumulate=(channel_tile > 0),
        )

    mean = sbm.alloc_stack(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    mean_square = sbm.alloc_stack(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    square_mean = sbm.alloc_stack(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    variance = sbm.alloc_stack(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    invstd = sbm.alloc_stack(
        (1, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    nisa.tensor_copy(dst=mean, src=sum_psum)
    nisa.tensor_copy(dst=mean_square, src=square_sum_psum)
    nisa.tensor_scalar(
        dst=mean,
        data=mean,
        op0=nl.multiply,
        operand0=(1.0 / _CHANNELS),
    )
    nisa.tensor_scalar(
        dst=mean_square,
        data=mean_square,
        op0=nl.multiply,
        operand0=(1.0 / _CHANNELS),
    )
    nisa.activation(dst=square_mean, data=mean, op=nl.square)
    nisa.tensor_tensor(
        dst=variance,
        data1=mean_square,
        data2=square_mean,
        op=nl.subtract,
    )
    nisa.tensor_scalar(
        dst=variance,
        data=variance,
        op0=nl.maximum,
        operand0=0.0,
    )
    nisa.tensor_scalar(
        dst=variance,
        data=variance,
        op0=nl.add,
        operand0=1e-6,
    )
    nisa.activation(dst=invstd, data=variance, op=nl.rsqrt)

    mean_broadcast = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    invstd_broadcast = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    stream_shuffle_broadcast(src=mean, dst=mean_broadcast)
    stream_shuffle_broadcast(src=invstd, dst=invstd_broadcast)

    normalized_tiles = []
    centered = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    normalized_fp32 = sbm.alloc_stack(
        (_PARTITION_TILE, _SPATIAL_TILE),
        dtype=nl.float32,
    )
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        nisa.tensor_copy(
            dst=input_fp32,
            src=hidden_tiles[channel_tile],
        )
        nisa.tensor_tensor(
            dst=centered,
            data1=input_fp32,
            data2=mean_broadcast,
            op=nl.subtract,
        )
        nisa.tensor_tensor(
            dst=normalized_fp32,
            data1=centered,
            data2=invstd_broadcast,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=normalized_fp32,
            data1=normalized_fp32,
            data2=TensorView(norm_weights[channel_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=normalized_fp32,
            data1=normalized_fp32,
            data2=TensorView(norm_biases[channel_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.add,
        )
        normalized_bf16 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.bfloat16,
        )
        nisa.tensor_copy(
            dst=normalized_bf16,
            src=normalized_fp32,
        )
        normalized_tiles.append(normalized_bf16)

    expanded_tiles = []
    for output_tile in nl.affine_range(_EXPANDED_TILES):
        expanded_psum = nl.ndarray(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for input_tile in nl.affine_range(_CHANNEL_TILES):
            nisa.nc_matmul(
                dst=expanded_psum,
                stationary=pw1_weights[output_tile][input_tile],
                moving=normalized_tiles[input_tile],
                accumulate=(input_tile > 0),
            )
        expanded_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        expanded_bf16 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.bfloat16,
        )
        nisa.tensor_copy(dst=expanded_fp32, src=expanded_psum)
        nisa.tensor_tensor(
            dst=expanded_fp32,
            data1=expanded_fp32,
            data2=TensorView(pw1_biases[output_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.add,
        )
        nisa.activation(
            dst=expanded_bf16,
            data=expanded_fp32,
            op=nl.gelu,
        )
        expanded_tiles.append(expanded_bf16)

    for output_tile in nl.affine_range(_CHANNEL_TILES):
        output_psum = nl.ndarray(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for input_tile in nl.affine_range(_EXPANDED_TILES):
            nisa.nc_matmul(
                dst=output_psum,
                stationary=pw2_weights[output_tile][input_tile],
                moving=expanded_tiles[input_tile],
                accumulate=(input_tile > 0),
            )

        output_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        residual_fp32 = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
            dtype=nl.float32,
        )
        channel_start = output_tile * _PARTITION_TILE

        nisa.tensor_copy(dst=output_fp32, src=output_psum)
        nisa.tensor_tensor(
            dst=output_fp32,
            data1=output_fp32,
            data2=TensorView(pw2_biases[output_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.add,
        )
        nisa.tensor_tensor(
            dst=output_fp32,
            data1=output_fp32,
            data2=TensorView(layer_scales[output_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.multiply,
        )
        nisa.dma_copy(
            dst=residual_fp32,
            src=residual_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [1, _SPATIAL_TILE],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + spatial_start
                ),
            ),
        )
        nisa.tensor_tensor(
            dst=output_fp32,
            data1=output_fp32,
            data2=residual_fp32,
            op=nl.add,
        )
        nisa.dma_copy(
            dst=output_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [1, _SPATIAL_TILE],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + spatial_start
                ),
            ),
            src=output_fp32,
        )

    sbm.close_scope()


def _stage0_full_height_tile(
    input_2d,
    output_2d,
    dw_filter_tiles,
    dw_bias_tiles,
    norm_weights,
    norm_biases,
    pw1_weights,
    pw1_biases,
    pw2_weights,
    pw2_biases,
    layer_scales,
    ones_fp32,
    sbm,
    source_row_start,
    source_row_count,
    padded_row_start,
    output_spatial_start,
):
    """Fuse one depthwise height tile with five pointwise spatial tiles."""

    sbm.open_scope()
    depthwise_tiles = _stage0_depthwise_height_tile_to_sbuf(
        input_2d,
        dw_filter_tiles,
        dw_bias_tiles,
        sbm,
        source_row_start=source_row_start,
        source_row_count=source_row_count,
        padded_row_start=padded_row_start,
    )

    spatial_tiles_per_height_tile = (
        _HEIGHT_TILE_SPATIAL // _SPATIAL_TILE
    )
    for local_spatial_tile in nl.sequential_range(
        spatial_tiles_per_height_tile
    ):
        local_spatial_start = local_spatial_tile * _SPATIAL_TILE
        hidden_tiles = []
        for channel_tile in nl.affine_range(_CHANNEL_TILES):
            hidden_tiles.append(
                depthwise_tiles[channel_tile].ap(
                    pattern=[
                        [_HEIGHT_TILE_SPATIAL, _PARTITION_TILE],
                        [1, _SPATIAL_TILE],
                    ],
                    offset=local_spatial_start,
                )
            )
        _stage0_post_spatial_tile(
            hidden_tiles,
            input_2d,
            output_2d,
            norm_weights,
            norm_biases,
            pw1_weights,
            pw1_biases,
            pw2_weights,
            pw2_biases,
            layer_scales,
            ones_fp32,
            sbm,
            spatial_start=(
                output_spatial_start + local_spatial_start
            ),
        )

    sbm.close_scope()


@nki.jit
def convnext_stage0_fused_7x7_block(
    hidden_ref: nl.ndarray,
    dw_filter_ref: nl.ndarray,
    dw_bias_ref: nl.ndarray,
    norm_weight_ref: nl.ndarray,
    norm_bias_ref: nl.ndarray,
    pw1_weight_ref: nl.ndarray,
    pw1_bias_ref: nl.ndarray,
    pw2_weight_ref: nl.ndarray,
    pw2_bias_ref: nl.ndarray,
    layer_scale_ref: nl.ndarray,
) -> nl.ndarray:
    """Fuse an entire stage-0 ConvNeXt-XL block into one NKI kernel."""

    kernel_assert(
        hidden_ref.shape == (1, _CHANNELS, _HEIGHT, _WIDTH),
        f"hidden_ref must be {(1, _CHANNELS, _HEIGHT, _WIDTH)}",
    )
    kernel_assert(
        dw_filter_ref.shape
        == (_CHANNEL_TILES, _PARTITION_TILE, _DW_KERNEL * _DW_KERNEL),
        "invalid dw_filter_ref shape",
    )
    kernel_assert(
        dw_bias_ref.shape == (_CHANNEL_TILES, _PARTITION_TILE, 1),
        "invalid dw_bias_ref shape",
    )
    kernel_assert(
        norm_weight_ref.shape == (_CHANNEL_TILES, _PARTITION_TILE, 1),
        "invalid norm_weight_ref shape",
    )
    kernel_assert(
        pw1_weight_ref.shape
        == (
            _EXPANDED_TILES,
            _CHANNEL_TILES,
            _PARTITION_TILE,
            _PARTITION_TILE,
        ),
        "invalid pw1_weight_ref shape",
    )
    kernel_assert(
        pw2_weight_ref.shape
        == (
            _CHANNEL_TILES,
            _EXPANDED_TILES,
            _PARTITION_TILE,
            _PARTITION_TILE,
        ),
        "invalid pw2_weight_ref shape",
    )

    program_count = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    kernel_assert(program_count == 2, "this kernel requires an LNC2 launch")

    output_ref = nl.ndarray(
        hidden_ref.shape,
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    input_2d = hidden_ref.reshape((_CHANNELS, _SPATIAL_SIZE))
    output_2d = output_ref.reshape((_CHANNELS, _SPATIAL_SIZE))

    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="convnext_stage0_fused_block")

    ones_fp32 = sbm.alloc_stack(
        (_PARTITION_TILE, 1),
        dtype=nl.float32,
        name="ones_fp32",
    )
    nisa.memset(dst=ones_fp32, value=1.0)

    dw_filter_tiles = []
    dw_bias_tiles = []
    norm_weights = []
    norm_biases = []
    layer_scales = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        dw_filter_tile = sbm.alloc_stack(
            (_PARTITION_TILE, _DW_KERNEL * _DW_KERNEL),
            dtype=nl.float32,
            name=f"dw_filter_{channel_tile}",
        )
        dw_bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"dw_bias_{channel_tile}",
        )
        norm_weight = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"norm_weight_{channel_tile}",
        )
        norm_bias = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"norm_bias_{channel_tile}",
        )
        layer_scale = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"layer_scale_{channel_tile}",
        )
        nisa.dma_copy(
            dst=dw_filter_tile,
            src=dw_filter_ref[
                channel_tile,
                0:_PARTITION_TILE,
                0 : _DW_KERNEL * _DW_KERNEL,
            ],
        )
        nisa.dma_copy(
            dst=dw_bias_tile,
            src=dw_bias_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        nisa.dma_copy(
            dst=norm_weight,
            src=norm_weight_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        nisa.dma_copy(
            dst=norm_bias,
            src=norm_bias_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        nisa.dma_copy(
            dst=layer_scale,
            src=layer_scale_ref[channel_tile, 0:_PARTITION_TILE, 0:1],
        )
        dw_filter_tiles.append(dw_filter_tile)
        dw_bias_tiles.append(dw_bias_tile)
        norm_weights.append(norm_weight)
        norm_biases.append(norm_bias)
        layer_scales.append(layer_scale)

    pw1_weights = []
    pw1_biases = []
    for output_tile in nl.affine_range(_EXPANDED_TILES):
        weight_row = []
        for input_tile in nl.affine_range(_CHANNEL_TILES):
            weight_tile = sbm.alloc_stack(
                (_PARTITION_TILE, _PARTITION_TILE),
                dtype=nl.bfloat16,
                name=f"pw1_weight_{output_tile}_{input_tile}",
            )
            nisa.dma_copy(
                dst=weight_tile,
                src=pw1_weight_ref[
                    output_tile,
                    input_tile,
                    0:_PARTITION_TILE,
                    0:_PARTITION_TILE,
                ],
            )
            weight_row.append(weight_tile)
        bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"pw1_bias_{output_tile}",
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=pw1_bias_ref[output_tile, 0:_PARTITION_TILE, 0:1],
        )
        pw1_weights.append(weight_row)
        pw1_biases.append(bias_tile)

    pw2_weights = []
    pw2_biases = []
    for output_tile in nl.affine_range(_CHANNEL_TILES):
        weight_row = []
        for input_tile in nl.affine_range(_EXPANDED_TILES):
            weight_tile = sbm.alloc_stack(
                (_PARTITION_TILE, _PARTITION_TILE),
                dtype=nl.bfloat16,
                name=f"pw2_weight_{output_tile}_{input_tile}",
            )
            nisa.dma_copy(
                dst=weight_tile,
                src=pw2_weight_ref[
                    output_tile,
                    input_tile,
                    0:_PARTITION_TILE,
                    0:_PARTITION_TILE,
                ],
            )
            weight_row.append(weight_tile)
        bias_tile = sbm.alloc_stack(
            (_PARTITION_TILE, 1),
            dtype=nl.float32,
            name=f"pw2_bias_{output_tile}",
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=pw2_bias_ref[output_tile, 0:_PARTITION_TILE, 0:1],
        )
        pw2_weights.append(weight_row)
        pw2_biases.append(bias_tile)

    edge_source_row = program_id * (_HEIGHT - (_HEIGHT_TILE + _DW_PADDING))
    edge_padded_row = (1 - program_id) * _DW_PADDING
    edge_output_spatial = (
        program_id * (_HEIGHT_TILES - 1) * _HEIGHT_TILE_SPATIAL
    )
    _stage0_full_height_tile(
        input_2d,
        output_2d,
        dw_filter_tiles,
        dw_bias_tiles,
        norm_weights,
        norm_biases,
        pw1_weights,
        pw1_biases,
        pw2_weights,
        pw2_biases,
        layer_scales,
        ones_fp32,
        sbm,
        source_row_start=edge_source_row,
        source_row_count=_HEIGHT_TILE + _DW_PADDING,
        padded_row_start=edge_padded_row,
        output_spatial_start=edge_output_spatial,
    )

    interior_tiles_per_program = (_HEIGHT_TILES - 2) // program_count
    for local_tile in nl.sequential_range(interior_tiles_per_program):
        height_tile = (
            1
            + program_id * interior_tiles_per_program
            + local_tile
        )
        source_row = height_tile * _HEIGHT_TILE - _DW_PADDING
        output_spatial = height_tile * _HEIGHT_TILE_SPATIAL
        _stage0_full_height_tile(
            input_2d,
            output_2d,
            dw_filter_tiles,
            dw_bias_tiles,
            norm_weights,
            norm_biases,
            pw1_weights,
            pw1_biases,
            pw2_weights,
            pw2_biases,
            layer_scales,
            ones_fp32,
            sbm,
            source_row_start=source_row,
            source_row_count=_PADDED_HEIGHT_TILE,
            padded_row_start=0,
            output_spatial_start=output_spatial,
        )

    sbm.close_scope()
    return output_ref


def _blocked_transposed_weight(linear: nn.Linear) -> Tensor:
    output_tiles = linear.out_features // _PARTITION_TILE
    input_tiles = linear.in_features // _PARTITION_TILE
    return (
        linear.weight.detach()
        .reshape(
            output_tiles,
            _PARTITION_TILE,
            input_tiles,
            _PARTITION_TILE,
        )
        .permute(0, 2, 3, 1)
        .contiguous()
        .to(torch.bfloat16)
    )


def _blocked_vector(vector: Tensor, tile_count: int) -> Tensor:
    return (
        vector.detach()
        .reshape(tile_count, _PARTITION_TILE, 1)
        .contiguous()
        .to(torch.float32)
    )


class NkiConvNextStage0DepthwiseCore(nn.Module):
    """Torch wrapper for the specialized stage-0 depthwise kernel."""

    def __init__(self, layer: nn.Module, lnc: int = 2):
        super().__init__()
        if layer.dwconv.in_channels != _CHANNELS:
            raise ValueError("This prototype only supports stage-0 C=256")
        if layer.dwconv.groups != _CHANNELS:
            raise ValueError("A depthwise Conv2d is required")
        if layer.dwconv.kernel_size != (_DW_KERNEL, _DW_KERNEL):
            raise ValueError("This prototype requires a 7x7 kernel")
        if layer.dwconv.bias is None:
            raise ValueError("A depthwise-convolution bias is required")
        if lnc != 2:
            raise ValueError("This prototype currently requires LNC2")

        self.lnc = lnc
        self.register_buffer(
            "filter",
            layer.dwconv.weight.detach()
            .reshape(
                _CHANNEL_TILES,
                _PARTITION_TILE,
                _DW_KERNEL * _DW_KERNEL,
            )
            .contiguous()
            .to(torch.float32),
        )
        self.register_buffer(
            "bias",
            layer.dwconv.bias.detach()
            .reshape(_CHANNEL_TILES, _PARTITION_TILE, 1)
            .contiguous()
            .to(torch.float32),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return convnext_stage0_depthwise_7x7[self.lnc](
            hidden_states.to(torch.bfloat16),
            self.filter,
            self.bias,
        )


class NkiConvNextStage0PostDwCore(nn.Module):
    """Torch wrapper for the specialized fused stage-0 NKI kernel."""

    def __init__(self, layer: nn.Module, lnc: int = 2):
        super().__init__()
        if layer.pwconv1.in_features != _CHANNELS:
            raise ValueError("This prototype only supports stage-0 C=256")
        if layer.pwconv1.out_features != _EXPANDED_CHANNELS:
            raise ValueError("This prototype only supports expansion=4")
        if layer.layer_scale_parameter is None:
            raise ValueError("ConvNeXt layer scale is required")
        if lnc != 2:
            raise ValueError("This prototype currently requires LNC2")

        self.lnc = lnc
        self.register_buffer(
            "norm_weight",
            _blocked_vector(layer.layernorm.weight, _CHANNEL_TILES),
        )
        self.register_buffer(
            "norm_bias",
            _blocked_vector(layer.layernorm.bias, _CHANNEL_TILES),
        )
        self.register_buffer(
            "pw1_weight",
            _blocked_transposed_weight(layer.pwconv1),
        )
        self.register_buffer(
            "pw1_bias",
            _blocked_vector(layer.pwconv1.bias, _EXPANDED_TILES),
        )
        self.register_buffer(
            "pw2_weight",
            _blocked_transposed_weight(layer.pwconv2),
        )
        self.register_buffer(
            "pw2_bias",
            _blocked_vector(layer.pwconv2.bias, _CHANNEL_TILES),
        )
        self.register_buffer(
            "layer_scale",
            _blocked_vector(
                layer.layer_scale_parameter,
                _CHANNEL_TILES,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
    ) -> Tensor:
        return convnext_stage0_post_dw_fused[self.lnc](
            hidden_states.to(torch.bfloat16),
            residual.to(torch.bfloat16),
            self.norm_weight,
            self.norm_bias,
            self.pw1_weight,
            self.pw1_bias,
            self.pw2_weight,
            self.pw2_bias,
            self.layer_scale,
        )


class NkiConvNextStage0FusedLayerCore(nn.Module):
    """Fully fused stage-0 DWConv-to-residual ConvNeXt block."""

    def __init__(self, layer: nn.Module, lnc: int = 2):
        super().__init__()
        self.lnc = lnc
        self.depthwise = NkiConvNextStage0DepthwiseCore(
            layer,
            lnc=lnc,
        )
        self.post_dw = NkiConvNextStage0PostDwCore(
            layer,
            lnc=lnc,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return convnext_stage0_fused_7x7_block[self.lnc](
            hidden_states,
            self.depthwise.filter,
            self.depthwise.bias,
            self.post_dw.norm_weight,
            self.post_dw.norm_bias,
            self.post_dw.pw1_weight,
            self.post_dw.pw1_bias,
            self.post_dw.pw2_weight,
            self.post_dw.pw2_bias,
            self.post_dw.layer_scale,
        )


class NkiConvNextStage0LayerCore(nn.Module):
    """Depthwise Conv2d followed by the fused NKI post-depthwise path."""

    def __init__(self, layer: nn.Module, lnc: int = 2):
        super().__init__()
        self.dwconv = layer.dwconv
        self.post_dw = NkiConvNextStage0PostDwCore(layer, lnc=lnc)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.post_dw(self.dwconv(hidden_states), hidden_states)
