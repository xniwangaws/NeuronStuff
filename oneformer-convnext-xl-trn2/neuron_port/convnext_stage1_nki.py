"""Stage-1 ConvNeXt-XL NKI megakernel prototypes."""

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

from .convnext_nki import _blocked_transposed_weight, _blocked_vector


_CHANNELS = 512
_EXPANDED_CHANNELS = 2048
_HEIGHT = 80
_WIDTH = 80
_SPATIAL_SIZE = _HEIGHT * _WIDTH
_PARTITION_TILE = 128
_CHANNEL_TILES = _CHANNELS // _PARTITION_TILE
_EXPANDED_TILES = _EXPANDED_CHANNELS // _PARTITION_TILE

_DW_KERNEL = 7
_DW_PADDING = _DW_KERNEL // 2
_HEIGHT_TILE = 5
_HEIGHT_TILES = _HEIGHT // _HEIGHT_TILE
_SPATIAL_TILE = _HEIGHT_TILE * _WIDTH
_PADDED_HEIGHT_TILE = _HEIGHT_TILE + 2 * _DW_PADDING
_PADDED_WIDTH = _WIDTH + 2 * _DW_PADDING
_PADDED_TILE_SPATIAL = _PADDED_HEIGHT_TILE * _PADDED_WIDTH


def _stage1_full_height_tile(
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
    """Compute one five-row stage-1 tile without HBM intermediates."""

    sbm.open_scope()

    expanded_tiles = []
    for expanded_tile in nl.affine_range(_EXPANDED_TILES):
        expanded_tiles.append(
            sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
        )

    sbm.open_scope()
    normalized_tiles = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        normalized_tiles.append(
            sbm.alloc_stack(
                (_PARTITION_TILE, _SPATIAL_TILE),
                dtype=nl.bfloat16,
            )
        )

    sbm.open_scope()
    depthwise_tiles = []
    for channel_tile in nl.affine_range(_CHANNEL_TILES):
        depthwise_output = sbm.alloc_stack(
            (_PARTITION_TILE, _SPATIAL_TILE),
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
                    dst=depthwise_output,
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
                    operand0=dw_filter_tiles[channel_tile][
                        0:_PARTITION_TILE,
                        kernel_index : kernel_index + 1,
                    ],
                    op1=nl.add,
                    operand1=depthwise_output,
                )

        nisa.tensor_tensor(
            dst=depthwise_output,
            data1=depthwise_output,
            data2=TensorView(dw_bias_tiles[channel_tile])
            .broadcast(dim=1, size=_SPATIAL_TILE)
            .get_view(),
            op=nl.add,
        )
        sbm.close_scope()

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
            src=depthwise_tiles[channel_tile],
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
            src=depthwise_tiles[channel_tile],
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
        nisa.tensor_copy(
            dst=normalized_tiles[channel_tile],
            src=normalized_fp32,
        )

    sbm.close_scope()

    for output_tile in nl.affine_range(_EXPANDED_TILES):
        sbm.open_scope()
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
            dst=expanded_tiles[output_tile],
            data=expanded_fp32,
            op=nl.gelu,
        )
        sbm.close_scope()

    sbm.close_scope()

    for output_tile in nl.affine_range(_CHANNEL_TILES):
        sbm.open_scope()
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
            src=input_2d.ap(
                pattern=[
                    [_SPATIAL_SIZE, _PARTITION_TILE],
                    [1, _SPATIAL_TILE],
                ],
                offset=(
                    channel_start * _SPATIAL_SIZE
                    + output_spatial_start
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
                    + output_spatial_start
                ),
            ),
            src=output_fp32,
        )
        sbm.close_scope()

    sbm.close_scope()


@nki.jit
def convnext_stage1_fused_7x7_block(
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
    """Fuse a complete C=512, H=W=80 ConvNeXt-XL block."""

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
        _HEIGHT_TILES % program_count == 0,
        "height tiles must split evenly across LNC2",
    )

    output_ref = nl.ndarray(
        hidden_ref.shape,
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    input_2d = hidden_ref.reshape((_CHANNELS, _SPATIAL_SIZE))
    output_2d = output_ref.reshape((_CHANNELS, _SPATIAL_SIZE))

    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="convnext_stage1_fused_block")

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

    edge_source_row = program_id * (
        _HEIGHT - (_HEIGHT_TILE + _DW_PADDING)
    )
    edge_padded_row = (1 - program_id) * _DW_PADDING
    edge_output_spatial = (
        program_id * (_HEIGHT_TILES - 1) * _SPATIAL_TILE
    )
    _stage1_full_height_tile(
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
        _stage1_full_height_tile(
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
            source_row_start=(
                height_tile * _HEIGHT_TILE - _DW_PADDING
            ),
            source_row_count=_PADDED_HEIGHT_TILE,
            padded_row_start=0,
            output_spatial_start=height_tile * _SPATIAL_TILE,
        )

    sbm.close_scope()
    return output_ref


class NkiConvNextStage1FusedLayerCore(nn.Module):
    """Torch wrapper for the C=512, 80x80 stage-1 megakernel."""

    def __init__(self, layer: nn.Module, lnc: int = 2):
        super().__init__()
        if layer.dwconv.in_channels != _CHANNELS:
            raise ValueError("This kernel only supports stage-1 C=512")
        if layer.pwconv1.out_features != _EXPANDED_CHANNELS:
            raise ValueError("This kernel requires expansion=4")
        if layer.dwconv.kernel_size != (_DW_KERNEL, _DW_KERNEL):
            raise ValueError("This kernel requires a 7x7 depthwise conv")
        if layer.dwconv.bias is None:
            raise ValueError("Depthwise bias is required")
        if layer.layer_scale_parameter is None:
            raise ValueError("Layer scale is required")
        if lnc != 2:
            raise ValueError("This kernel currently requires LNC2")

        self.lnc = lnc
        self.register_buffer(
            "dw_filter",
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
            "dw_bias",
            _blocked_vector(layer.dwconv.bias, _CHANNEL_TILES),
        )
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

    def forward(self, hidden_states: Tensor) -> Tensor:
        return convnext_stage1_fused_7x7_block[self.lnc](
            hidden_states,
            self.dw_filter,
            self.dw_bias,
            self.norm_weight,
            self.norm_bias,
            self.pw1_weight,
            self.pw1_bias,
            self.pw2_weight,
            self.pw2_bias,
            self.layer_scale,
        )
