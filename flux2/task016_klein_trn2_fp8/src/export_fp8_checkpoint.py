#!/usr/bin/env python3
"""Export an NxDI-format per-row E4M3 checkpoint for FLUX.2 Klein."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("FLUX2_FP8_SCOPE", "mlp")
os.environ.setdefault("FLUX2_FP8_ACTIVATION", "dynamic")

from application import create_flux2_klein_config
from modeling_flux2_klein import (
    NeuronFlux2KleinBackboneApplication,
    _is_fp8_attention_weight,
    _is_fp8_auxiliary_weight,
    _is_fp8_mlp_weight,
    _is_fp8_weight,
    _is_prequantized_fp8_state_dict,
)
from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    save_state_dict_safetensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/nvme/flux2-klein/weights",
        help="Original Hugging Face FLUX.2 Klein model directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for sharded safetensors and manifest.json.",
    )
    parser.add_argument(
        "--scope",
        choices=("mlp", "attention", "mlp_attention", "all_linear"),
        default=os.environ.get("FLUX2_FP8_SCOPE", "mlp"),
        help="Transformer projection groups to quantize.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["FLUX2_FP8_SCOPE"] = args.scope
    model_root = Path(args.model)
    transformer_path = model_root / "transformer"
    output_path = Path(
        args.output
        or f"/mnt/nvme/flux2-klein/checkpoints/fp8_{args.scope}_per_row"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    existing = list(output_path.glob("*.safetensors")) + list(
        output_path.glob("*.index.json")
    )
    if existing:
        raise SystemExit(
            f"Refusing to overwrite existing checkpoint files in {output_path}"
        )

    config = create_flux2_klein_config(
        model_path=str(model_root),
        backbone_tp_degree=4,
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
    )
    source_state_dict = load_state_dict(str(transformer_path))
    converted_state_dict = (
        NeuronFlux2KleinBackboneApplication.convert_hf_to_neuron_state_dict(
            source_state_dict,
            config,
        )
    )
    del source_state_dict
    gc.collect()

    if not _is_prequantized_fp8_state_dict(converted_state_dict):
        raise RuntimeError(
            f"Export did not produce a complete FP8 checkpoint for {args.scope}"
        )

    fp8_weight_keys = sorted(
        key
        for key, value in converted_state_dict.items()
        if _is_fp8_weight(key) and value.dtype == torch.float8_e4m3fn
    )
    fp8_mlp_weight_count = sum(_is_fp8_mlp_weight(key) for key in fp8_weight_keys)
    fp8_attention_weight_count = sum(
        _is_fp8_attention_weight(key) for key in fp8_weight_keys
    )
    fp8_auxiliary_weight_count = sum(
        _is_fp8_auxiliary_weight(key) for key in fp8_weight_keys
    )
    total_fp8_weight_bytes = sum(
        converted_state_dict[key].numel()
        * converted_state_dict[key].element_size()
        for key in fp8_weight_keys
    )
    total_scale_bytes = sum(
        converted_state_dict[key.replace(".weight", ".scale")].numel()
        * converted_state_dict[key.replace(".weight", ".scale")].element_size()
        for key in fp8_weight_keys
    )

    save_state_dict_safetensors(
        state_dict=converted_state_dict,
        state_dict_dir=str(output_path),
        max_shard_size=args.max_shard_size,
    )
    manifest = {
        "format": "flux2-klein-nxdi-fp8-v2",
        "source_model": str(model_root),
        "scope": args.scope,
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "weight_quantization": "per_output_row_symmetric",
        "weight_scale_shape": "[out_features, 1]",
        "activation_quantization": "dynamic_e4m3",
        "fp8_weight_tensor_count": len(fp8_weight_keys),
        "fp8_mlp_weight_tensor_count": fp8_mlp_weight_count,
        "fp8_attention_weight_tensor_count": fp8_attention_weight_count,
        "fp8_auxiliary_weight_tensor_count": fp8_auxiliary_weight_count,
        "fp8_weight_bytes": total_fp8_weight_bytes,
        "scale_bytes": total_scale_bytes,
        "fp8_weight_keys": fp8_weight_keys,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
