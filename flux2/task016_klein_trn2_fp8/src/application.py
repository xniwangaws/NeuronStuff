# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FLUX.2-klein-base-9B NxDI application and pipeline integration.

This module provides:
- NeuronFlux2KleinApplication: Top-level orchestrator for compile/load/generate
- create_flux2_klein_config: Config factory for backbone + VAE
- NeuronTransformerWrapper: Drop-in replacement for Flux2Transformer2DModel in Diffusers pipeline

The Qwen3-8B text encoder (layers 0-26, three torch_neuronx segments on
logical cores 1/2/3) and the VAE decoder (one segment on logical core 0) also
run on Neuron; see neuron_aux.py. In the original port both ran on the host
CPU and accounted for 14.3 s of the 41.4 s per image.
"""

import gc
import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn

try:
    from neuronx_distributed_inference.models.config import (
        InferenceConfig,
        NeuronConfig,
    )
    from neuronx_distributed_inference.utils.diffusers_adapter import (
        load_diffusers_config,
    )

    try:
        from .modeling_flux2_klein import (
            Flux2KleinBackboneInferenceConfig,
            NeuronFlux2KleinBackboneApplication,
        )
        from . import neuron_aux
    except ImportError:
        from modeling_flux2_klein import (
            Flux2KleinBackboneInferenceConfig,
            NeuronFlux2KleinBackboneApplication,
        )
        import neuron_aux

    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_flux2_klein_config(
    model_path: str,
    backbone_tp_degree: int = 4,
    dtype=torch.bfloat16,
    height: int = 1024,
    width: int = 1024,
):
    """
    Create NxDI configuration for FLUX.2-klein backbone.

    Args:
        model_path: Path to the HuggingFace model directory
        backbone_tp_degree: Tensor parallelism degree (4 for trn2.3xlarge LNC=2)
        dtype: Model data type
        height: Image height
        width: Image width

    Returns:
        Flux2KleinBackboneInferenceConfig for the transformer backbone
    """
    transformer_path = os.path.join(model_path, "transformer")

    backbone_neuron_config = NeuronConfig(
        tp_degree=backbone_tp_degree,
        world_size=backbone_tp_degree,
        torch_dtype=dtype,
    )

    backbone_config = Flux2KleinBackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
    )

    # FLUX.2-klein specific defaults
    if (
        not hasattr(backbone_config, "out_channels")
        or backbone_config.out_channels is None
    ):
        backbone_config.out_channels = backbone_config.in_channels

    # VAE scale factor for FLUX.2: 16 (8x spatial * 2x patch)
    setattr(backbone_config, "vae_scale_factor", 16)

    return backbone_config


class NeuronTransformerWrapper(nn.Module):
    """
    Drop-in replacement for Flux2Transformer2DModel in the Diffusers pipeline.

    This wrapper accepts the same forward() signature as the HuggingFace model
    and delegates to the compiled NxDI backbone.
    """

    def __init__(self, neuron_app: "NeuronFlux2KleinBackboneApplication", config):
        super().__init__()
        self.neuron_app = neuron_app
        self.config = config
        # Expose config attributes that the pipeline expects
        self._internal_dict = getattr(config, "_internal_dict", {})

    def cache_context(self, name: str):
        """Delegate cache_context to the NxDI backbone application."""
        return self.neuron_app.cache_context(name)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=False,
        **kwargs,
    ):
        """Match Flux2Transformer2DModel.forward() signature."""
        output = self.neuron_app.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=return_dict,
        )
        # Wrap output to match Diffusers Transformer2DModelOutput
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)

    @property
    def dtype(self):
        return self.config.neuron_config.torch_dtype


class NeuronFlux2KleinApplication(nn.Module):
    """
    Top-level application for FLUX.2-klein-base-9B on Neuron.

    Components:
    - Transformer backbone: Compiled with NxDI (TP=4 on trn2.3xlarge)
    - Text encoder (Qwen3-8B, layers 0-26): torch_neuronx, 3 segments on cores 1/2/3
    - VAE decoder: torch_neuronx (unet-inference model type), core 0
    - Scheduler: FlowMatchEulerDiscreteScheduler from Diffusers

    Usage:
        app = NeuronFlux2KleinApplication(model_path, backbone_config)
        app.compile(compile_dir)
        app.load(compile_dir)
        image = app("A photo of a cat", height=1024, width=1024)
    """

    def __init__(
        self,
        model_path: str,
        backbone_config: "Flux2KleinBackboneInferenceConfig",
        height: int = 1024,
        width: int = 1024,
        transformer_checkpoint: Optional[str] = None,
        neuron_aux: bool = True,
        aux_compiled_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.backbone_config = backbone_config
        self.height = height
        self.width = width
        # Run the Qwen3 text encoder and VAE decoder on Neuron (True) or on the
        # host CPU as in the original port (False).
        self.neuron_aux = neuron_aux
        self.aux_compiled_path = aux_compiled_path
        self.neuron_vae_decode = None
        self._rope_cache = None

        transformer_path = os.path.join(model_path, "transformer")
        transformer_checkpoint = transformer_checkpoint or os.environ.get(
            "FLUX2_FP8_CHECKPOINT"
        )
        backbone_model_path = transformer_checkpoint or transformer_path
        self.transformer_checkpoint = transformer_checkpoint

        # Create NxDI backbone application
        self.backbone_app = NeuronFlux2KleinBackboneApplication(
            model_path=backbone_model_path,
            config=backbone_config,
        )

        # Diffusers pipeline (tokenizer, scheduler, VAE BatchNorm stats, and
        # the CPU fallbacks when neuron_aux=False)
        self.pipe = None

    def _load_pipeline(self):
        """Load the Diffusers pipeline for CPU components."""
        if self.pipe is not None:
            return

        from diffusers import Flux2KleinPipeline

        logger.info("Loading Flux2KleinPipeline...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Pipeline loaded successfully.")

    def compile(self, compiled_model_path: str, debug: bool = False):
        """Compile the transformer backbone, then (unless disabled) the text
        encoder and VAE decoder. The auxiliary artifacts are resolution/seq
        specific but precision independent, so ``aux_compiled_path`` lets
        BF16 and FP8 transformer builds share them."""
        transformer_dir = os.path.join(compiled_model_path, "transformer")
        compiled_model = os.path.join(transformer_dir, "model.pt")
        if os.path.isfile(compiled_model):
            logger.info(f"Transformer already compiled at {transformer_dir}, skipping.")
        else:
            os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
                compiled_model_path, "compiler_workdir"
            )
            self.backbone_app.compile(transformer_dir, debug)

        if not self.neuron_aux:
            return
        aux_dir = self.aux_compiled_path or compiled_model_path
        needs_te = not all(
            os.path.isfile(neuron_aux.te_segment_path(aux_dir, name))
            for name, _, _ in neuron_aux.TE_SEGMENTS
        )
        needs_vae = not os.path.isfile(neuron_aux.vae_path(aux_dir))
        if needs_te or needs_vae:
            self._load_pipeline()
        if needs_te:
            neuron_aux.compile_text_encoder(self.pipe.text_encoder, self.pipe.tokenizer, aux_dir)
        if needs_vae:
            neuron_aux.compile_vae_decoder(self.pipe.vae, aux_dir, self.height, self.width)

    def load(self, compiled_model_path: str, skip_warmup: bool = False):
        """Load the compiled transformer backbone plus Neuron text encoder /
        VAE decoder, and hot-swap them into the Diffusers pipeline."""
        transformer_dir = os.path.join(compiled_model_path, "transformer")
        self.backbone_app.load(transformer_dir, skip_warmup=skip_warmup)

        # Load pipeline for CPU components
        self._load_pipeline()

        # Hot-swap transformer with Neuron version
        self.pipe.transformer = NeuronTransformerWrapper(
            self.backbone_app,
            self.backbone_config,
        )

        # The RoPE table depends only on the (fixed) resolution and text length;
        # without this the wrapper recomputes it on the host for every forward.
        self._rope_cache = self.backbone_app.image_rotary_emb_cache_context()
        self._rope_cache.__enter__()

        if self.neuron_aux:
            aux_dir = self.aux_compiled_path or compiled_model_path
            # Text encoder: three Qwen3 segments on logical cores 1/2/3. This
            # drops the 16 GB CPU Qwen3ForCausalLM.
            self.pipe.text_encoder = neuron_aux.NeuronKleinTextEncoder(aux_dir)
            # VAE decode on logical core 0. The CPU VAE object is kept because
            # the pipeline reads its BatchNorm statistics to denormalize the
            # latents before decoding.
            self.neuron_vae_decode = neuron_aux.NeuronVaeDecode(aux_dir)
            self.pipe.vae.decode = self.neuron_vae_decode

    def __call__(
        self,
        prompt: str,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        **kwargs,
    ):
        """Generate an image."""
        height = height or self.height
        width = width or self.width

        # Diffusers 0.37.1 implements classic CFG for Klein Base by encoding an
        # empty unconditional prompt internally. Its pipeline call does not
        # accept the older `negative_prompt` keyword.
        negative_prompt = kwargs.pop("negative_prompt", None)
        is_empty_negative_prompt = negative_prompt in (None, "") or (
            isinstance(negative_prompt, list)
            and all(prompt == "" for prompt in negative_prompt)
        )
        if not is_empty_negative_prompt:
            raise ValueError(
                "Flux2KleinPipeline only supports its built-in empty "
                "unconditional prompt; a non-empty negative_prompt was provided."
            )

        return self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
