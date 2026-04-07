"""
Qwen2.5-VL-7B text model for NeuronX Distributed Inference.

Based on neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text.
Key change: update_state_dict_for_tied_weights handles tie_word_embeddings=false.
"""

import logging
import gc

import torch

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

# Reuse everything from qwen2_vl base
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
    NeuronQwen2VLTextModel,
    convert_state_dict_to_fused_qkv,
)

logger = logging.getLogger("Neuron")


class NeuronQwen2_5_VLTextForCausalLM(NeuronBaseForCausalLM):
    """
    Text CausalLM for Qwen2.5-VL.
    Only difference from NeuronQwen2VLTextForCausalLM:
    - update_state_dict_for_tied_weights: handles tie_word_embeddings=false
      (Qwen2.5-VL-7B has separate lm_head.weight, unlike Qwen2-VL which ties them)
    - load_hf_model: uses Qwen2_5_VLForConditionalGeneration
    """

    _model_cls = NeuronQwen2VLTextModel

    @staticmethod
    def load_hf_model(model_path):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
        except ImportError:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # Same conversion as Qwen2-VL text model
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            if 'model.' in dict_key:
                new_key = dict_key.replace('model.', "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            replacement_atten_key = attention_keys[atten_key]
                            new_key = new_key.replace(atten_key, replacement_atten_key)
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]

        if inference_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(new_state_dict, inference_config)

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # CHANGED: Qwen2.5-VL-7B has tie_word_embeddings=false,
        # so lm_head.weight exists separately in the HF checkpoint.
        # Only copy from embed_tokens if lm_head.weight is missing.
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return InferenceConfig
