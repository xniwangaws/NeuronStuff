"""
Smoke test for Qwen2.5-VL-7B on Neuron.
"""
import sys
sys.path.insert(0, "/home/ubuntu/NeuronStuff/qwen_evaluation")

print("1. Testing imports...")
from qwen2_5_vl_7b import Qwen2_5_VLInferenceConfig, NeuronQwen2_5_VLForCausalLM
from qwen2_5_vl_7b.modeling_qwen2_5_vl_text import NeuronQwen2_5_VLTextForCausalLM
from qwen2_5_vl_7b.modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding
print("   All imports OK!")

print("\n2. Testing HF model class availability...")
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
print("   Qwen2_5_VLForConditionalGeneration: OK")

print("\n3. Verifying Qwen2.5-VL-7B config (from HuggingFace)...")
config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("   hidden_size:", config.hidden_size)
print("   num_hidden_layers:", config.num_hidden_layers)
print("   num_attention_heads:", config.num_attention_heads)
print("   num_key_value_heads:", config.num_key_value_heads)
print("   intermediate_size:", config.intermediate_size)
print("   vocab_size:", config.vocab_size)
print("   tie_word_embeddings:", config.tie_word_embeddings)
rs = getattr(config, "rope_scaling", {}) or {}
print("   rope_scaling type:", rs.get("type", "N/A"))
print("   vision depth:", config.vision_config.depth)
print("   vision hidden_size:", config.vision_config.hidden_size)

# Verify key 7B-specific values
assert config.hidden_size == 3584
assert config.num_hidden_layers == 28
assert config.num_attention_heads == 28
assert config.num_key_value_heads == 4
assert config.vocab_size == 152064
assert config.tie_word_embeddings == False

print("\n4. All assertions passed!")
print("\nQwen2.5-VL-7B adapter is ready.")
print("Recommended TP: tp=4 (7 heads/rank, 1 KV head/rank) or tp=2 (14 heads/rank, 2 KV heads/rank)")
