#!/bin/bash
# Setup script for Qwen2.5-VL-7B-Instruct on Neuron (vLLM 0.13 + NxDI 0.8)
#
# Prerequisites:
#   - trn2 instance with vLLM 0.13 Neuron AMI
#   - venv at /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/
#   - Model downloaded to /home/ubuntu/models/Qwen2.5-VL-7B-Instruct/
#
# Usage:
#   bash qwen2_5_vl_7b_setup.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13"
VLLM_NEURON="/vllm/vllm_neuron/worker"
MODEL_PATH="/home/ubuntu/models/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR="${SCRIPT_DIR}/qwen2_5_vl_7b"

echo "=========================================="
echo " Qwen2.5-VL-7B-Instruct Neuron Setup"
echo "=========================================="

# -----------------------------------------------
# Step 1: Download model if not present
# -----------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[1/4] Downloading model..."
    source ${VENV}/bin/activate
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ${MODEL_PATH}
else
    echo "[1/4] Model already exists at ${MODEL_PATH}"
fi

# -----------------------------------------------
# Step 2: Patch model config.json
# Add missing fields that NxDI qwen2_vl expects
# -----------------------------------------------
echo "[2/4] Patching model config.json..."
source ${VENV}/bin/activate
python3 - <<'PYEOF'
import json

config_path = "/home/ubuntu/models/Qwen2.5-VL-7B-Instruct/config.json"
with open(config_path, "r") as f:
    config = json.load(f)

vc = config.get("vision_config", {})
changed = False

# mlp_ratio: Qwen2.5-VL uses intermediate_size instead, but NxDI expects mlp_ratio
# mlp_ratio = intermediate_size / hidden_size = 3420 / 1280 = 2.671875
if "mlp_ratio" not in vc:
    vc["mlp_ratio"] = vc.get("intermediate_size", 3420) / vc.get("hidden_size", 1280)
    changed = True
    print(f"  Added mlp_ratio={vc['mlp_ratio']}")

# embed_dim: alias for hidden_size, NxDI expects this
if "embed_dim" not in vc:
    vc["embed_dim"] = vc.get("hidden_size", 1280)
    changed = True
    print(f"  Added embed_dim={vc['embed_dim']}")

# in_channels: NxDI expects this (Qwen2.5-VL uses in_chans)
if "in_channels" not in vc:
    vc["in_channels"] = vc.get("in_chans", 3)
    changed = True
    print(f"  Added in_channels={vc['in_channels']}")

# out_hidden_size: PatchMerger output dim = text hidden_size (3584 for 7B)
if "out_hidden_size" not in vc:
    vc["out_hidden_size"] = config.get("hidden_size", 3584)
    changed = True
    print(f"  Added out_hidden_size={vc['out_hidden_size']}")

if changed:
    config["vision_config"] = vc
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  Config patched successfully")
else:
    print("  Config already patched")
PYEOF

# -----------------------------------------------
# Step 3: Patch vllm_neuron to support Qwen2.5-VL
# -----------------------------------------------
echo "[3/4] Patching vllm_neuron..."
python3 - <<PYEOF
import sys

# --- Patch constants.py: add Qwen2_5_VLForConditionalGeneration to supported models ---
constants_path = "${VLLM_NEURON}/constants.py"
with open(constants_path, "r") as f:
    content = f.read()

if "Qwen2_5_VLForConditionalGeneration" not in content:
    content = content.replace(
        '"Qwen2VLForConditionalGeneration"',
        '"Qwen2VLForConditionalGeneration",\n    "Qwen2_5_VLForConditionalGeneration"'
    )
    with open(constants_path, "w") as f:
        f.write(content)
    print("  Patched constants.py")
else:
    print("  constants.py already patched")

# --- Patch neuronx_distributed_model_loader.py ---
loader_path = "${VLLM_NEURON}/neuronx_distributed_model_loader.py"
with open(loader_path, "r") as f:
    content = f.read()

# Patch 1: _get_neuron_model_cls - return our custom class instead of qwen2_vl
old_mapping = '''            if model == "qwen2_5_vl":
                model = "qwen2_vl"'''
new_mapping = '''            if model == "qwen2_5_vl":
                import sys as _sys
                _sys.path.insert(0, "${SCRIPT_DIR}")
                from qwen2_5_vl_7b.modeling_qwen2_5_vl import NeuronQwen2_5_VLForCausalLM
                return NeuronQwen2_5_VLForCausalLM'''

if old_mapping in content:
    content = content.replace(old_mapping, new_mapping)
    print("  Patched model class mapping")
elif "qwen2_5_vl" not in content:
    # First time: add the mapping
    insert_after = '''            if model == "qwen3_vl":
                model = "qwen3_vl"'''
    if insert_after in content:
        content = content.replace(insert_after, insert_after + "\n\n" + new_mapping.lstrip())
        print("  Added qwen2_5_vl model class mapping")
    else:
        print("  WARNING: Could not find insertion point for model mapping")
elif 'from qwen2_5_vl_7b.modeling_qwen2_5_vl import' in content:
    print("  Model class mapping already patched")
else:
    print("  WARNING: qwen2_5_vl mapping exists but may need manual review")

# Patch 2: get_neuron_model - add architecture dispatch
if "Qwen2_5_VLForConditionalGeneration" not in content:
    old_dispatch = '    elif architecture == "Qwen2VLForConditionalGeneration":'
    new_dispatch = '''    elif architecture == "Qwen2_5_VLForConditionalGeneration":
        model = NeuronQwen2_5_VLForCausalLM(model_config.hf_config)
''' + old_dispatch
    if old_dispatch in content:
        content = content.replace(old_dispatch, new_dispatch)
        print("  Added architecture dispatch")
    else:
        print("  WARNING: Could not find dispatch insertion point")
else:
    print("  Architecture dispatch already exists")

# Patch 3: Add import for NeuronQwen2_5_VLForCausalLM at top if class reference exists
# (the get_neuron_model function needs it)
if 'class NeuronQwen2_5_VLForCausalLM' not in content:
    # Add a minimal class definition near the other VL classes
    insert_after_class = 'class NeuronQwen2VLForCausalLM(NeuronMultiModalCausalLM):'
    if insert_after_class in content and 'class NeuronQwen2_5_VLForCausalLM' not in content:
        # Find the class and add after it
        new_class = '''

class NeuronQwen2_5_VLForCausalLM(NeuronQwen2VLForCausalLM):
    def _save_pretrained_model(self, model_name):
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path

'''
        # Find the end of NeuronQwen2VLForCausalLM or a good insertion point
        # Insert before NeuronQwen3VLForCausalLM if it exists
        if 'class NeuronQwen3VLForCausalLM' in content:
            content = content.replace(
                'class NeuronQwen3VLForCausalLM',
                new_class + 'class NeuronQwen3VLForCausalLM'
            )
            print("  Added NeuronQwen2_5_VLForCausalLM class")
        else:
            print("  WARNING: Could not find insertion point for class")
else:
    print("  NeuronQwen2_5_VLForCausalLM class already exists")

with open(loader_path, "w") as f:
    f.write(content)

# --- Patch neuronx_distributed_model_runner.py: add qwen2_5_vl mm data processing ---
runner_path = "${VLLM_NEURON}/neuronx_distributed_model_runner.py"
with open(runner_path, "r") as f:
    content = f.read()

if "qwen2_5_vl" not in content:
    old_line = 'elif self.model.model.config.model_type == "qwen3_vl":'
    new_line = 'elif self.model.model.config.model_type in ("qwen2_5_vl", "qwen3_vl"):'
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(runner_path, "w") as f:
            f.write(content)
        print("  Patched model_runner mm data processing")
    else:
        print("  WARNING: Could not find mm data processing insertion point")
else:
    print("  Model runner mm data processing already patched")
PYEOF

# -----------------------------------------------
# Step 4: Verify setup
# -----------------------------------------------
echo "[4/4] Verifying setup..."
source ${VENV}/bin/activate
python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from qwen2_5_vl_7b import Qwen2_5_VLInferenceConfig, NeuronQwen2_5_VLForCausalLM
from qwen2_5_vl_7b.modeling_qwen2_5_vl_text import NeuronQwen2_5_VLTextForCausalLM
from qwen2_5_vl_7b.modeling_qwen2_5_vl_vision import NeuronQwen2_5_VLForImageEncoding
print('  All imports OK!')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "To serve the model:"
echo "  bash ${SCRIPT_DIR}/vllm_neuron_serve_qwen2_5_vl_7b.sh"
echo ""
echo "To benchmark:"
echo "  bash ${SCRIPT_DIR}/vllm_bench_qwen2_5_vl_7b.sh"
