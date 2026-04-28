#!/bin/bash
# Full setup on a fresh trn2 + trace VAE encoder + decoder.
# Assumes SDK 2.29 DLAMI (AMI 20260410) so /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference already exists.

set -ex

# 1. Install python3.10 (for the S3Diff venv) + nothing else fancy
sudo apt-get update -qq 2>&1 | tail -2
sudo apt-get install -y software-properties-common 2>&1 | tail -2
sudo add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -2
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev 2>&1 | tail -2

# 2. Ensure we have enough swap for compile
if [ ! -f /swapfile ]; then
  sudo fallocate -l 64G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi

# 3. Setup S3Diff repo + deps in the pre-existing NxDI venv (torch 2.9, Python 3.12)
mkdir -p ~/s3diff
cd ~/s3diff
[ ! -d repo ] && git clone https://github.com/ArcticHare105/S3Diff.git repo 2>&1 | tail -3
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

pip install -q 'huggingface_hub[cli]' diffusers==0.34.0 'peft>=0.15.0' accelerate einops omegaconf timm scipy pyyaml scipy 2>&1 | tail -3
pip install -q 'setuptools<81' 2>&1 | tail -2

# 4. Patch torchvision.transforms.functional_tensor (removed in tv 0.17+)
TV_PATH=$(python -c 'import torchvision, os; print(os.path.dirname(torchvision.__file__))')
cat > $TV_PATH/transforms/functional_tensor.py <<'SHIM'
from torchvision.transforms.functional import *  # noqa: F401,F403
from torchvision.transforms.functional import rgb_to_grayscale  # noqa: F401
SHIM

# 5. Patch S3Diff: cuda -> cpu
cd ~/s3diff/repo/src
cp s3diff_tile.py s3diff_tile.py.bak 2>/dev/null || true
sed -i 's/\.cuda()/\.cpu()/g; s/to("cuda")/to("cpu")/g; s/device="cuda"/device="cpu"/g' s3diff_tile.py
sed -i 's/\.cuda()/\.cpu()/g; s/device="cuda"/device="cpu"/g' model.py
sed -i 's|torch.device("cuda")|torch.device("cpu")|g' my_utils/devices.py

# 6. Download checkpoints
export HF_TOKEN=$(cat ~/credentials/hugging-face-token.txt 2>/dev/null || echo "")
mkdir -p ~/s3diff/models ~/s3diff/smoke_in
cd ~/s3diff/models
huggingface-cli download zhangap/S3Diff --local-dir ./S3Diff 2>&1 | tail -2
huggingface-cli download stabilityai/sd-turbo --local-dir ./sd-turbo 2>&1 | tail -2

# 7. Prepare LQ inputs
python - <<'PY'
from PIL import Image
import urllib.request, os
# Reuse previously downloaded sample images if any, else fetch one bridge image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/1280px-GoldenGateBridge-001.jpg"
dst = "/home/ubuntu/s3diff/test_images/face.jpg"
os.makedirs("/home/ubuntu/s3diff/test_images", exist_ok=True)
if not os.path.exists(dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(dst, "wb") as f: f.write(r.read())
im = Image.open(dst).convert("RGB")
s = min(im.size); cx, cy = im.size[0]//2, im.size[1]//2
im = im.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2))
for side in [256, 512, 1024]:
    im.resize((side, side), Image.BICUBIC).save(f"/home/ubuntu/s3diff/smoke_in/bridge_LQ_{side}.png")
print("LQ inputs ready")
PY

echo "=== SETUP DONE ==="
