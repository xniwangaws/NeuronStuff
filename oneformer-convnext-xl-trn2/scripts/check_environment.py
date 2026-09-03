#!/usr/bin/env python3

import json
import platform
import subprocess

import PIL
import numpy
import scipy
import torch
import torch_neuronx
import torchvision
import transformers


def main() -> None:
    neuron_ls = subprocess.run(
        ["neuron-ls"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "unknown"),
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "scipy": scipy.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "neuron_ls": neuron_ls.stdout.strip(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
