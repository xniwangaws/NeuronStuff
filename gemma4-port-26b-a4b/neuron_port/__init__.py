# NeuronX Distributed Inference port of google/gemma-4-26B-A4B-it.
# See README.md for status and usage.
#
# This package requires NxDI / NxD installed (e.g. trn2 host with the
# `aws_neuronx_venv_pytorch_2_9_nxd_inference` venv). The configuration shim
# is parseable without NxDI; modeling is not.

from .configuration_gemma4_neuron import Gemma4TextConfig  # noqa: F401

__all__ = [
    "Gemma4TextConfig",
]
