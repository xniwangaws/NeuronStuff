# NeuronX Distributed Inference port of Google's Gemma-4-26B-A4B-it.
# See README.md for status and usage.

from .configuration_gemma4_neuron import (  # noqa: F401
    Gemma4TextConfig,
    make_gemma4_inference_config_class,
    make_gemma4_neuron_config_class,
)

# The modeling file imports NxDI / NxD at module-load time, so it can only be
# imported on a machine that has those packages installed (e.g. the trn2
# instance during compilation). We therefore expose the modeling symbols
# lazily — importing this package on a laptop without NxDI keeps working.

__all__ = [
    "Gemma4TextConfig",
    "make_gemma4_inference_config_class",
    "make_gemma4_neuron_config_class",
]


def _load_modeling():
    """Lazy import of the NxDI-dependent modeling module.

    Call ``neuron_port._load_modeling()`` (or simply
    ``from neuron_port.modeling_gemma4_neuron import ...``) on a host with
    NxDI installed. We keep this out of the eager import path so the package
    parses cleanly on a dev laptop.
    """

    from . import modeling_gemma4_neuron  # noqa: PLC0415

    return modeling_gemma4_neuron
