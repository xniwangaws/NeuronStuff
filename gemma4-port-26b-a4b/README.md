# gemma4-port-26b-a4b

Dry-run port of `google/gemma-4-26B-A4B-it` (MoE, 25.2 B total / 3.8 B active) to NxDI for AWS Trainium 2.

## Layout

```
gemma4-port-26b-a4b/
├── neuron_port/                    # ← the deliverable (NxDI implementation)
│   ├── modeling_gemma4_neuron.py
│   ├── configuration_gemma4_neuron.py
│   ├── __init__.py
│   └── README.md                   # full port documentation
├── agent_artifacts/
│   ├── traces/
│   │   ├── architecture_analysis.md
│   │   └── port_summary.md
│   └── tmp/
│       └── validation_config.json
├── transformers_src/   (gitignored — upstream HF source, read-only)
└── reference_prs/      (gitignored — large reference materials)
```

## Status

**Code-only dry-run.** No hardware available locally. The next session on a trn2 instance
should be able to compile + smoke-test without rewriting code. Read
`neuron_port/README.md` and `agent_artifacts/traces/port_summary.md` first.

## Quickstart (when hardware exists)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
# Then follow the compile/validate instructions in neuron_port/README.md
```
