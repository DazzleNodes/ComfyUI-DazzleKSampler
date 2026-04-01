# ComfyUI DazzleKSampler

Enhanced KSampler for ComfyUI with SmartResCalc noise passthrough and DazzleCommand integration. Part of the [DazzleNodes](https://github.com/DazzleNodes/DazzleNodes) collection.

Based on [RES4LYF](https://github.com/ClownsharkBeta/RES4LYF)'s ClownsharKSampler sampling engine.

## Features

- All RES4LYF beta samplers and schedulers (100+ RK methods)
- SmartResCalc `-2` seed passthrough (use input latent as noise)
- Coexists with RES4LYF (unique node class names)
- DazzleCommand integration (planned)

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DazzleNodes/ComfyUI-DazzleKSampler.git
```

### Development Setup

```bash
git clone https://github.com/DazzleNodes/ComfyUI-DazzleKSampler.git
cd ComfyUI-DazzleKSampler
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Install git hooks (if using repokit-common submodule)
bash scripts/install-hooks.sh
```

## Nodes

| Node | Description |
|------|-------------|
| **Dazzle KSampler** | All-in-one sampler (model + conditioning + latent in, latent out) |
| **Dazzle KSampler Advanced** | SAMPLER output for use with SamplerCustomAdvanced |
| **Dazzle KSampler Chain** | Chained sampling (continues from previous run's state) |
| **Dazzle Shark Sampler** | Split orchestrator (accepts separate SAMPLER object) |

## Acknowledgements

This project is built on the sampling engine from [RES4LYF](https://github.com/ClownsharkBeta/RES4LYF) by ClownsharkBeta. The RK solver mathematics, noise generation system, and scheduler infrastructure are derived from that work.

## License

AGPL-3.0 with commercial restriction. See [LICENSE](LICENSE) for details.