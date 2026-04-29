# ComfyUI DazzleKSampler

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI-Registry-green.svg)](https://registry.comfy.org/publishers/djdarcy/nodes/comfyui-dazzle-ksampler)
[![GitHub release](https://img.shields.io/github/v/release/DazzleNodes/ComfyUI-DazzleKSampler?include_prereleases&label=version)](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

Enhanced KSampler nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Built on the [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) sampling engine (100+ RK methods, schedulers, ancestral variants) and adds first-class support for **shaped-noise passthrough** — letting upstream nodes (notably [Smart Resolution Calculator](https://github.com/DazzleNodes/ComfyUI-Smart-Resolution-Calc)) feed structured noise tensors into the sampler instead of relying on seed-driven Gaussian noise. Part of the [DazzleNodes](https://github.com/DazzleNodes/DazzleNodes) collection.

## What this is for

Stock ComfyUI samplers always generate noise from a seed. If you want to influence composition through the noise itself — using fill patterns, image-shaped noise, or spectrally blended composites from an upstream node — the seed-driven path overwrites everything you tried to set up. DazzleKSampler closes that gap. When the upstream latent dict carries a shaped-noise payload, the sampler uses it directly. When it doesn't, behavior is identical to the standard KSampler. The result: **the noise tensor your upstream node produced is the noise tensor the sampler diffuses against** — no double-anchoring, no silent overrides.

## Headline features

### Explicit `latent_role` widget *(v0.1.2-alpha)*

Every Dazzle sampler node exposes a `latent_role` dropdown that controls how the upstream latent dict is interpreted:

| Role | Behavior |
|------|----------|
| `auto` *(default)* | Inspect the dict shape and dispatch automatically. Honors `use_as_noise=True` regardless of seed value. |
| `noise` | Use the upstream tensor as noise; init slot zeroed. |
| `latent_image` | Standard img2img: samples are the init, noise comes from the seed. |
| `noise+latent_image` | Layered: noise from the upstream `noise` key, init from `samples`. |
| `seed_driven` | True txt2img override: zero init, noise from seed. Ignores upstream flags. |

The seed widget hides automatically when `latent_role` doesn't use it. `auto` is the right default for almost every workflow — explicit roles exist for power users and for diagnosing mismatches.

### Four-shape latent-dict protocol

A formalized contract for what an upstream node can put in the `LATENT` socket:

| Shape | Dict | Produced when |
|-------|------|---------------|
| **A — Pure init** | `{ samples: vae_encoded }` | Standard img2img |
| **B — Pure noise** | `{ samples: shaped_noise, use_as_noise: True }` | SmartResCalc with `dimensions only`, `img2noise`, or `image + noise` |
| **C — Layered** | `{ samples: encoded, noise: shaped_noise, use_as_noise: True }` | SmartResCalc with `img2img + img2noise` |
| **D — Empty** | `{ samples: zeros }` | `EmptyLatentImage` or `fill_type ∈ {black, white, custom_color}` |

Documented in detail in [`docs/wiki/Noise-Passthrough.md`](docs/wiki/Noise-Passthrough.md). The dispatch logic lives in a single pure helper module ([`py/beta/_latent_noise_protocol.py`](py/beta/_latent_noise_protocol.py)) with a 35-case test matrix covering every role × shape combination, so other custom-node authors can rely on the contract.

### Shape B init-zeroing fix *(v0.1.1-alpha)*

Earlier alpha versions of the latent-as-noise path had a dual-role bug: the same tensor was passed in both the noise role and the init-image role, anchoring composition through `initial_x = z_norm(samples) * sigmas[0] + samples`. Symptom was that changing upstream noise-shaping (`fill_type`, `blend_strength`) had near-zero visible effect. Fixed in v0.1.1 by zeroing the init slot when latent-as-noise mode activates without a separate noise tensor. See the [v0.1.1-alpha release notes](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/releases/tag/v0.1.1-alpha) for the math.

## Installation

### Via ComfyUI Manager (recommended)

Install through the ComfyUI Manager UI — search for **"Dazzle KSampler"**. Pulls the latest registry release.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DazzleNodes/ComfyUI-DazzleKSampler.git
```

Restart ComfyUI. Nodes appear under `RES4LYF/samplers` (will move to `DazzleNodes/Sampling` — see [#6](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/issues/6)).

## Nodes

| Node | Purpose |
|------|---------|
| **Dazzle KSampler** | All-in-one sampler (model + conditioning + latent → latent). Has the `latent_role` widget. |
| **Dazzle KSampler Advanced** | Returns a `SAMPLER` object for use with `SamplerCustomAdvanced`. Has the `latent_role` widget. |
| **Dazzle KSampler Chain** | Continues from a previous run's state (orchestrator pattern). |
| **Dazzle Shark Sampler** | Split orchestrator that accepts a separate `SAMPLER` object. Has the `latent_role` widget. |
| **Dazzle Clown Sampler** | Ported from RES4LYF. |
| **Dazzle Bong Sampler** | Ported from RES4LYF. Has the `latent_role` widget. |
| **Dazzle Tau Sampler** | Preview — see below. |

## Quick start: noise passthrough with SmartResCalc

1. **[SmartResCalc](https://github.com/DazzleNodes/ComfyUI-Smart-Resolution-Calc)** — pick an `image_purpose` that produces shaped noise:
   - `dimensions only` + a non-trivial `fill_type` (e.g. `DazNoise:Brown`) → Shape B
   - `img2noise` (with image attached) → Shape B
   - `image + noise` → Shape B
   - `img2img + img2noise` → Shape C
2. **DazzleKSampler** — leave `latent_role` on `auto`. Pick any seed value.
3. **Connect** SmartResCalc's `LATENT` output to DazzleKSampler's `latent_image` input.

`fill_type ∈ {black, white, custom_color}` and stock `EmptyLatentImage` produce Shape D and fall through to seed-driven noise — that's by design.

## Tau Sampler *(preview)*

`Dazzle Tau Sampler` is in the repo as a preview. It implements a simplified form of **tau complement sampling** based on the Tau Operator from D. Darcy's Scarcity Framework — a sampling step that subtracts a residual from each prediction (`x_0 - x_next`) at controllable strength.

Variants: `tau/res_2m`, `tau/res_2s`, `tau/dpmpp_2m`, `tau/dpmpp_2m_sde`, `tau/dpmpp_2s`, `tau/dpmpp_3m`. Three modes: `hard` (fixed strength), `soft` (sigma-aware), `cosine` (smooth ramp).

`tau_strength = 0` is bit-identical to the standard variant — safe to leave at zero in production workflows. Treat values above zero as experimental.

**Caveat (verbatim from CHANGELOG):** *v1 implementation is a simplified complement (`x_0 - x_next`). Future versions will implement proper structure/noise separation in the complement via the resolution function R.* The `tau4` spectral per-bin variant exists in the codebase but is not yet wired to a widget.

## Development

```bash
git clone https://github.com/DazzleNodes/ComfyUI-DazzleKSampler.git
cd ComfyUI-DazzleKSampler
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install torch  # required by the test harness
```

Run tests via the manual harness — `pytest` collides with this project's local `py/` directory (the project's `py/` package shadows the PyPI `py` package that pytest depends on, so `python -m pytest` fails at import time):

```bash
python tests/test_latent_noise_protocol.py
```

The harness exits with code 1 on any failure, so CI is honest. All 35 cases cover shape detection, legacy `seed=-2` parity, the full 5×4 role × shape dispatch matrix, auto-mode + normal seed paths, protocol invariants, and defensive paths.

## Documentation

- [`docs/wiki/Noise-Passthrough.md`](docs/wiki/Noise-Passthrough.md) — full four-shape protocol, dispatch matrix, migration from `seed=-2`
- [`CHANGELOG.md`](CHANGELOG.md) — release-by-release breakdown
- [Issues](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/issues) — known limitations and roadmap

## Acknowledgements

The core sampling engine — RK solver mathematics, noise generation, scheduler infrastructure, and the six base nodes — is a port of [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) by **ClownsharkBatwing**. DazzleNodes-specific additions (the four-shape protocol, the `latent_role` widget, Shape B init-zeroing, TauSampler) are by D. Darcy.

## License

AGPL-3.0 inherited from upstream RES4LYF. See [LICENSE](LICENSE) for details.
