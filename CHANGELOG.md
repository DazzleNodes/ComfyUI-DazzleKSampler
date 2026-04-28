# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Dazzle TauSampler node** -- tau complement sampling based on the Tau
  Operator from D. Darcy's Scarcity Framework
  - New node: "Dazzle TauSampler (DazzleNodes)" with clean, focused UI
  - Tau sampler variants: tau/res_2m, tau/res_2s, tau/dpmpp_2m,
    tau/dpmpp_2m_sde, tau/dpmpp_2s, tau/dpmpp_3m
  - Three modes: hard (fixed strength), soft (sigma-aware), cosine (smooth ramp)
  - tau_strength=0 produces bit-identical output to standard samplers
  - Params passed via extra_options (EO mechanism), RES4LYF compatible
  - Note: v1 implementation is a simplified complement (x_0 - x_next).
    Future versions will implement proper structure/noise separation
    in the complement via the resolution function R.

## [0.1.1-alpha] - 2026-04-28

### Fixed
- **`seed=-2` dual-role bug** in noise passthrough mode. Previously, when an
  upstream latent had `use_as_noise=True` but **no separate `noise` key**
  (Shape B in the protocol), the same tensor was passed to the underlying
  ComfyUI sampler in BOTH the noise role AND the init-image role —
  anchoring composition through `initial_x = z_norm(samples) * sigmas[0] + samples`.
  This caused upstream noise-shaping changes (`fill_type`, `blend_strength`)
  to have minimal visible effect, while the latent's own structure dominated
  output. Fix zeroes the init slot when latent-as-noise mode activates without
  a separate noise tensor, so the latent acts purely as noise:
  `initial_x = z_norm(samples) * sigmas[0] + 0`.
  - Shape C (img2img + img2noise, with `noise` key) is **unchanged** —
    img2img-with-shaped-noise semantics preserved.
  - Shapes A and D (no `use_as_noise` flag) are **unchanged** — `seed=-2`
    falls through to deterministic seed-driven noise generation.

### Added
- **Latent-dict protocol module** (`py/beta/_latent_noise_protocol.py`) —
  documents the four-shape protocol (A: pure init, B: pure noise, C: layered,
  D: empty) and centralizes the `seed=-2` dispatch logic in a pure helper.
- **Test suite** (`tests/test_latent_noise_protocol.py`) — 13 cases covering
  every shape × seed combination, including a regression guard for Shape C
  (img2img + img2noise) and a deterministic-seed-noise guard for Shape D
  (the `fill_type=black` case).

### Changed
- **`docs/wiki/Noise-Passthrough.md`** rewritten with the four-shape protocol
  explainer, the math behind the fix, and caveats about which latents do
  and don't trigger passthrough mode (notably: `fill_type=black`, stock
  `EmptyLatentImage`, and `image_purpose=img2img` alone all fall through
  to seed-driven noise).

## [0.1.0-alpha] - 2026-03-31

### Added
- Initial DazzleKSampler node based on RES4LYF ClownsharKSampler (AGPL-3.0)
- 6 nodes: DazzleKSampler, DazzleKSampler_Advanced, DazzleKSampler_Chain,
  DazzleSharkSampler, DazzleClownSampler, DazzleBongSampler
- Wiki documentation: samplers, schedulers, noise passthrough, config tips
