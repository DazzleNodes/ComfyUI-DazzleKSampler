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

## [0.1.0-alpha] - 2026-03-31

### Added
- Initial DazzleKSampler node based on RES4LYF ClownsharKSampler (AGPL-3.0)
- 6 nodes: DazzleKSampler, DazzleKSampler_Advanced, DazzleKSampler_Chain,
  DazzleSharkSampler, DazzleClownSampler, DazzleBongSampler
- Wiki documentation: samplers, schedulers, noise passthrough, config tips
