# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.4-alpha] - 2026-04-30

### Added
- **DazNoise integration** — `noise_type_init`, `noise_type_sde`, and
  `noise_type_sde_substep` dropdowns on `DazzleKSampler` (and
  `noise_type_init` on `SharkSampler`) now include the DazNoise palette
  (`DazNoise: Plasma / Pink / Brown / Greyscale / Gaussian`) when the
  `dazzle-comfy-plasma-fast` package is installed. Detection mirrors the
  pattern SmartResCalc uses: a `sys.modules` walk for `JDC_OmniNoise` plus a
  `folder_paths` custom-nodes fallback, both gated try/except so DazNoise
  remains a soft dependency.
- **`noise_all` master selector** on `DazzleKSampler` — a top-of-noise-group
  combo that drives all three `noise_type_*` sub-widgets. When set to any
  value other than `custom`, the three sub-widgets are hidden from the UI
  via splice-based widget-array manipulation in `web/dazzle_ksampler.js`,
  and their values are force-synced server-side (defense in depth: works
  even if the JS is bypassed or the noise_all socket is wired as input).
  Default is `gaussian`, preserving v0.1.3-alpha output for users who do
  not touch the new widget. Selecting `custom` reveals the three sub-widgets
  with their last-edited values.
- **`daznoise_adapter.py`** — new module wrapping DazNoise's RGB image
  generators (which produce `(1, H, W, 3)` tensors in [0,1]) in the
  RES4LYF `NoiseGenerator` interface (`(B, C, H, W)` ~unit-variance
  Gaussian). Adapter pipeline: generate RGB → mean across channels →
  broadcast to latent channels → normalize to mean=0, std=1. On generator
  failure the adapter falls back to `torch.randn_like(x)` rather than
  crashing the sampler.
- **`tests/test_daznoise_adapter.py`** — 11 tests covering detection-absent
  (returns base RES4LYF list unchanged), detection-present (DazNoise types
  appended, factory returned for resolve), adapter behavior (4D + 5D latent
  shapes produce normalized noise, seed determinism), and failure fallback.

### Changed
- `noise_type_sde` and `noise_type_sde_substep` tooltips now flag that
  highly structured noise types (raw plasma, brown, pyramid) produce visual
  artifacts (chromatic fringing, halftoning) when used as per-step
  injection because the diffusion model expects ~unit-variance Gaussian
  statistics. Suggests SmartResCalc upstream + spectral_blend +
  `latent_role=noise` for shaped per-step noise without artifacts.
  `DazNoise: Gaussian` is called out as safe.
- `py/beta/rk_noise_sampler_beta.py:159-160` — SDE dispatch now uses
  `resolve_noise_class()` instead of a direct `NOISE_GENERATOR_CLASSES_SIMPLE`
  lookup, so DazNoise types resolve correctly when selected for per-step
  injection. Includes explicit `ValueError` on unknown types.

### Notes
- v0.1.3-alpha workflows load with `noise_all` defaulting to `gaussian`,
  which forces the three sub-widgets to `gaussian` and hides them. To
  recover v0.1.3-alpha widget exposure (independent control of each
  sub-widget), set `noise_all = custom`.
- DazNoise types are exposed on all three noise sub-widgets even though
  most of them produce per-step artifacts when used outside `noise_type_init`.
  This is intentional for the alpha — experimentation is the point. The
  tooltips warn; the per-step-noise whitelist proposed in #14 Phase 5 is
  deferred.

## [0.1.3-alpha] - 2026-04-29

### Fixed
- **[#12](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/issues/12)** —
  Seed widget permanently disappeared after wiring an input to the seed socket
  and then changing any other widget value. The hide-logic JS extension
  introduced in v0.1.2-alpha (`web/dazzle_ksampler.js`) mutated `widget.type`
  and `widget.computeSize` in ways that conflicted with ComfyUI's
  widget→input conversion machinery. Fixed by removing the hide logic
  entirely; the seed widget now stays visible regardless of `latent_role`.
- **Misleading documentation in v0.1.2-alpha** — README, wiki, tooltips, and
  CHANGELOG claimed the seed widget was "unused" under
  `latent_role ∈ {noise, noise+latent_image}`. The seed *is* still used:
  it independently feeds `init_noise_samplers` in
  `py/beta/rk_noise_sampler_beta.py:135-160`, which instantiates the
  per-step ancestral / SDE noise sampler. That sampler is called at every
  step (`super_sigma_up * NS.noise_sampler(...)`), so changing the seed
  changes per-step injection even when the initial noise tensor comes from
  upstream. Tooltips, README, and wiki language corrected.

### Changed
- The conditional seed-widget-hiding JS extension has been replaced with an
  empty registration shell. The hide rule was based on a wrong premise
  (per-step noise machinery still consumes the seed), and the mechanism
  itself was incompatible with widget→input conversion. The shell is
  preserved so future frontend-only enhancements have a stable import path.
- `latent_role` widget tooltip and `noise_seed` tooltip on every sampler
  node now state explicitly that `latent_role` controls only the *initial*
  noise tensor at sigma_max, and that the seed still drives per-step noise
  injection regardless of role.

### Added
- **Structured first-step noise-source banner.** Every generation prints a
  banner identifying the active noise-source configuration. Five variants
  cover the common cases: `auto`/`latent_image`, `seed_driven`, and three
  sub-cases for `noise`/`noise+latent_image` (seed=-1, seed=-2, and
  deterministic `seed≥0` — the latter recommends `eta=0` for full
  upstream-determinism).
- **"Determinism recipe" subsection** in README and wiki: with `eta=0`
  (collapses `super_sigma_up` to 0 under the default `noise_mode_sde="hard"`)
  and `latent_role=noise`, output is fully determined by the upstream
  noise tensor. Empirically verified 2026-04-29: KSampler seeds 5225, 9999,
  and -2 all produce bit-identical output under `eta=0`.
- **"Surprises / FAQ" section** in `docs/wiki/Noise-Passthrough.md`
  documenting why per-step noise often dominates visible variation when
  `blend_strength` is low (the shaped pattern only contributes a small
  fraction of the initial latent), and recommending the
  SmartResCalc.seed → DazzleKSampler.seed wiring pattern as a coherent
  single-knob workflow.
- **`noise_type_init`, `noise_type_sde`, `noise_type_sde_substep` widgets
  on DazzleKSampler** ([#13](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/issues/13)
  Phase 1+2). Lets users select the spectral content of the initial-noise
  budget and the per-step ancestral/SDE budget independently:
  brown/pink (low-frequency dominant), blue/violet (high-frequency
  dominant), plasma/pyramid_*/fractal (structured / multi-scale), and the
  default `gaussian`. All default to `gaussian` so behavior is identical
  to v0.1.2 unless the widgets are changed. Tooltips lead with
  *"Advanced. Leave at gaussian for default behavior."* and the widgets
  are positioned at the bottom of the required block to visually
  deprioritize them. Per-noise-type alpha/k/scale params remain deferred.
- **Regression test `test_seed_passes_through_dispatch_unchanged`** in
  `tests/test_latent_noise_protocol.py` pinning the seed-plumbing
  contract: the dispatch helper does not zero, override, or transform
  `noise_seed` for any role. Per-step machinery downstream depends on
  receiving the seed verbatim. Test count: 35 → 36.

## [0.1.2-alpha] - 2026-04-28

### Added
- **Explicit `latent_role` widget** on every Dazzle sampler node
  (Issue [#11](https://github.com/DazzleNodes/ComfyUI-DazzleKSampler/issues/11)).
  Replaces the seed=-2 magic with five explicit choices:
  - `auto` *(default)* — inspect dict shape, dispatch accordingly. Honors
    `use_as_noise=True` regardless of seed value (the legacy seed=-2 path
    still works under `auto`, with a deprecation log).
  - `noise` — force Shape B (samples as noise, init zeroed).
  - `latent_image` — force Shape A (samples as init, noise from seed).
  - `noise+latent_image` — force Shape C (init from samples, noise from
    `noise` key).
  - `seed_driven` — ignore upstream flags, generate noise from seed.
  - The seed widget is auto-hidden when `latent_role ∈ {noise,
    noise+latent_image}` (the seed is unused on those paths). Implemented
    via a JS extension at `web/dazzle_ksampler.js`.
  - Mismatch warnings printed to console when an explicit role doesn't
    match the detected dict shape (e.g., `latent_role=noise+latent_image`
    on a Shape A dict falls back to Shape A with a warning).
  - Auto-mode dispatching to Shape B/C without seed=-2 emits a one-line
    informational notice on the first step so users see the modern
    behavior is active (Option C semantics).
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

### Changed
- **`py/beta/_latent_noise_protocol.py`** upgraded from a 3-tuple return
  to a `LatentNoiseDispatch` dataclass with `noise`, `x`, `detected_shape`,
  `applied`, `warning`, `deprecation` fields. The new helper signature is
  `resolve_latent_as_noise(latent_unbatch, x, noise_seed,
  latent_role="auto") -> LatentNoiseDispatch`. The pure-function contract
  is preserved (no I/O, no logging — the dispatch site emits prints).
- **Tests rewritten** (`tests/test_latent_noise_protocol.py`) — 35 cases
  covering shape detection (5), legacy v0.1.1 parity with auto+seed=-2 (5),
  the full 5×4 explicit role × shape matrix (16), auto-mode + normal seed
  (4), protocol invariants (3), defensive (1), use_as_noise=False (1).
- **`docs/wiki/Noise-Passthrough.md`** — documents the new widget,
  dispatch matrix, migration path from seed=-2, and the seed-widget
  hiding rule.

### Fixed
- **`latent_role=noise` on Shape C upstream** previously cloned `samples`
  (the VAE-encoded init image) as the noise tensor, producing
  "image-as-noise" output where the encoded figure leaked through as a
  ghost silhouette. Now correctly uses the `noise` key (the upstream's
  actual noise tensor) and zeroes the init slot. To preserve the
  img2img anchor, use `latent_role=noise+latent_image` instead.
- **`latent_role=noise` on Shape A upstream** (no `use_as_noise` flag,
  no `noise` key) now falls back to seed-driven noise generation with a
  warning, instead of cloning the encoded image as noise (which produced
  the same ghost-silhouette artifact class).

### Changed (semantic, within v0.1.2-alpha pre-release)
- **`latent_role=seed_driven`** is now a true txt2img override: the init
  slot is zeroed AND noise is generated from the seed. Previously this
  was equivalent to `latent_image` (samples preserved as init). The
  change makes `seed_driven` distinct from `latent_image` so users can
  compare "img2img with this seed" (`latent_image`) vs "what this prompt
  would produce from scratch" (`seed_driven`) without changing upstream
  connections.

### Deprecated
- **`seed = -2`** as a dispatch trigger. Still works under `latent_role =
  auto` (with a one-time deprecation log per generation) but will lose
  its special meaning in a future release. Set `latent_role` explicitly
  to control dispatch.

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
