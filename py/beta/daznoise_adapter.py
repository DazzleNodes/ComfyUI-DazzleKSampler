"""
DazNoise adapter for DazzleKSampler's `noise_type_init` widget.

Detection mirrors SmartResCalc's `_get_plasma_fast()` pattern (sys.modules
walk + folder_paths fallback). When `dazzle-comfy-plasma-fast` is detected,
the DazNoise type names are appended to the `noise_type_init` dropdown via
`get_noise_choices()`.

The DazNoiseGeneratorAdapter class wraps DazNoise's RGB-image generators in
the RES4LYF NoiseGenerator interface so the dispatch site at
`samplers.py` (the `noise_sampler_init = ...` line) can call them with
the same shape as the existing Gaussian/Brown/etc. generators.

Why noise_type_init only:
  Per-step ancestral/SDE noise injection assumes ~unit-variance decorrelated
  Gaussian statistics. DazNoise produces structured RGB images which, even
  after grayscale + normalize, are spatially correlated. Using them as
  per-step noise produces visual artifacts similar to (and often worse than)
  the brown artifact described in v0.1.3-alpha's HV.5. The initial-noise
  canvas at sigma_max is the right place for structured input — the model
  treats it as something to "imagine into" rather than as injection
  statistics. Hence DazNoise types are exposed to noise_type_init only.

Spectral blending (deferred to a later phase):
  SmartResCalc's full Plasma pipeline blends the structured pattern with
  Gaussian noise spectrally (only low-frequency content from the structured
  pattern; Gaussian elsewhere). This adapter currently produces the raw
  structured pattern. For clean composition control, recommend using
  SmartResCalc upstream with `spectral_blend` and `latent_role=noise` on
  DazzleKSampler. Internal blending may be added in v0.1.4 Phase 3.5.
"""

import os
import sys
import logging
import random as py_random
import importlib.util

import torch

# NoiseGenerator import is wrapped so the structural helpers
# (detection, choices list, type-check, resolve) remain usable in
# environments where noise_classes' deps (pywt, comfy.k_diffusion)
# aren't available -- e.g. CI test runners that don't have a Comfy
# install. The adapter class itself becomes a stub in that case;
# resolve_noise_class still returns a callable factory but
# instantiating it at sample time will raise.
try:
    from .noise_classes import NoiseGenerator
    _NOISE_GENERATOR_AVAILABLE = True
except ImportError:
    NoiseGenerator = object  # stub base so class definition below parses
    _NOISE_GENERATOR_AVAILABLE = False

logger = logging.getLogger("DazzleKSampler")

# Type names appended to the `noise_type_init` dropdown when DazNoise is
# available. Order matches SmartResCalc's DAZNOISE_FILL_TYPES for parity.
DAZNOISE_FILL_TYPES = (
    "DazNoise: Pink",
    "DazNoise: Brown",
    "DazNoise: Plasma",
    "DazNoise: Greyscale",
    "DazNoise: Gaussian",
)

# fill_type -> (NODE_CLASS_MAPPINGS key, method_name, extra_kwargs)
# Lifted from SmartResCalc/py/noise_utils.py to keep behavior consistent.
_DAZNOISE_TYPE_MAP = {
    "DazNoise: Pink":      ("JDC_PinkNoise",  "generate_noise",  {}),
    "DazNoise: Brown":     ("JDC_BrownNoise", "generate_noise",  {}),
    "DazNoise: Plasma":    ("JDC_Plasma",     "generate_plasma", {"turbulence": 2.75}),
    "DazNoise: Greyscale": ("JDC_GreyNoise",  "generate_noise",  {}),
    "DazNoise: Gaussian":  ("JDC_OmniNoise",  "generate_noise",  {
        "noise_type": "Random",
        "random_distribution": "Gaussian (Centered Gray)",
    }),
}

# Cache: positive-only (None means "not yet checked or not found").
# Re-checks each call until found, since DazzleKSampler may load before
# dazzle-comfy-plasma-fast in some custom_nodes load orders.
_plasma_fast_module = None


def _get_plasma_fast():
    """Detect dazzle-comfy-plasma-fast NODE_CLASS_MAPPINGS, if available.

    Mirrors SmartResCalc's two-strategy detection:
      1. Walk sys.modules for a module with NODE_CLASS_MAPPINGS containing
         JDC_OmniNoise (a reliable indicator of dazzle-comfy-plasma-fast).
      2. Fallback to importlib path-based loading at known custom_nodes
         locations.

    Returns the NODE_CLASS_MAPPINGS dict or None.
    """
    global _plasma_fast_module
    if _plasma_fast_module is not None:
        return _plasma_fast_module

    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
        except Exception:
            # Some modules use custom __getattr__ that raise non-AttributeError
            # exceptions (e.g., ImportError from SeedVR2's compatibility.py).
            # Catch broadly to keep detection robust against third-party quirks.
            continue
        if isinstance(mappings, dict) and "JDC_OmniNoise" in mappings:
            _plasma_fast_module = mappings
            logger.debug(f"Found dazzle-comfy-plasma-fast via sys.modules: {name}")
            print("[DazzleKSampler] Detected dazzle-comfy-plasma-fast "
                  "(DazNoise types enabled for noise_type_init)")
            return mappings

    try:
        import folder_paths
        base = folder_paths.base_path
        candidates = [
            os.path.join(base, "custom_nodes", "dazzle-comfy-plasma-fast", "nodes.py"),
            os.path.join(base, "custom_nodes", "DazzleNodes",
                         "nodes", "dazzle-comfy-plasma-fast", "nodes.py"),
        ]
        for path in candidates:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location(
                    "_plasma_fast_nodes_dazksampler", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
                if isinstance(mappings, dict):
                    _plasma_fast_module = mappings
                    logger.debug(f"Found dazzle-comfy-plasma-fast via path: {path}")
                    print("[DazzleKSampler] Detected dazzle-comfy-plasma-fast "
                          "(DazNoise types enabled for noise_type_init)")
                    return mappings
    except Exception as e:
        logger.debug(f"Path-based DazNoise detection failed: {e}")

    return None


def is_daznoise_available():
    """True if DazNoise types should appear in the noise_type_init dropdown."""
    return _get_plasma_fast() is not None


def is_daznoise_type(name):
    """True if `name` is a `DazNoise: ...` type."""
    return name in _DAZNOISE_TYPE_MAP


def get_noise_choices(base_choices):
    """Append DazNoise types to `base_choices` if DazNoise is available.

    `base_choices` is the existing NOISE_GENERATOR_NAMES_SIMPLE tuple.
    Returns a tuple — keep mutation-free since INPUT_TYPES caches the list.
    """
    if is_daznoise_available():
        return tuple(base_choices) + DAZNOISE_FILL_TYPES
    return tuple(base_choices)


def resolve_noise_class(name, base_classes):
    """Return a noise-generator class/factory for the given init noise type.

    For RES4LYF-native types, returns the entry from `base_classes`
    (NOISE_GENERATOR_CLASSES_SIMPLE). For DazNoise types, returns a factory
    that constructs DazNoiseGeneratorAdapter with the right fill_type bound.

    Returns None if the name is unknown.
    """
    if is_daznoise_type(name):
        # Closure binds `name` so the dispatch site can call it like any
        # other entry: cls(x=..., seed=..., sigma_max=..., sigma_min=...)
        bound_name = name
        def factory(**kwargs):
            return DazNoiseGeneratorAdapter(fill_type=bound_name, **kwargs)
        return factory
    return base_classes.get(name)


def _generate_daznoise_image(fill_type, width, height, seed):
    """Call DazNoise's generator and return its (1, H, W, 3) RGB tensor.

    Returns None on failure (logs the cause).
    """
    mappings = _get_plasma_fast()
    if mappings is None:
        return None

    node_id, method_name, extra_kwargs = _DAZNOISE_TYPE_MAP[fill_type]
    generator_class = mappings.get(node_id)
    if generator_class is None:
        logger.warning(
            f"DazNoise node '{node_id}' missing from NODE_CLASS_MAPPINGS; "
            f"falling back to gaussian for fill_type='{fill_type}'.")
        return None

    generator = generator_class()
    generate_fn = getattr(generator, method_name)

    try:
        # value_min/max=-1, red/green/blue_min/max=-1 are the "use defaults"
        # sentinels in dazzle-comfy-plasma-fast's generator API.
        result = generate_fn(
            width=width, height=height,
            value_min=-1, value_max=-1,
            red_min=-1, red_max=-1,
            green_min=-1, green_max=-1,
            blue_min=-1, blue_max=-1,
            seed=seed,
            **extra_kwargs,
        )
        return result[0]  # (1, H, W, 3)
    except Exception as e:
        logger.error(f"DazNoise generation failed for '{fill_type}': {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def _normalize_to_unit_gaussian(t):
    """Center to mean=0 and scale to std=1.

    Mirrors the `(noise - noise.mean()) / noise.std()` pattern used
    throughout noise_classes.py.
    """
    std = t.std()
    if std < 1e-8:
        # Degenerate constant tensor; fall back to small Gaussian to avoid
        # division-by-zero. This shouldn't happen with real DazNoise output
        # but guards against pathological inputs.
        return torch.randn_like(t)
    return (t - t.mean()) / std


class DazNoiseGeneratorAdapter(NoiseGenerator):
    """Wraps DazNoise RGB image generators in the RES4LYF NoiseGenerator
    interface for use as `noise_type_init` only.

    Pipeline per __call__:
      1. Generate DazNoise RGB image at latent spatial resolution
      2. Convert RGB -> grayscale (mean across channels)
      3. Broadcast to all latent channels and batch
      4. Normalize to mean=0, std=1

    The grayscale + broadcast step loses inter-channel decorrelation that
    pure Gaussian noise has. This is acceptable for noise_type_init (the
    model imagines structure into the canvas regardless of channel
    correlation) but is part of why DazNoise is intentionally NOT exposed
    for noise_type_sde / noise_type_sde_substep — per-step injection
    needs decorrelated channels.
    """

    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None,
                 seed=42, generator=None, sigma_min=None, sigma_max=None,
                 fill_type="DazNoise: Plasma"):
        super().__init__(x, size, dtype, layout, device,
                         seed, generator, sigma_min, sigma_max)
        self.fill_type = fill_type

    def __call__(self, *, sigma=None, sigma_next=None, **kwargs):
        self.last_seed += 1

        # Latent shape: (B, C, H, W) for 4D models, (B, C, T, H, W) for 5D.
        size = self.size
        if len(size) == 4:
            B, C, H, W = size
            T = None
        elif len(size) == 5:
            B, C, T, H, W = size
        else:
            raise ValueError(
                f"DazNoiseGeneratorAdapter: unexpected latent shape {size}")

        # DazNoise calls py_random internally to derive the actual generator
        # seed. Seed py_random deterministically from our last_seed so the
        # same DazzleKSampler seed produces the same DazNoise pattern.
        py_random.seed(int(self.last_seed) & 0xFFFFFFFF)
        daznoise_seed = py_random.randint(0, 2**32 - 1)

        rgb = _generate_daznoise_image(self.fill_type, W, H, daznoise_seed)
        if rgb is None:
            # Detection lost / generator missing / generation failed -- fall
            # back to gaussian noise so the sampler can still complete.
            return torch.randn(self.size, dtype=self.dtype,
                               layout=self.layout, device=self.device,
                               generator=self.generator)

        # rgb: (1, H, W, 3) on CPU in [0, 1]. Move to latent device/dtype.
        rgb = rgb.to(device=self.device, dtype=self.dtype)

        # Collapse RGB -> grayscale: (1, H, W).
        gray = rgb.mean(dim=-1, keepdim=False)

        # Broadcast to latent channel count and batch. expand() returns a
        # view; .contiguous() materializes an independent tensor.
        if T is None:
            noise = gray.unsqueeze(1).expand(B, C, H, W).contiguous()
        else:
            noise = gray.unsqueeze(1).unsqueeze(2).expand(B, C, T, H, W).contiguous()

        return _normalize_to_unit_gaussian(noise)
