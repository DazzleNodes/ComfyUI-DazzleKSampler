"""Tests for the DazNoise adapter module.

Coverage:

- Detection-absent path: when dazzle-comfy-plasma-fast is not loaded,
  get_noise_choices() returns the base RES4LYF list unchanged
  and resolve_noise_class() falls through to the RES4LYF lookup
  for native types.
- Detection-present path: when a stub NODE_CLASS_MAPPINGS containing
  JDC_OmniNoise is injected into sys.modules, DazNoise type names are
  appended to the choices list and resolve_noise_class() returns
  a factory for them.
- Adapter behavior: with a stub generator that returns a known RGB
  image, the adapter produces a normalized (mean~0, std~1) tensor at
  the latent's shape and channel count.
- 5D latent shape (Qwen/Wan video) compatibility.
- Failure fallback: when the DazNoise generator raises, the adapter
  falls back to gaussian rather than crashing.

Run from the repo root:
    python tests/test_daznoise_adapter.py

Exits 0 on success, 1 on any failure.
"""

import sys
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import torch

# Silence the adapter's logger -- the failure-path test deliberately
# triggers errors and we don't want the noise in test output.
logging.getLogger("DazzleKSampler").setLevel(logging.CRITICAL)

# Load the adapter module directly by path (same pattern as
# test_latent_noise_protocol.py -- the project's local py/ shadows the
# PyPI py package required by pytest).
_REPO = Path(__file__).resolve().parent.parent
_NOISE_PATH = _REPO / "py" / "beta" / "noise_classes.py"
_ADAPTER_PATH = _REPO / "py" / "beta" / "daznoise_adapter.py"

# noise_classes has heavy imports (pywt, comfy.k_diffusion); skip the
# tests that need it if those aren't available -- the structural tests
# (detection on/off, choices list) don't need NoiseGenerator.
try:
    _spec_nc = importlib.util.spec_from_file_location("noise_classes_for_test", _NOISE_PATH)
    _nc_mod = importlib.util.module_from_spec(_spec_nc)
    _spec_nc.loader.exec_module(_nc_mod)
    NoiseGenerator = _nc_mod.NoiseGenerator
    NOISE_CLASSES_AVAILABLE = True
except Exception as e:
    print(f"Note: noise_classes import failed ({e}); skipping adapter-instance tests.")
    NOISE_CLASSES_AVAILABLE = False


def _load_adapter_fresh():
    """Re-import the adapter with module-level cache reset.

    Each test that manipulates sys.modules state needs a fresh adapter
    so the `_plasma_fast_module` cache doesn't leak between tests.
    """
    spec = importlib.util.spec_from_file_location(
        "daznoise_adapter_test", _ADAPTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- detection-absent (default) tests ----------

def test_detection_absent_choices_unchanged():
    """Without DazNoise, get_noise_choices returns base list verbatim."""
    adapter = _load_adapter_fresh()
    base = ("gaussian", "brown", "pink")
    result = adapter.get_noise_choices(base)
    assert result == base, f"expected {base}, got {result}"


def test_detection_absent_is_daznoise_available_false():
    adapter = _load_adapter_fresh()
    assert adapter.is_daznoise_available() is False


def test_detection_absent_resolve_native_type():
    """Native RES4LYF types resolve via the base_classes lookup."""
    adapter = _load_adapter_fresh()
    fake_classes = {"gaussian": "GAUSSIAN_CLASS_SENTINEL"}
    result = adapter.resolve_noise_class("gaussian", fake_classes)
    assert result == "GAUSSIAN_CLASS_SENTINEL"


def test_detection_absent_resolve_unknown_returns_none():
    adapter = _load_adapter_fresh()
    assert adapter.resolve_noise_class("not_a_real_type", {}) is None


# ---------- detection-present (with stub NODE_CLASS_MAPPINGS) tests ----------

class _StubGenerator:
    """Stand-in for JDC_OmniNoise / JDC_Plasma / etc.

    Returns a deterministic (1, H, W, 3) tensor so the adapter can be
    exercised without dazzle-comfy-plasma-fast installed.
    """
    def __init__(self):
        self.last_call_kwargs = None

    def generate_noise(self, **kwargs):
        self.last_call_kwargs = kwargs
        H, W = kwargs["height"], kwargs["width"]
        seed = kwargs.get("seed", 0)
        # deterministic but seed-dependent so the test can verify seeding
        gen = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
        rgb = torch.rand((1, H, W, 3), generator=gen)
        return (rgb,)

    def generate_plasma(self, **kwargs):
        return self.generate_noise(**kwargs)


def _install_stub_daznoise():
    """Inject a fake module with JDC_OmniNoise into sys.modules."""
    stub_mod = ModuleType("_test_daznoise_stub")
    stub_mod.NODE_CLASS_MAPPINGS = {
        "JDC_OmniNoise":  _StubGenerator,
        "JDC_PinkNoise":  _StubGenerator,
        "JDC_BrownNoise": _StubGenerator,
        "JDC_Plasma":     _StubGenerator,
        "JDC_GreyNoise":  _StubGenerator,
    }
    sys.modules["_test_daznoise_stub"] = stub_mod
    return stub_mod


def _uninstall_stub_daznoise():
    sys.modules.pop("_test_daznoise_stub", None)


def test_detection_present_choices_appended():
    """With DazNoise stub installed, choices list is extended."""
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        base = ("gaussian", "brown")
        result = adapter.get_noise_choices(base)
        assert "gaussian" in result and "brown" in result
        assert "DazNoise: Plasma" in result
        assert "DazNoise: Pink" in result
        assert len(result) == len(base) + 5  # 5 DazNoise types
    finally:
        _uninstall_stub_daznoise()


def test_detection_present_is_daznoise_type():
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        assert adapter.is_daznoise_type("DazNoise: Plasma") is True
        assert adapter.is_daznoise_type("gaussian") is False
        assert adapter.is_daznoise_type("not_real") is False
    finally:
        _uninstall_stub_daznoise()


def test_detection_present_resolve_returns_factory():
    """For DazNoise types, resolve returns a callable factory (not a class)."""
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        result = adapter.resolve_noise_class("DazNoise: Plasma", {})
        assert callable(result)
        # The factory takes the same kwargs as a NoiseGenerator constructor.
        # We can't actually instantiate without NoiseGenerator's deps, so we
        # just verify it's a closure (not the same as the base lookup).
        assert result.__name__ == "factory"
    finally:
        _uninstall_stub_daznoise()


# ---------- adapter behavior tests (need NoiseGenerator) ----------

def test_adapter_produces_normalized_4d_latent():
    """Adapter output has correct shape, mean~0, std~1 for 4D latents."""
    if not NOISE_CLASSES_AVAILABLE:
        print("  SKIP  test_adapter_produces_normalized_4d_latent (noise_classes not loadable)")
        return
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        # 4-channel SDXL-style latent
        x = torch.zeros((1, 4, 64, 64))
        gen = adapter.DazNoiseGeneratorAdapter(
            x=x, seed=42,
            sigma_min=torch.tensor(0.029), sigma_max=torch.tensor(14.6),
            fill_type="DazNoise: Plasma")
        noise = gen()
        assert noise.shape == x.shape, f"shape mismatch: {noise.shape} vs {x.shape}"
        assert abs(noise.mean().item()) < 1e-5, f"mean not zero: {noise.mean().item()}"
        assert abs(noise.std().item() - 1.0) < 1e-3, f"std not 1: {noise.std().item()}"
    finally:
        _uninstall_stub_daznoise()


def test_adapter_produces_normalized_5d_latent():
    """5D video-latent (Qwen/Wan) compatibility."""
    if not NOISE_CLASSES_AVAILABLE:
        print("  SKIP  test_adapter_produces_normalized_5d_latent (noise_classes not loadable)")
        return
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        x = torch.zeros((1, 16, 1, 64, 48))  # Qwen-style
        gen = adapter.DazNoiseGeneratorAdapter(
            x=x, seed=42,
            sigma_min=torch.tensor(0.029), sigma_max=torch.tensor(14.6),
            fill_type="DazNoise: Pink")
        noise = gen()
        assert noise.shape == x.shape
        assert abs(noise.mean().item()) < 1e-5
        assert abs(noise.std().item() - 1.0) < 1e-3
    finally:
        _uninstall_stub_daznoise()


def test_adapter_seed_determinism():
    """Same seed -> same noise; different seed -> different noise."""
    if not NOISE_CLASSES_AVAILABLE:
        print("  SKIP  test_adapter_seed_determinism (noise_classes not loadable)")
        return
    _install_stub_daznoise()
    try:
        adapter = _load_adapter_fresh()
        x = torch.zeros((1, 4, 32, 32))

        def make(seed):
            return adapter.DazNoiseGeneratorAdapter(
                x=x, seed=seed,
                sigma_min=torch.tensor(0.029), sigma_max=torch.tensor(14.6),
                fill_type="DazNoise: Plasma")()

        a1 = make(42)
        a2 = make(42)
        b  = make(99)
        assert torch.equal(a1, a2), "same seed should produce identical noise"
        assert not torch.equal(a1, b), "different seeds should produce different noise"
    finally:
        _uninstall_stub_daznoise()


class _FailingGenerator:
    """Generator that raises on every call -- exercises the fallback path."""
    def generate_noise(self, **kwargs):
        raise RuntimeError("simulated DazNoise failure")
    def generate_plasma(self, **kwargs):
        raise RuntimeError("simulated DazNoise failure")


def test_adapter_falls_back_to_gaussian_on_generator_failure():
    """If the DazNoise generator raises, adapter returns gaussian noise
    (sized correctly) rather than crashing the sampler."""
    if not NOISE_CLASSES_AVAILABLE:
        print("  SKIP  test_adapter_falls_back_to_gaussian_on_generator_failure")
        return
    # Inject failing generators
    stub_mod = ModuleType("_test_daznoise_failing")
    stub_mod.NODE_CLASS_MAPPINGS = {
        "JDC_OmniNoise":  _FailingGenerator,
        "JDC_Plasma":     _FailingGenerator,
    }
    sys.modules["_test_daznoise_failing"] = stub_mod
    try:
        adapter = _load_adapter_fresh()
        x = torch.zeros((1, 4, 32, 32))
        gen = adapter.DazNoiseGeneratorAdapter(
            x=x, seed=42,
            sigma_min=torch.tensor(0.029), sigma_max=torch.tensor(14.6),
            fill_type="DazNoise: Plasma")
        noise = gen()
        # Must still be the right shape; values are gaussian-distributed.
        assert noise.shape == x.shape
    finally:
        sys.modules.pop("_test_daznoise_failing", None)


# ---------- manual harness ----------

if __name__ == "__main__":
    import traceback
    _module = sys.modules[__name__]
    _tests = [(n, getattr(_module, n)) for n in dir(_module)
              if n.startswith("test_") and callable(getattr(_module, n))]
    _passed = _failed = 0
    for _name, _fn in _tests:
        try:
            _fn()
            print(f"  PASS  {_name}")
            _passed += 1
        except Exception:
            print(f"  FAIL  {_name}")
            traceback.print_exc()
            _failed += 1
    print(f"\n--- {_passed}/{_passed + _failed} tests passed ---")
    sys.exit(0 if _failed == 0 else 1)
