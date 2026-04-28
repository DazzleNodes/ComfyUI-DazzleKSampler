"""Tests for the latent-dict protocol resolver in `_latent_noise_protocol`.

Covers the four-shape protocol × {seed=-2, seed=normal} combinations.

Mapping to the design doc's test table:
- Test 1:  Shape A + seed=-2          → not_applied
- Test 2:  Shape B + seed=-2          → shape_b (the FIX path)
- Test 3:  Shape B noise-fill + seed=-2 → shape_b
- Test 4:  Shape B image+noise + seed=-2 → shape_b
- Test 5:  Shape C + seed=-2          → shape_c (preserved behavior)
- Test 6:  Shape D + seed=-2          → not_applied
- Test 7:  Shape B + seed=12345       → not_applied
- Test 8:  Shape C + seed=12345       → not_applied
- Test 12: Shape A no use_as_noise + seed=-2 → not_applied
- Test 13: Shape D zeros + seed=-2    → not_applied (deterministic noise from seed)
"""

import torch

try:
    import pytest
except ImportError:
    pytest = None  # tests run via manual harness in __main__ as fallback

# Load the helper module directly by path. The project's local `py/` directory
# collides with the site-packages `py` package (pytest dependency), so we
# bypass normal package resolution.
import importlib.util
from pathlib import Path
_HELPER_PATH = Path(__file__).resolve().parent.parent / "py" / "beta" / "_latent_noise_protocol.py"
_spec = importlib.util.spec_from_file_location("_latent_noise_protocol", _HELPER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resolve_latent_as_noise = _mod.resolve_latent_as_noise


# ---------- fixtures ----------

LATENT_SHAPE_4D = (1, 4, 64, 64)        # SDXL-style
LATENT_SHAPE_5D = (1, 16, 1, 152, 114)  # qwen/Wan video VAE


def _seeded_tensor(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def make_shape_A(shape=LATENT_SHAPE_4D, seed=42):
    """Pure init — img2img. No use_as_noise flag."""
    return {"samples": _seeded_tensor(shape, seed)}


def make_shape_B(shape=LATENT_SHAPE_4D, seed=42):
    """Pure noise — dimensions only / img2noise / image+noise."""
    return {"samples": _seeded_tensor(shape, seed), "use_as_noise": True}


def make_shape_C(shape=LATENT_SHAPE_4D, seed=42):
    """Layered — img2img + img2noise."""
    return {
        "samples": _seeded_tensor(shape, seed),       # encoded image (init)
        "noise":   _seeded_tensor(shape, seed + 1),   # shaped noise
        "use_as_noise": True,
    }


def make_shape_D(shape=LATENT_SHAPE_4D):
    """Empty — txt2img / EmptyLatentImage."""
    return {"samples": torch.zeros(shape)}


# ---------- pure helper tests ----------

def test_1_shape_A_seed_minus_2_not_applied():
    """Shape A (img2img, no use_as_noise) + seed=-2 → falls through to seed-driven."""
    latent = make_shape_A()
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is None
    assert torch.equal(x_out, x_orig)
    assert applied == "not_applied"


def test_2_shape_B_seed_minus_2_zeros_init_THE_FIX():
    """Shape B (pure noise) + seed=-2 → noise = samples, init = zeros."""
    latent = make_shape_B()
    x = latent["samples"].clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is not None
    assert torch.equal(noise, latent["samples"])    # noise IS the samples (fresh clone)
    assert x_out.abs().max().item() == 0.0          # init is zeroed — THE FIX
    assert applied == "shape_b"


def test_3_shape_B_noise_fill_seed_minus_2_zeros_init():
    """Shape B variant (different content) + seed=-2 → same fix path."""
    latent = make_shape_B(seed=99)
    x = latent["samples"].clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is not None
    assert torch.equal(noise, latent["samples"])
    assert x_out.abs().max().item() == 0.0
    assert applied == "shape_b"


def test_4_shape_B_5D_video_latent_seed_minus_2_zeros_init():
    """Shape B with 5D (video) latent shape — fix works for any shape."""
    latent = make_shape_B(shape=LATENT_SHAPE_5D, seed=7)
    x = latent["samples"].clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is not None
    assert noise.shape == LATENT_SHAPE_5D
    assert x_out.shape == LATENT_SHAPE_5D
    assert x_out.abs().max().item() == 0.0
    assert applied == "shape_b"


def test_5_shape_C_seed_minus_2_preserves_init_AND_uses_noise_key():
    """Shape C (img2img + img2noise) + seed=-2 → both roles preserved.

    THIS IS THE REGRESSION GUARD for the case the user worried about.
    The fix must NOT zero the init when a separate "noise" key is present.
    """
    latent = make_shape_C()
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is not None
    # noise comes from the "noise" key, NOT from samples
    assert torch.equal(noise, latent["noise"])
    assert not torch.equal(noise, latent["samples"])
    # x (init image) is preserved — img2img semantics intact
    assert torch.equal(x_out, x_orig)
    assert applied == "shape_c"


def test_6_shape_D_zeros_seed_minus_2_not_applied():
    """Shape D (empty/EmptyLatentImage, no flags) + seed=-2 → falls through.

    Documents the user's pixel-identical 'fill_type=black' test result:
    Shape D never enters the latent-as-noise branch; seed=-2 is handled
    downstream by the seed-driven noise generator.
    """
    latent = make_shape_D()
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is None
    assert torch.equal(x_out, x_orig)
    assert applied == "not_applied"


def test_7_shape_B_seed_normal_not_applied():
    """Shape B + seed=12345 → falls through (seed != -2 means no opt-in)."""
    latent = make_shape_B()
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=12345)
    assert noise is None
    assert torch.equal(x_out, x_orig)
    assert applied == "not_applied"


def test_8_shape_C_seed_normal_not_applied():
    """Shape C + seed=12345 → falls through (noise key ignored when seed != -2)."""
    latent = make_shape_C()
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=12345)
    assert noise is None
    assert torch.equal(x_out, x_orig)
    assert applied == "not_applied"


def test_12_shape_A_use_as_noise_false_seed_minus_2_not_applied():
    """Explicit use_as_noise=False + seed=-2 → still falls through.

    Belt-and-suspenders: ensures the dispatch checks use_as_noise truthily,
    not just dict-key presence.
    """
    latent = {"samples": _seeded_tensor(LATENT_SHAPE_4D, 42), "use_as_noise": False}
    x = latent["samples"].clone()
    x_orig = x.clone()
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is None
    assert torch.equal(x_out, x_orig)
    assert applied == "not_applied"


def test_13_shape_D_zeros_samples_seed_minus_2_falls_through():
    """Shape D with zeros samples + seed=-2 → falls through unchanged.

    This codifies the behavior the user empirically verified with
    fill_type=black: zeros latent + seed=-2 → seed-driven noise path.
    A future 'fix' that zeros x for Shape D would break this assertion.
    """
    latent = make_shape_D()
    x = latent["samples"].clone()
    assert x.abs().max().item() == 0.0  # sanity: it's zeros to start
    noise, x_out, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert noise is None
    # x stays as zeros (it was zeros to begin with) — but not because we zeroed it
    assert torch.equal(x_out, x)
    assert applied == "not_applied"


# ---------- additional protocol invariants ----------

def test_shape_B_noise_is_independent_clone_not_alias():
    """Mutating returned noise must not affect latent['samples']."""
    latent = make_shape_B()
    x = latent["samples"].clone()
    noise, x_out, _ = resolve_latent_as_noise(latent, x, noise_seed=-2)
    original_sum = float(latent["samples"].sum().item())
    noise.mul_(0)  # mutate noise in-place
    assert float(latent["samples"].sum().item()) == original_sum


def test_shape_B_x_out_is_independent_zeros_not_alias():
    """Mutating returned x_out must not affect latent['samples']."""
    latent = make_shape_B()
    x = latent["samples"].clone()
    _, x_out, _ = resolve_latent_as_noise(latent, x, noise_seed=-2)
    original_sum = float(latent["samples"].sum().item())
    x_out.add_(1.0)  # mutate x_out
    assert float(latent["samples"].sum().item()) == original_sum


def test_shape_C_noise_dtype_device_match_x():
    """Returned noise should be cast to x's dtype/device (the existing .to() call)."""
    latent = make_shape_C()
    # Simulate x being float32 on cpu while noise is float64 on cpu
    latent["noise"] = latent["noise"].to(dtype=torch.float64)
    x = latent["samples"].clone().to(dtype=torch.float32)
    noise, _, applied = resolve_latent_as_noise(latent, x, noise_seed=-2)
    assert applied == "shape_c"
    assert noise.dtype == x.dtype


if __name__ == "__main__":
    # Manual harness — works without pytest
    import sys, traceback
    _mod = sys.modules[__name__]
    _tests = [(n, getattr(_mod, n)) for n in dir(_mod)
              if n.startswith("test_") and callable(getattr(_mod, n))]
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
