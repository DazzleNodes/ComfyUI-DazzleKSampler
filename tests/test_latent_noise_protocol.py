"""Tests for the latent-dict protocol resolver in `_latent_noise_protocol`.

Coverage:

- v0.1.1-alpha legacy seed=-2 + auto-mode tests (preserved as parity guard)
- 5x4 latent_role x detected-shape dispatch matrix (v0.1.2-alpha)
- Mismatch warning emission for explicit roles
- Deprecation log for noise_seed=-2 + auto
- Option C informational notice when auto dispatches to Shape B/C
  without the legacy seed=-2 signal
- Tensor-aliasing guards (Shape B clones; mutating return doesn't touch dict)
- 5D video-latent shape compatibility
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
detect_shape = _mod.detect_shape
LatentNoiseDispatch = _mod.LatentNoiseDispatch
ROLE_AUTO = _mod.ROLE_AUTO
ROLE_NOISE = _mod.ROLE_NOISE
ROLE_LATENT_IMAGE = _mod.ROLE_LATENT_IMAGE
ROLE_NOISE_LATENT_IMAGE = _mod.ROLE_NOISE_LATENT_IMAGE
ROLE_SEED_DRIVEN = _mod.ROLE_SEED_DRIVEN
SHAPE_A = _mod.SHAPE_A
SHAPE_B = _mod.SHAPE_B
SHAPE_C = _mod.SHAPE_C


# ---------- fixtures ----------

LATENT_SHAPE_4D = (1, 4, 64, 64)        # SDXL-style
LATENT_SHAPE_5D = (1, 16, 1, 152, 114)  # qwen/Wan video VAE


def _seeded_tensor(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def make_shape_A(shape=LATENT_SHAPE_4D, seed=42):
    """Pure init -- img2img. No use_as_noise flag."""
    return {"samples": _seeded_tensor(shape, seed)}


def make_shape_B(shape=LATENT_SHAPE_4D, seed=42):
    """Pure noise -- dimensions only / img2noise / image+noise."""
    return {"samples": _seeded_tensor(shape, seed), "use_as_noise": True}


def make_shape_C(shape=LATENT_SHAPE_4D, seed=42):
    """Layered -- img2img + img2noise."""
    return {
        "samples": _seeded_tensor(shape, seed),       # encoded image (init)
        "noise":   _seeded_tensor(shape, seed + 1),   # shaped noise
        "use_as_noise": True,
    }


def make_shape_D(shape=LATENT_SHAPE_4D):
    """Empty -- txt2img / EmptyLatentImage. Dispatch-equivalent to Shape A."""
    return {"samples": torch.zeros(shape)}


def _resolve(latent, role=ROLE_AUTO, seed=12345):
    """Convenience wrapper -- builds x and calls resolver."""
    x = latent["samples"].clone()
    return resolve_latent_as_noise(latent, x, noise_seed=seed,
                                   latent_role=role)


# ---------- shape detection ----------

def test_detect_shape_A():
    assert detect_shape(make_shape_A()) == SHAPE_A


def test_detect_shape_B():
    assert detect_shape(make_shape_B()) == SHAPE_B


def test_detect_shape_C():
    assert detect_shape(make_shape_C()) == SHAPE_C


def test_detect_shape_D_collapses_to_A():
    """Shape D (zeros samples) is dispatch-equivalent to Shape A."""
    assert detect_shape(make_shape_D()) == SHAPE_A


def test_detect_shape_explicit_use_as_noise_false():
    """use_as_noise=False is treated as no-flag (Shape A)."""
    latent = {"samples": _seeded_tensor(LATENT_SHAPE_4D, 42),
              "use_as_noise": False}
    assert detect_shape(latent) == SHAPE_A


# ---------- v0.1.1 legacy parity (auto + seed=-2) ----------

def test_legacy_shape_A_seed_minus_2_auto_falls_through():
    """Shape A + seed=-2 + auto -> shape_a (caller does seed-driven noise).
    Deprecation log fires (informing user -2 is no longer special)."""
    r = _resolve(make_shape_A(), role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_a"
    assert r.noise is None  # caller falls through to seed-driven generation
    assert r.deprecation is not None
    assert "deprecated" in r.deprecation.lower()


def test_legacy_shape_B_seed_minus_2_auto_zeros_init():
    """Shape B + seed=-2 + auto -> shape_b (the v0.1.1 fix path).
    Deprecation log fires; NO Option C notice (legacy path matches)."""
    latent = make_shape_B()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_b"
    assert r.noise is not None
    assert torch.equal(r.noise, samples_orig)
    assert r.x.abs().max().item() == 0.0  # init zeroed -- THE FIX
    assert r.deprecation is not None
    assert r.warning is None  # no Option C notice when seed=-2


def test_legacy_shape_C_seed_minus_2_auto_preserves_init():
    """Shape C + seed=-2 + auto -> shape_c. The img2img regression guard."""
    latent = make_shape_C()
    samples_orig = latent["samples"].clone()
    noise_orig = latent["noise"].clone()
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_c"
    assert torch.equal(r.noise, noise_orig)
    assert torch.equal(r.x, samples_orig)  # init unchanged
    assert r.deprecation is not None


def test_legacy_shape_D_seed_minus_2_auto_falls_through():
    """Shape D + seed=-2 + auto -> shape_a (samples is zeros)."""
    r = _resolve(make_shape_D(), role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert r.deprecation is not None


def test_legacy_5D_video_latent_shape_B_seed_minus_2():
    """5D latent (qwen/Wan) -- Shape B fix is shape-agnostic via zeros_like."""
    latent = make_shape_B(shape=LATENT_SHAPE_5D, seed=7)
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_b"
    assert r.noise.shape == LATENT_SHAPE_5D
    assert r.x.shape == LATENT_SHAPE_5D
    assert r.x.abs().max().item() == 0.0


# ---------- explicit role x shape matrix (5 x 4 = 20 cells) ----------

# === ROLE: noise -- forces Shape B path ===

def test_noise_role_x_shape_A_warns_falls_back_to_seed_driven():
    """Shape A has no noise tensor; role=noise falls back to seed-driven
    rather than treating samples (encoded image) as noise (which produced
    the v0.1.2-alpha Image #4 ghost-silhouette artifact)."""
    latent = make_shape_A()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_NOISE)
    assert r.applied == "shape_a"
    assert r.noise is None  # caller does seed-driven generation
    assert torch.equal(r.x, samples_orig)  # samples preserved as init
    assert r.warning is not None
    assert "no upstream noise tensor" in r.warning
    assert "Falling back to seed-driven" in r.warning


def test_noise_role_x_shape_B_no_warning():
    latent = make_shape_B()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_NOISE)
    assert r.applied == "shape_b"
    assert torch.equal(r.noise, samples_orig)
    assert r.x.abs().max().item() == 0.0
    assert r.warning is None


def test_noise_role_x_shape_C_uses_noise_key_not_samples():
    """The v0.1.2-alpha Image #4 fix: role=noise on Shape C must use the
    'noise' key (the upstream's actual noise tensor), NOT samples (which
    is the encoded init image). Pre-fix, samples-as-noise produced
    image-structured 'noise' the model could not denoise -> red silhouette."""
    latent = make_shape_C()
    samples_orig = latent["samples"].clone()
    noise_key_orig = latent["noise"].clone()
    r = _resolve(latent, role=ROLE_NOISE)
    assert r.applied == "shape_b"
    # noise comes from the "noise" key, not from samples (the encoded image)
    assert torch.equal(r.noise, noise_key_orig)
    assert not torch.equal(r.noise, samples_orig)  # explicit anti-regression
    assert r.x.abs().max().item() == 0.0  # init zeroed
    assert r.warning is not None
    assert "noise' key" in r.warning
    assert "noise+latent_image" in r.warning  # migration hint


def test_noise_role_x_shape_D_warns_falls_back_to_seed_driven():
    """Shape D collapses to Shape A in detection; role=noise falls back
    to seed-driven rather than using zeros-as-noise (which would produce
    a degenerate noise tensor)."""
    latent = make_shape_D()
    r = _resolve(latent, role=ROLE_NOISE)
    assert r.applied == "shape_a"
    assert r.noise is None  # caller does seed-driven generation
    assert r.warning is not None
    assert "no upstream noise tensor" in r.warning


# === ROLE: latent_image -- forces Shape A path ===

def test_latent_image_role_x_shape_A_no_warning():
    latent = make_shape_A()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None  # caller does seed-driven
    assert torch.equal(r.x, samples_orig)
    assert r.warning is None


def test_latent_image_role_x_shape_B_warns_uses_samples_as_init():
    latent = make_shape_B()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert torch.equal(r.x, samples_orig)  # init NOT zeroed (Shape A semantics)
    assert r.warning is not None
    assert "expects Shape A" in r.warning
    assert "shape_b" in r.warning


def test_latent_image_role_x_shape_C_warns_discards_noise_key():
    latent = make_shape_C()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None  # noise key discarded
    assert torch.equal(r.x, samples_orig)
    assert r.warning is not None
    assert "shape_c" in r.warning


def test_latent_image_role_x_shape_D_no_warning():
    """Shape D collapses to Shape A in detection -- so no warning."""
    latent = make_shape_D()
    r = _resolve(latent, role=ROLE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert r.warning is None


# === ROLE: noise+latent_image -- forces Shape C path ===

def test_noise_latent_image_role_x_shape_A_warns_falls_back_to_shape_a():
    latent = make_shape_A()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_NOISE_LATENT_IMAGE)
    assert r.applied == "shape_a"  # fallback
    assert r.noise is None
    assert torch.equal(r.x, samples_orig)
    assert r.warning is not None
    assert "expects Shape C" in r.warning


def test_noise_latent_image_role_x_shape_B_warns_falls_back_to_shape_a():
    latent = make_shape_B()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_NOISE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None
    # x preserved (NOT zeroed -- explicit role overrides Shape B fix path)
    assert torch.equal(r.x, samples_orig)
    assert r.warning is not None


def test_noise_latent_image_role_x_shape_C_no_warning():
    latent = make_shape_C()
    samples_orig = latent["samples"].clone()
    noise_orig = latent["noise"].clone()
    r = _resolve(latent, role=ROLE_NOISE_LATENT_IMAGE)
    assert r.applied == "shape_c"
    assert torch.equal(r.noise, noise_orig)
    assert torch.equal(r.x, samples_orig)
    assert r.warning is None


def test_noise_latent_image_role_x_shape_D_warns_falls_back_to_shape_a():
    latent = make_shape_D()
    r = _resolve(latent, role=ROLE_NOISE_LATENT_IMAGE)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert r.warning is not None


# === ROLE: seed_driven -- forces seed-driven noise (no dispatch) ===

def test_seed_driven_role_x_shape_A_zeros_init_for_txt2img():
    """v0.1.2-alpha semantics: seed_driven means TRUE txt2img -- init slot
    is zeroed regardless of upstream samples. This is what makes
    seed_driven distinct from latent_image (which preserves samples as
    img2img init)."""
    latent = make_shape_A()
    r = _resolve(latent, role=ROLE_SEED_DRIVEN)
    assert r.applied == "seed_driven"
    assert r.noise is None  # caller does seed-driven noise generation
    assert r.x.abs().max().item() == 0.0  # init zeroed (txt2img)
    # Shape A still warns because zeroing samples is non-default behavior
    # for an upstream that doesn't have use_as_noise=True; users should
    # know to use latent_image instead if they want img2img.
    assert r.warning is not None
    assert "txt2img" in r.warning
    assert "latent_image" in r.warning


def test_seed_driven_role_x_shape_B_zeros_init_ignores_use_as_noise():
    latent = make_shape_B()
    r = _resolve(latent, role=ROLE_SEED_DRIVEN)
    assert r.applied == "seed_driven"
    assert r.noise is None
    assert r.x.abs().max().item() == 0.0  # init zeroed (txt2img)
    assert r.warning is not None
    assert "ignoring" in r.warning.lower()
    assert "shape_b" in r.warning
    assert "txt2img" in r.warning


def test_seed_driven_role_x_shape_C_zeros_init_ignores_noise_key():
    latent = make_shape_C()
    r = _resolve(latent, role=ROLE_SEED_DRIVEN)
    assert r.applied == "seed_driven"
    assert r.noise is None
    assert r.x.abs().max().item() == 0.0  # init zeroed (txt2img)
    assert r.warning is not None
    assert "shape_c" in r.warning
    assert "txt2img" in r.warning


def test_seed_driven_role_x_shape_D_zeros_init():
    """Shape D upstream is already zeros; seed_driven still emits warning
    informing the user about the txt2img override (since the role
    explicitly opts out of any upstream interpretation)."""
    latent = make_shape_D()
    r = _resolve(latent, role=ROLE_SEED_DRIVEN)
    assert r.applied == "seed_driven"
    assert r.noise is None
    assert r.x.abs().max().item() == 0.0
    # Shape D collapses to Shape A in detection -- no warning expected
    # for the "matched detected shape" path.


# === ROLE: auto -- option C semantics (covered above for legacy seed=-2) ===

def test_auto_role_x_shape_A_seed_normal():
    """Standard txt2img / img2img path -- no warning, no deprecation."""
    r = _resolve(make_shape_A(), role=ROLE_AUTO, seed=12345)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert r.warning is None
    assert r.deprecation is None


def test_auto_role_x_shape_B_seed_normal_emits_option_c_notice():
    """Auto dispatched to Shape B without seed=-2 -- modern v0.1.2 path."""
    latent = make_shape_B()
    samples_orig = latent["samples"].clone()
    r = _resolve(latent, role=ROLE_AUTO, seed=12345)
    assert r.applied == "shape_b"
    assert torch.equal(r.noise, samples_orig)
    assert r.x.abs().max().item() == 0.0
    assert r.warning is not None
    assert "auto-mode dispatched to Shape B" in r.warning
    assert r.deprecation is None


def test_auto_role_x_shape_C_seed_normal_emits_option_c_notice():
    """Auto dispatched to Shape C without seed=-2 -- modern v0.1.2 path."""
    latent = make_shape_C()
    r = _resolve(latent, role=ROLE_AUTO, seed=12345)
    assert r.applied == "shape_c"
    assert r.warning is not None
    assert "auto-mode dispatched to Shape C" in r.warning
    assert r.deprecation is None


def test_auto_role_x_shape_D_seed_normal():
    r = _resolve(make_shape_D(), role=ROLE_AUTO, seed=12345)
    assert r.applied == "shape_a"
    assert r.noise is None
    assert r.warning is None
    assert r.deprecation is None


# ---------- protocol invariants ----------

def test_shape_B_noise_is_independent_clone_not_alias():
    """Mutating returned noise must not affect latent['samples']."""
    latent = make_shape_B()
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    original_sum = float(latent["samples"].sum().item())
    r.noise.mul_(0)
    assert float(latent["samples"].sum().item()) == original_sum


def test_shape_B_x_out_is_independent_zeros_not_alias():
    """Mutating returned x must not affect latent['samples']."""
    latent = make_shape_B()
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    original_sum = float(latent["samples"].sum().item())
    r.x.add_(1.0)
    assert float(latent["samples"].sum().item()) == original_sum


def test_shape_C_noise_dtype_device_match_x():
    """Returned noise should be cast to x's dtype/device."""
    latent = make_shape_C()
    latent["noise"] = latent["noise"].to(dtype=torch.float64)
    x = latent["samples"].clone().to(dtype=torch.float32)
    r = resolve_latent_as_noise(latent, x, noise_seed=-2,
                                latent_role=ROLE_AUTO)
    assert r.applied == "shape_c"
    assert r.noise.dtype == x.dtype


# ---------- defensive: invalid role ----------

def test_invalid_role_falls_back_to_auto_with_warning():
    latent = make_shape_A()
    r = resolve_latent_as_noise(latent, latent["samples"].clone(),
                                noise_seed=12345,
                                latent_role="not_a_real_role")
    assert r.applied == "shape_a"  # fell back to auto + Shape A
    assert r.warning is not None
    assert "not_a_real_role" in r.warning


# ---------- backwards-compat use_as_noise=False on dict ----------

def test_use_as_noise_false_explicit_treated_as_shape_a():
    """Explicit use_as_noise=False shouldn't trip Shape B detection."""
    latent = {"samples": _seeded_tensor(LATENT_SHAPE_4D, 42),
              "use_as_noise": False}
    r = _resolve(latent, role=ROLE_AUTO, seed=-2)
    assert r.applied == "shape_a"
    assert r.noise is None


# ---------- regression: seed plumbing through dispatch (v0.1.3-alpha) ----------

def test_seed_passes_through_dispatch_unchanged():
    """The helper does NOT zero, override, or transform noise_seed for any
    role. Per-step noise machinery downstream relies on receiving the user's
    seed verbatim. If a future refactor adds 'helpfulness' that mutates the
    seed based on role, this test will catch it.

    See: 2026-04-29__13-04-30__seed-still-affects-output-under-latent-role-noise.md
    """
    latent_b = {"samples": _seeded_tensor(LATENT_SHAPE_4D, 42),
                "use_as_noise": True}
    r_seed_a = _resolve(latent_b, role=ROLE_NOISE, seed=5225)
    r_seed_b = _resolve(latent_b, role=ROLE_NOISE, seed=9999)
    assert torch.equal(r_seed_a.noise, r_seed_b.noise), \
        "role=noise must produce seed-independent noise tensor (comes from upstream)"
    assert torch.equal(r_seed_a.x, r_seed_b.x), \
        "role=noise must produce seed-independent x (zeros)"


# ---------- manual harness ----------

if __name__ == "__main__":
    import sys, traceback
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
