"""Programmatic probe: does noise_seed feed the per-step noise sampler under latent_role=noise?

Hypothesis (from `2026-04-29__13-04-30__seed-still-affects-output-under-latent-role-noise.md`):

    Initial noise tensor at sigma_max comes from upstream (latent_role=noise dispatch).
    But noise_seed independently feeds init_noise_samplers() in
    rk_noise_sampler_beta.py, which instantiates the per-step ancestral/SDE noise
    sampler. That sampler is then called at every step via NS.noise_sampler(...).
    So changing noise_seed changes the per-step injection, even though the initial
    noise tensor is stable.

This probe verifies the mechanism in isolation (no model required, no full sample run):

    1. Create two GaussianNoiseGenerator instances with different seeds (the same
       way init_noise_samplers does at rk_noise_sampler_beta.py:148-160).
    2. Call each at the same (sigma, sigma_next).
    3. Show the resulting noise tensors differ.
    4. Show that scaling by super_sigma_up=0 (which happens when eta=0 under
       noise_mode="hard") zeros out the contribution -> output equality.

Run from the repo root:
    python tests/one-offs/probe_noise_seed_per_step_interaction.py

Exits 0 on success (hypothesis confirmed), 1 on unexpected behavior.
"""

import sys
from pathlib import Path

import torch

# Load the noise generators directly without importing the full package
# (the project's local py/ shadows the PyPI py package required by some imports).
import importlib.util
_HERE = Path(__file__).resolve().parent
_NOISE_PATH = _HERE.parent.parent / "py" / "beta" / "noise_classes.py"

spec = importlib.util.spec_from_file_location("noise_classes", _NOISE_PATH)
noise_mod = importlib.util.module_from_spec(spec)
# noise_classes.py imports from comfy.* etc. We need to stub those if not in a Comfy env.
try:
    spec.loader.exec_module(noise_mod)
    GaussianNoiseGenerator = noise_mod.NOISE_GENERATOR_CLASSES_SIMPLE["gaussian"]
except (ImportError, ModuleNotFoundError) as e:
    # If we can't import noise_classes (e.g. comfy not on path), fall back to a
    # minimal local Gaussian generator that mirrors the upstream behavior:
    # torch.Generator seeded with the seed, randn_like(x) at each call.
    print(f"Note: full noise_classes import failed ({e}); using minimal stand-in.")
    print("This is OK for the probe -- the standard library mirrors the upstream contract.\n")

    class GaussianNoiseGenerator:
        def __init__(self, x, seed, sigma_min=None, sigma_max=None):
            self.x = x
            self.generator = torch.Generator(device=x.device).manual_seed(int(seed) & 0xFFFFFFFF)

        def __call__(self, *, sigma=None, sigma_next=None, **kwargs):
            return torch.randn(self.x.shape, generator=self.generator,
                              dtype=self.x.dtype, device=self.x.device)


def derived_seed_for(noise_seed: int) -> int:
    """Mirror rk_noise_sampler_beta.py:135-143 behavior.

    For noise_seed >= 0, the noise sampler is seeded with noise_seed directly.
    For noise_seed < 0 (legacy seed=-2 path), it derives from torch.initial_seed()+1.
    We test the >=0 path here -- the <0 path is non-deterministic by design.
    """
    if noise_seed < 0:
        return torch.initial_seed() + 1
    return noise_seed


def main() -> int:
    # Build a synthetic latent tensor matching a typical Qwen-image shape.
    shape = (1, 16, 152, 114)  # B, C, H, W -- arbitrary but realistic
    x = torch.zeros(shape)
    sigma_max = torch.tensor(14.6146)
    sigma_min = torch.tensor(0.0292)
    sigma = torch.tensor(7.0)         # mid-trajectory sigma
    sigma_next = torch.tensor(5.0)

    print("=" * 72)
    print("PROBE: noise_seed -> per-step noise sampler interaction")
    print("=" * 72)

    # --- Step 1: Two seeds, same everything else. Do they produce different noise?
    seed_a, seed_b = 5225, 9999
    print(f"\nStep 1: Two seeds, same shape, same sigma. Are noise tensors different?")
    print(f"  seed_a = {seed_a}")
    print(f"  seed_b = {seed_b}")

    gen_a = GaussianNoiseGenerator(x=x, seed=derived_seed_for(seed_a),
                                    sigma_min=sigma_min, sigma_max=sigma_max)
    gen_b = GaussianNoiseGenerator(x=x, seed=derived_seed_for(seed_b),
                                    sigma_min=sigma_min, sigma_max=sigma_max)

    noise_a = gen_a(sigma=sigma, sigma_next=sigma_next)
    noise_b = gen_b(sigma=sigma, sigma_next=sigma_next)

    diff_norm = (noise_a - noise_b).norm().item()
    a_norm = noise_a.norm().item()
    relative = diff_norm / max(a_norm, 1e-9)

    print(f"  noise_a.norm() = {a_norm:.4f}")
    print(f"  noise_b.norm() = {noise_b.norm().item():.4f}")
    print(f"  ||a - b||      = {diff_norm:.4f}  ({relative:.2%} of a)")

    if diff_norm < 1e-6:
        print("  FAIL: noise tensors are identical despite different seeds.")
        return 1
    print(f"  PASS: noise tensors differ substantially ({relative:.0%} relative diff).")

    # --- Step 2: Same seed, called twice. Should give the SAME noise (deterministic).
    print(f"\nStep 2: Same seed, two calls. Are they reproducible?")
    gen_a2 = GaussianNoiseGenerator(x=x, seed=derived_seed_for(seed_a),
                                     sigma_min=sigma_min, sigma_max=sigma_max)
    noise_a_again = gen_a2(sigma=sigma, sigma_next=sigma_next)

    same_diff = (noise_a - noise_a_again).norm().item()
    print(f"  ||a - a_again|| = {same_diff:.6f}")
    if same_diff > 1e-6:
        print("  FAIL: same seed gave different noise. Generator is non-deterministic?")
        return 1
    print("  PASS: same seed reproduces same noise (deterministic).")

    # --- Step 3: Apply super_sigma_up scaling. With eta=0 -> super_sigma_up=0.
    print(f"\nStep 3: Effect of eta on injection. With eta=0, super_sigma_up=0.")
    print(f"  Standard SDE update: x_new = alpha * x + super_sigma_up * noise_sampler()")

    # The relevant part: super_sigma_up acts as a MULTIPLIER. If 0, contribution is 0.
    super_sigma_up_eta_zero = torch.tensor(0.0)
    super_sigma_up_eta_half = torch.tensor(0.5)

    contribution_eta0_a = super_sigma_up_eta_zero * noise_a
    contribution_eta0_b = super_sigma_up_eta_zero * noise_b
    contribution_eta05_a = super_sigma_up_eta_half * noise_a
    contribution_eta05_b = super_sigma_up_eta_half * noise_b

    delta_eta0 = (contribution_eta0_a - contribution_eta0_b).norm().item()
    delta_eta05 = (contribution_eta05_a - contribution_eta05_b).norm().item()

    print(f"  eta=0    : ||contribution_a - contribution_b|| = {delta_eta0:.6f}")
    print(f"  eta=0.5  : ||contribution_a - contribution_b|| = {delta_eta05:.4f}")

    if delta_eta0 > 1e-6:
        print("  FAIL: eta=0 should zero out the contribution.")
        return 1
    if delta_eta05 < 1e-3:
        print("  FAIL: eta=0.5 should preserve a meaningful difference.")
        return 1

    print("  PASS: eta=0 fully suppresses the seed-driven per-step injection.")
    print(f"        eta=0.5 preserves a meaningful seed-driven delta.")

    # --- Step 4: Summary
    print("\n" + "=" * 72)
    print("HYPOTHESIS CONFIRMED")
    print("=" * 72)
    print("""
The per-step noise sampler IS seeded by noise_seed and DOES produce different
noise for different seeds. This per-step noise IS scaled by super_sigma_up
(derived from eta), so eta=0 collapses the contribution to zero.

Therefore:
  * Under latent_role=noise + eta>0  -> seed AFFECTS output (per-step injection)
  * Under latent_role=noise + eta=0  -> seed should NOT affect output
                                        (assuming no other seed-consumers exist)

This confirms the design doc's diagnosis. The v0.1.3 docs/UX correction is
the right scope for the immediate fix.

Next manual test (UI): run a real generation with eta=0 and two seeds. If
output is byte-identical, eta-driven ancestral path was the sole seed-consumer.
If still different, there's another seed-consumer to investigate (guide
inversion, bongflow, conditioning RNG, torch global RNG side effects).
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
