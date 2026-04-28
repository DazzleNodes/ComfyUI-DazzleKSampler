"""Latent-dict protocol for noise-passthrough samplers.

This module defines the contract between upstream latent producers (e.g.,
SmartResCalc) and Dazzle KSampler's `seed=-2` mode.

The latent dict can take one of four shapes:

    Shape A — Pure init (img2img):
        { "samples": vae_encoded_image }
        Use samples as init image; generate noise from seed.

    Shape B — Pure noise (dimensions-only / img2noise / image+noise):
        { "samples": shaped_noise, "use_as_noise": True }
        samples IS the noise. There is no init image — caller should treat
        init as zero so the latent acts purely as the sampling noise.

    Shape C — Layered (img2img + img2noise):
        { "samples": vae_encoded_image,
          "noise":   shaped_noise,
          "use_as_noise": True }
        samples is init image; noise key is the noise tensor.

    Shape D — Empty (txt2img):
        { "samples": zeros_tensor }
        Stock empty latent. Sampler generates noise from seed; init is the
        (zero) samples tensor.

The `seed=-2` magic value on Dazzle KSampler activates "use latent as noise"
mode, but only takes effect when the upstream latent has `use_as_noise=True`
(Shapes B and C). Shapes A and D fall through to seed-driven noise generation.

Background: when both `samples` and `use_as_noise` are present without a
separate `noise` key (Shape B), the bug we fixed in v0.1.1-alpha was that
`samples` was being passed in BOTH the noise role AND the init-image role
to the underlying ComfyUI sampler — anchoring composition through both
roles via `initial_x = z_norm(L)·sigmas[0] + L`. The fix: zero the init
slot in Shape B so the latent acts purely as noise.

See: design doc at private/claude/2026-04-28__13-18-58__dev-workflow-implement-seed-minus-2-dual-role-fix.md
"""

import torch


def resolve_latent_as_noise(latent_unbatch, x, noise_seed):
    """Resolve the latent-as-noise dispatch for `seed=-2` mode.

    Honors the latent-dict protocol described in this module's docstring.

    Args:
        latent_unbatch: dict with required key "samples" and optional keys
            "use_as_noise" (bool) and "noise" (torch.Tensor).
        x: torch.Tensor — the current init-image tensor, typically equal to
            `latent_unbatch["samples"].clone()`.
        noise_seed: int — the user's seed value. The dispatch only activates
            when this equals `-2` AND `use_as_noise` is True.

    Returns:
        Tuple[Optional[torch.Tensor], torch.Tensor, str]:
            noise: the noise tensor, or None if this dispatch does not apply
                (caller should fall through to seed-driven noise generation).
            x_init: the init-image tensor to pass forward. Equals zeros for
                Shape B, equals `x` (unchanged) for Shapes A/C/D.
            applied: one of "shape_b", "shape_c", or "not_applied" — included
                for diagnostics and logging.
    """
    use_latent_as_noise = latent_unbatch.get("use_as_noise", False)
    if not (use_latent_as_noise and noise_seed == -2):
        return None, x, "not_applied"

    if "noise" in latent_unbatch:
        noise = latent_unbatch["noise"].to(device=x.device, dtype=x.dtype)
        return noise, x, "shape_c"

    noise = x.clone()
    x_init = torch.zeros_like(x)
    return noise, x_init, "shape_b"
