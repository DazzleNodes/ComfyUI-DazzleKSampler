"""Latent-dict protocol for noise-passthrough samplers.

This module defines the contract between upstream latent producers (e.g.,
SmartResCalc) and Dazzle KSampler's latent-role dispatch. As of v0.1.2-alpha
the dispatch is driven by an explicit `latent_role` widget choice; the
legacy `noise_seed=-2` magic value remains supported under `auto` mode with
a deprecation log.

The latent dict can take one of four documented shapes:

    Shape A -- Pure init (img2img):
        { "samples": vae_encoded_image }
        Use samples as init image; generate noise from seed.

    Shape B -- Pure noise (dimensions only / img2noise / image+noise):
        { "samples": shaped_noise, "use_as_noise": True }
        samples IS the noise. There is no init image; init slot is zeroed
        so the latent acts purely as the sampling noise (the v0.1.1-alpha
        fix: prevents `samples` being dual-roled in
        `initial_x = noise * sigmas[0] + latent_image`).

    Shape C -- Layered (img2img + img2noise):
        { "samples": vae_encoded_image,
          "noise":   shaped_noise,
          "use_as_noise": True }
        samples is init image; "noise" key is the noise tensor.

    Shape D -- Empty (txt2img):
        { "samples": zeros_tensor }
        Stock empty latent. Dispatch-equivalent to Shape A: init from
        samples (which happens to be zero), noise from seed. Distinguished
        only in user-facing docs; the helper does not GPU-sync to detect it.

Dispatch is driven by the user's `latent_role` widget choice:

    "auto"               -- inspect dict shape, dispatch accordingly.
                            Honors `use_as_noise=True` even without
                            seed=-2 and emits an informational notice when
                            it does so (the modern v0.1.2 behavior). The
                            legacy seed=-2 path still works under auto with
                            a deprecation log.
    "noise"              -- force Shape B path (samples used as noise,
                            init zeroed). Warns on shape mismatch.
    "latent_image"       -- force Shape A path (samples used as init,
                            noise from seed). Warns on shape mismatch.
    "noise+latent_image" -- force Shape C path (separate noise key
                            required). Warns if no noise key present.
    "seed_driven"        -- force seed-driven noise generation, ignoring
                            any use_as_noise flag. Used as a debugging
                            override or to opt out of upstream shaping.

Background -- the v0.1.1-alpha bug: when both `samples` and `use_as_noise`
are present without a separate `noise` key (Shape B), `samples` was
previously passed in BOTH the noise role AND the init-image role to the
underlying ComfyUI sampler -- anchoring composition through both roles via
`initial_x = z_norm(L) * sigmas[0] + L`. The fix (now layered into every
Shape B dispatch path here): zero the init slot so the latent acts purely
as noise.
"""

from dataclasses import dataclass
from typing import Optional

import torch


# ---- Public role constants (match the latent_role widget options) -----

ROLE_AUTO = "auto"
ROLE_NOISE = "noise"
ROLE_LATENT_IMAGE = "latent_image"
ROLE_NOISE_LATENT_IMAGE = "noise+latent_image"
ROLE_SEED_DRIVEN = "seed_driven"

VALID_ROLES = (
    ROLE_AUTO,
    ROLE_NOISE,
    ROLE_LATENT_IMAGE,
    ROLE_NOISE_LATENT_IMAGE,
    ROLE_SEED_DRIVEN,
)


# ---- Shape detection tokens (only what we infer from dict keys) -------
# Note: Shape D (zeros samples) is not distinguished from Shape A here --
# they dispatch identically and detecting D would require a GPU sync.

SHAPE_A = "shape_a"  # samples only (no use_as_noise) -- img2img or empty
SHAPE_B = "shape_b"  # samples + use_as_noise, no noise key -- pure noise
SHAPE_C = "shape_c"  # samples + use_as_noise + noise key -- layered


# ---- Dispatch result ---------------------------------------------------

@dataclass
class LatentNoiseDispatch:
    """Result of resolve_latent_as_noise.

    Fields:
        noise: noise tensor, OR None if caller should fall through to
            seed-driven noise generation.
        x: init-image tensor (possibly zeroed for Shape B dispatch).
        detected_shape: SHAPE_A / SHAPE_B / SHAPE_C inferred from dict keys.
        applied: SHAPE_A / SHAPE_B / SHAPE_C / "seed_driven" -- the
            dispatch path actually taken (may differ from detected_shape
            if the user picked an explicit role that overrides detection).
        warning: optional str -- emitted when explicit role mismatches the
            detected shape, OR when auto-mode dispatches to Shape B/C
            without seed=-2 (Option C informational notice).
        deprecation: optional str -- emitted when seed=-2 is used with
            latent_role=auto, signaling the legacy magic value path.
    """
    noise: Optional[torch.Tensor]
    x: torch.Tensor
    detected_shape: str
    applied: str
    warning: Optional[str] = None
    deprecation: Optional[str] = None


# ---- Shape detection ---------------------------------------------------

def detect_shape(latent_unbatch):
    """Classify the latent dict by inspecting its keys (no tensor ops).

    Returns one of SHAPE_A / SHAPE_B / SHAPE_C. Shape D is collapsed
    into SHAPE_A because the dispatch behavior is identical and
    distinguishing them would require a GPU sync.
    """
    use_as_noise = bool(latent_unbatch.get("use_as_noise", False))
    if use_as_noise and "noise" in latent_unbatch:
        return SHAPE_C
    if use_as_noise:
        return SHAPE_B
    return SHAPE_A


# ---- Internal dispatch primitives --------------------------------------

def _shape_a_dispatch(x):
    """Init from samples (x unchanged); caller generates noise from seed."""
    return None, x


def _shape_b_dispatch(x):
    """Init zeroed; noise = samples (the v0.1.1-alpha fix path)."""
    noise = x.clone()
    x_zero = torch.zeros_like(x)
    return noise, x_zero


def _shape_c_dispatch(latent_unbatch, x):
    """Init from samples (unchanged); noise from "noise" key."""
    noise = latent_unbatch["noise"].to(device=x.device, dtype=x.dtype)
    return noise, x


# ---- Main entry --------------------------------------------------------

def resolve_latent_as_noise(latent_unbatch, x, noise_seed,
                            latent_role=ROLE_AUTO):
    """Resolve dispatch from user-selected role + detected dict shape.

    Args:
        latent_unbatch: dict with "samples" + optional "noise" /
            "use_as_noise". The dispatch reads these keys; it does not
            mutate the dict.
        x: init-image tensor (typically `samples.clone()`).
        noise_seed: int -- the user's seed value. Under latent_role=auto,
            seed=-2 still activates the legacy latent-as-noise path
            (with a deprecation log); under explicit roles the seed is
            used only for downstream noise generation when the role
            calls for it.
        latent_role: one of VALID_ROLES; default ROLE_AUTO.

    Returns:
        LatentNoiseDispatch with the computed noise/x/diagnostics.
    """
    # Defensive: unknown role falls back to auto with a warning.
    role_warning = None
    if latent_role not in VALID_ROLES:
        role_warning = (
            f"Unknown latent_role={latent_role!r}; valid options are "
            f"{list(VALID_ROLES)}. Falling back to 'auto'."
        )
        latent_role = ROLE_AUTO

    detected = detect_shape(latent_unbatch)

    # ===== ROLE: seed_driven =====
    # True txt2img override: zero the init slot and let the dispatch site
    # generate noise from the seed. Distinct from latent_image (which keeps
    # samples as init and only forces seed-driven noise) -- seed_driven
    # ignores the upstream entirely so the only thing controlling the
    # output is the seed and the conditioning.
    if latent_role == ROLE_SEED_DRIVEN:
        x_zero = torch.zeros_like(x)
        if detected != SHAPE_A:
            info = (
                f"latent_role=seed_driven: ignoring upstream "
                f"use_as_noise/noise/samples (detected {detected}); "
                f"zeroing init and generating noise from seed (true "
                f"txt2img). To preserve samples as img2img init, use "
                f"latent_role=latent_image instead."
            )
            return LatentNoiseDispatch(
                noise=None, x=x_zero, detected_shape=detected,
                applied="seed_driven", warning=info,
            )
        # Shape A upstream: still zero init for txt2img semantics.
        info = (
            "latent_role=seed_driven: zeroing init slot regardless of "
            "upstream samples (true txt2img). Use latent_role=latent_image "
            "to preserve samples as img2img init."
        )
        return LatentNoiseDispatch(
            noise=None, x=x_zero, detected_shape=detected,
            applied="seed_driven", warning=role_warning or info,
        )

    # ===== ROLE: noise (use upstream's noise tensor; init zeroed) =====
    # Picks the right noise tensor based on dict shape:
    #   SHAPE_B -> samples IS the noise (clone, zero init)
    #   SHAPE_C -> noise key is the noise (zero init; img2img anchor
    #              discarded -- use noise+latent_image to preserve it)
    #   SHAPE_A -> no upstream noise tensor available; fall back to
    #              seed-driven generation, keep samples as init.
    if latent_role == ROLE_NOISE:
        if detected == SHAPE_B:
            n, x_out = _shape_b_dispatch(x)
            return LatentNoiseDispatch(
                noise=n, x=x_out, detected_shape=detected, applied="shape_b",
            )
        if detected == SHAPE_C:
            # Use the upstream's actual noise tensor (the "noise" key),
            # not samples (which is the encoded init image). This is the
            # v0.1.2-alpha fix for the Image #4 silhouette bug: previously
            # we always cloned samples, treating the encoded image as
            # noise -- producing image-structured "noise" the model could
            # not denoise.
            noise_tensor = latent_unbatch["noise"].to(
                device=x.device, dtype=x.dtype,
            )
            x_zero = torch.zeros_like(x)
            msg = (
                "latent_role=noise on Shape C: using 'noise' key as the "
                "noise tensor (the upstream's layered noise); init zeroed. "
                "Samples (the VAE-encoded image) is discarded. To preserve "
                "the img2img anchor, use latent_role=noise+latent_image "
                "instead."
            )
            return LatentNoiseDispatch(
                noise=noise_tensor, x=x_zero, detected_shape=detected,
                applied="shape_b", warning=msg,
            )
        # SHAPE_A: no real noise tensor available. Fall back to
        # seed-driven generation rather than treat the encoded image as
        # noise (which produced the Image #4 ghost-silhouette artifact).
        n, x_out = _shape_a_dispatch(x)
        msg = (
            f"latent_role=noise on Shape A: no upstream noise tensor "
            f"available (no use_as_noise flag, no noise key). Falling "
            f"back to seed-driven noise generation; samples used as "
            f"init. Connect a Shape B/C upstream (e.g., SmartResCalc with "
            f"image_purpose in {{dimensions only, img2noise, image+noise, "
            f"img2img+img2noise}}) for true latent-as-noise behavior."
        )
        return LatentNoiseDispatch(
            noise=n, x=x_out, detected_shape=detected, applied="shape_a",
            warning=msg,
        )

    # ===== ROLE: latent_image (force Shape A) =====
    if latent_role == ROLE_LATENT_IMAGE:
        n, x_out = _shape_a_dispatch(x)
        if detected == SHAPE_A:
            return LatentNoiseDispatch(
                noise=n, x=x_out, detected_shape=detected, applied="shape_a",
            )
        msg = (
            f"latent_role=latent_image expects Shape A (no use_as_noise "
            f"flag); got {detected}. Using samples as init and "
            f"generating noise from seed; any noise tensor in the dict "
            f"is discarded."
        )
        return LatentNoiseDispatch(
            noise=n, x=x_out, detected_shape=detected, applied="shape_a",
            warning=msg,
        )

    # ===== ROLE: noise+latent_image (force Shape C) =====
    if latent_role == ROLE_NOISE_LATENT_IMAGE:
        if detected == SHAPE_C:
            n, x_out = _shape_c_dispatch(latent_unbatch, x)
            return LatentNoiseDispatch(
                noise=n, x=x_out, detected_shape=detected, applied="shape_c",
            )
        # Mismatch -- can't honor "noise+latent_image" without a noise key.
        # Fall back to Shape A so img2img semantics are at least preserved.
        n, x_out = _shape_a_dispatch(x)
        msg = (
            f"latent_role=noise+latent_image expects Shape C (separate "
            f"noise key + use_as_noise=True); got {detected}. Falling "
            f"back to Shape A: samples as init, noise generated from "
            f"seed. Set upstream node to produce a layered latent for "
            f"the requested behavior."
        )
        return LatentNoiseDispatch(
            noise=n, x=x_out, detected_shape=detected, applied="shape_a",
            warning=msg,
        )

    # ===== ROLE: auto (Option C semantics) =====
    # Default. Dispatch on detected shape. Emit informational notice if
    # we apply Shape B/C without the legacy seed=-2 signal, so users see
    # the modern behavior is active. Emit deprecation log when seed=-2.
    deprecation = None
    if noise_seed == -2:
        deprecation = (
            "noise_seed=-2 is deprecated; latent_role=auto now infers "
            "dispatch from the latent dict shape regardless of seed. "
            "The seed=-2 magic will lose its special meaning in a "
            "future release -- set latent_role explicitly to control "
            "dispatch."
        )

    if detected == SHAPE_C:
        n, x_out = _shape_c_dispatch(latent_unbatch, x)
        warn = role_warning
        if noise_seed != -2 and warn is None:
            warn = (
                "auto-mode dispatched to Shape C (img2img + img2noise) "
                "based on dict shape (use_as_noise=True + noise key). "
                "Pre-v0.1.2-alpha this required noise_seed=-2; modern "
                "auto mode infers from the dict alone."
            )
        return LatentNoiseDispatch(
            noise=n, x=x_out, detected_shape=detected, applied="shape_c",
            warning=warn, deprecation=deprecation,
        )

    if detected == SHAPE_B:
        n, x_out = _shape_b_dispatch(x)
        warn = role_warning
        if noise_seed != -2 and warn is None:
            warn = (
                "auto-mode dispatched to Shape B (latent as noise; init "
                "zeroed) based on use_as_noise=True. Pre-v0.1.2-alpha "
                "this required noise_seed=-2; modern auto mode infers "
                "from the dict alone."
            )
        return LatentNoiseDispatch(
            noise=n, x=x_out, detected_shape=detected, applied="shape_b",
            warning=warn, deprecation=deprecation,
        )

    # detected == SHAPE_A -- standard img2img / txt2img path.
    n, x_out = _shape_a_dispatch(x)
    return LatentNoiseDispatch(
        noise=n, x=x_out, detected_shape=detected, applied="shape_a",
        warning=role_warning, deprecation=deprecation,
    )
