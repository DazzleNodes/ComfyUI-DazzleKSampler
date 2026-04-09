"""
Tau-Augmented Sampler for ComfyUI.

Implements the tau complement operation: at each denoising step,
recovers information that standard denoising discarded and resolves
the contradiction between kept and discarded structure.

Based on the Tau Operator from D. Darcy's Scarcity Framework.
Standard attention/denoising computes the "addition half" (what to keep).
The tau complement computes the "subtraction half" (what was discarded).
Together they complete the analogy.

Reference: DazzleML/contradiction-extraction-poc#3
"""

import torch
import math
from torch import Tensor
from typing import Optional

from .samplers import (
    ClownSamplerAdvanced_Beta,
    SharkSampler,
    OptionsManager,
)
from .constants import MAX_STEPS

# Import scheduler list from helper (already loaded by samplers)
from ..helper import get_res4lyf_scheduler_list


# Tau-compatible base samplers (prefix stripped before coefficient lookup)
TAU_BASE_SAMPLERS = [
    "res_2m",
    "res_2s",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2s",
    "dpmpp_3m",
]

TAU_SAMPLER_NAMES = [f"tau/{name}" for name in TAU_BASE_SAMPLERS]


def get_tau_sampler_list():
    """Return the list of tau sampler names for the UI dropdown."""
    return TAU_SAMPLER_NAMES


def parse_tau_sampler_name(sampler_name: str) -> tuple[str, bool]:
    """Parse a tau sampler name into base sampler name and tau flag.

    Returns:
        (base_name, is_tau): e.g., ("res_2m", True) from "tau/res_2m"
    """
    if sampler_name.startswith("tau/"):
        return sampler_name[4:], True
    return sampler_name, False


def apply_tau_complement(
    x_next: Tensor,
    x_0: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
    tau_strength: float,
    tau_mode: str = "hard",
    step: int = 0,
    total_steps: int = 1,
) -> Tensor:
    """Apply tau complement: recover discarded information from the denoising step.

    The complement is what the denoising step "removed" from the latent.
    Standard denoising keeps structure (addition). The complement recovers
    what was discarded (subtraction). The resolution blends them.

    Args:
        x_next: The denoised latent for this step
        x_0: The reference/original latent before this step
        sigma: Current noise level
        sigma_next: Next noise level
        tau_strength: Blend strength (0 = standard, >0 = recover complement)
        tau_mode: Schedule mode ("hard", "soft", "cosine")
        step: Current step number
        total_steps: Total number of steps

    Returns:
        Modified x_next with tau complement applied
    """
    if tau_strength == 0.0:
        return x_next

    # The complement: what denoising removed
    complement = x_0 - x_next

    if tau_mode == "hard":
        # Fixed strength at every step
        x_next = x_next + tau_strength * complement

    elif tau_mode == "soft":
        # Sigma-aware: more complement at low noise (detail phase)
        # progress goes from 0 (start) to 1 (end of denoising)
        if sigma.dim() == 0:
            progress = 1.0 - (sigma_next / sigma).clamp(0, 1).item()
        else:
            progress = 1.0 - (sigma_next / sigma).clamp(0, 1)
        x_next = x_next + tau_strength * progress * complement

    elif tau_mode == "cosine":
        # Cosine schedule: smooth ramp from 0 to tau_strength
        progress = step / max(total_steps - 1, 1)
        weight = (1 - math.cos(progress * math.pi)) / 2
        x_next = x_next + tau_strength * weight * complement

    return x_next


class TauSampler_Beta:
    """Dazzle TauSampler -- tau-augmented sampling for contradiction resolution.

    Same inputs/outputs as DazzleKSampler but with a clean, focused UI
    for tau-specific parameters. Uses the tau complement operation at each
    denoising step to recover and resolve information that standard
    denoising discards.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta":          ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01, "round": False,
                                           "tooltip": "Noise amount added then removed at each step (for SDE/ancestral variants)."}),
                "sampler_name": (get_tau_sampler_list(), {"default": "tau/res_2m"}),
                "scheduler":    (get_res4lyf_scheduler_list(), {"default": "beta57"}),
                "steps":        ("INT",   {"default": 30,  "min": 1,      "max": MAX_STEPS}),
                "steps_to_run": ("INT",   {"default": -1,  "min": -1,     "max": MAX_STEPS}),
                "denoise":      ("FLOAT", {"default": 1.0, "min": -10000, "max": MAX_STEPS, "step": 0.01}),
                "cfg":          ("FLOAT", {"default": 5.5, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
                "seed":         ("INT",   {"default": 0,   "min": -2,     "max": 0xffffffffffffffff,
                                           "tooltip": "Noise seed. -2 = use input latent as noise pattern."}),
                "sampler_mode": (['unsample', 'standard', 'resample'], {"default": "standard"}),
                "bongmath":     ("BOOLEAN", {"default": True}),
                "tau_strength": ("FLOAT", {"default": 0.2, "min": -2.0,   "max": 2.0,   "step": 0.01, "round": False,
                                           "tooltip": "Tau complement strength. How much discarded information to recover. 0 = standard sampling."}),
                "tau_mode":     (["hard", "soft", "cosine"], {"default": "soft",
                                           "tooltip": "hard = fixed strength every step. soft = more complement at low noise (detail phase). cosine = smooth ramp."}),
            },
            "optional": {
                "model":        ("MODEL",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sigmas":       ("SIGMAS",),
                "guides":       ("GUIDES",),
                "options":      ("OPTIONS", {}),
            }
        }

    RETURN_TYPES = ("LATENT",
                    "LATENT",
                    "OPTIONS",
                    )

    RETURN_NAMES = ("output",
                    "denoised",
                    "options",
                    )

    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"

    def main(self,
             model                                                  = None,
             denoise                       : float                  = 1.0,
             scheduler                     : str                    = "beta57",
             cfg                           : float                  = 7.0,
             seed                          : int                    = -1,
             positive                                               = None,
             negative                                               = None,
             latent_image                  : Optional[dict]         = None,
             steps                         : int                    = 30,
             steps_to_run                  : int                    = -1,
             eta                           : float                  = 0.5,
             sampler_name                  : str                    = "tau/res_2m",
             sampler_mode                  : str                    = "standard",
             bongmath                      : bool                   = True,
             tau_strength                  : float                  = 0.2,
             tau_mode                      : str                    = "soft",
             sigmas                        : Optional[Tensor]       = None,
             guides                                                 = None,
             options                                                = None,
             **kwargs
             ):

        options_mgr = OptionsManager(options, **kwargs)

        # Parse tau sampler name -> base sampler name
        base_sampler_name, is_tau = parse_tau_sampler_name(sampler_name)

        # Override from options if provided
        tau_strength = options_mgr.get('tau_strength', tau_strength)
        tau_mode     = options_mgr.get('tau_mode', tau_mode)
        eta          = options_mgr.get('eta', eta)

        # Pass tau params via extra_options string (read by EO() in sample_rk_beta)
        tau_extra_options = f"tau_strength={tau_strength}\ntau_mode={tau_mode}"

        # Handle chained inputs (same as ClownsharKSampler_Beta)
        if latent_image is not None and 'positive' in latent_image and positive is None:
            positive = latent_image['positive']
        if latent_image is not None and 'negative' in latent_image and negative is None:
            negative = latent_image['negative']
        if latent_image is not None and 'model' in latent_image and model is None:
            model = latent_image['model']

        # Handle cascade models
        noise_type_sde = "gaussian"
        noise_type_sde_substep = "gaussian"
        if model is not None and model.model.model_config.unet_config.get('stable_cascade_stage') == 'b':
            noise_type_sde = "pyramid-cascade_B"
            noise_type_sde_substep = "pyramid-cascade_B"

        # Determine SDE mode based on sampler name
        noise_mode_sde = "hard"
        if "sde" in base_sampler_name:
            noise_mode_sde = "hard"

        # Build sampler via ClownSamplerAdvanced_Beta (reuse existing infrastructure)
        sampler, = ClownSamplerAdvanced_Beta().main(
            noise_type_sde                = noise_type_sde,
            noise_type_sde_substep        = noise_type_sde_substep,
            noise_mode_sde                = noise_mode_sde,
            noise_mode_sde_substep        = noise_mode_sde,
            eta                           = eta,
            eta_substep                   = eta,
            overshoot                     = 0.0,
            overshoot_substep             = 0.0,
            overshoot_mode                = "hard",
            overshoot_mode_substep        = "hard",
            momentum                      = 1.0,
            alpha_sde                     = -1.0,
            k_sde                         = 1.0,
            cfgpp                         = 0.0,
            c1                            = 0.0,
            c2                            = 0.5,
            c3                            = 1.0,
            sampler_name                  = base_sampler_name,
            implicit_sampler_name         = "use_explicit",
            implicit_type                 = "bongmath",
            implicit_type_substeps        = "bongmath",
            implicit_steps                = 0,
            implicit_substeps             = 0,
            rescale_floor                 = True,
            noise_seed_sde                = -1,
            guides                        = guides,
            options                       = options_mgr.as_dict(),
            extra_options                 = tau_extra_options,
            s_noise                       = 1.0,
            s_noise_substep               = 1.0,
            d_noise                       = 1.0,
            d_noise_start_step            = 0,
            d_noise_inv                   = 1.0,
            d_noise_inv_start_step        = 0,
            bongmath                      = bongmath,
        )

        # Run sampling via SharkSampler (reuse existing infrastructure)
        output, denoised, sde_noise = SharkSampler().main(
            model           = model,
            cfg             = cfg,
            scheduler       = scheduler,
            steps           = steps,
            steps_to_run    = steps_to_run,
            denoise         = denoise,
            latent_image    = latent_image,
            positive        = positive,
            negative        = negative,
            sampler         = sampler,
            cfgpp           = 0.0,
            noise_seed      = seed,
            options         = options_mgr.as_dict(),
            noise_type_init = "gaussian",
            noise_stdev     = 1.0,
            sampler_mode    = sampler_mode,
            denoise_alt     = 1.0,
            sigmas          = sigmas,
            extra_options   = "",
        )

        return (output, denoised, options_mgr.as_dict(),)
