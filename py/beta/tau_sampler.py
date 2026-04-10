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
    ClownsharKSampler_Beta,
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
                "tau_version":  (["tau1", "tau2", "tau3", "tau4"], {"default": "tau2",
                                           "tooltip": "tau1 = raw complement (noise vs coherence). tau2 = B+C structural filtering (alignment + temporal). tau3 = seed-variant perturbation via ODE chaos sensitivity (same seed, slight structural variations). tau4 = spectral per-bin complement: injects denoised's phase at frequency bands the denoiser isn't acting on this step (genuinely orthogonal via DFT basis)."}),
                "tau_strength": ("FLOAT", {"default": 0.5, "min": 0.0,    "max": 1.0,   "step": 0.01, "round": False,
                                           "tooltip": "Tau complement strength (0-1 range, internally mapped to effective 0-0.12). 0 = standard sampling."}),
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
             tau_version                   : str                    = "tau2",
             tau_strength                  : float                  = 0.5,
             tau_mode                      : str                    = "soft",
             sigmas                        : Optional[Tensor]       = None,
             guides                                                 = None,
             options                                                = None,
             **kwargs
             ):

        # Parse tau sampler name -> base sampler name
        base_sampler_name, is_tau = parse_tau_sampler_name(sampler_name)

        # Build extra_options with tau params
        tau_extra = f"tau_strength={tau_strength}\ntau_mode={tau_mode}\ntau_version={tau_version}"

        # Effective strength: tau1/tau2 are range-compressed to 0-0.12;
        # tau3 uses the full 0-1 range (its perturbation is FP-scale anyway).
        eff_str = tau_strength if tau_version == "tau3" else tau_strength * 0.12
        print(f"[TauSampler] base={base_sampler_name} v={tau_version} "
              f"str={tau_strength} (eff={eff_str:.4f}) "
              f"mode={tau_mode} bongmath={bongmath} eta={eta}")

        # Delegate entirely to ClownsharKSampler_Beta -- it handles ALL param
        # defaults, options merging, chained inputs, cascade detection, etc.
        # We just override the sampler_name and inject tau extra_options.
        output, denoised, options_out = ClownsharKSampler_Beta().main(
            model           = model,
            denoise         = denoise,
            scheduler       = scheduler,
            cfg             = cfg,
            seed            = seed,
            positive        = positive,
            negative        = negative,
            latent_image    = latent_image,
            steps           = steps,
            steps_to_run    = steps_to_run,
            bongmath        = bongmath,
            sampler_mode    = sampler_mode,
            eta             = eta,
            sampler_name    = base_sampler_name,
            sigmas          = sigmas,
            guides          = guides,
            options         = options,
            extra_options   = tau_extra,
        )

        return (output, denoised, options_out,)
