"""
ComfyUI DazzleKSampler - DazzleNodes Custom Node
Enhanced KSampler with SmartResCalc noise passthrough and DazzleCommand integration.

Based on RES4LYF's ClownsharKSampler (AGPL-3.0).
Part of the DazzleNodes collection.
"""

import logging
import os
import sys

# Configure module logger
_logger = logging.getLogger("DazzleKSampler")

if os.environ.get('DKS_DEBUG', '').lower() in ('1', 'true', 'yes'):
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
        _logger.addHandler(_handler)

# Dual-loading detection
_DKS_SENTINEL = '_dazzle_ksampler_loaded'
_is_duplicate_load = hasattr(sys, _DKS_SENTINEL)

if _is_duplicate_load:
    _first_path = getattr(sys, _DKS_SENTINEL)
    _this_path = os.path.dirname(os.path.abspath(__file__))
    print(f"[DazzleKSampler] WARNING: Duplicate installation detected!")
    print(f"[DazzleKSampler]   Already loaded from: {_first_path}")
    print(f"[DazzleKSampler]   Skipping this copy:  {_this_path}")
else:
    setattr(sys, _DKS_SENTINEL, os.path.dirname(os.path.abspath(__file__)))

# Initialize sampling engine (registers schedulers, patches calculate_sigmas)
try:
    from .py.res4lyf import init as _init_engine
    _init_engine()
except Exception as e:
    print(f"[DazzleKSampler] WARNING: Engine init failed: {e}")
    # Fallback: register beta57 scheduler manually
    try:
        import comfy.samplers
        if "beta57" not in comfy.samplers.SCHEDULER_NAMES:
            comfy.samplers.SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]
        print(f"[DazzleKSampler] Registered beta57 scheduler (fallback)")
    except Exception:
        pass

# Import node classes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .py.beta.samplers import (
        ClownsharKSampler_Beta,
        SharkSampler_Beta,
        ClownSamplerAdvanced_Beta,
        ClownsharkChainsampler_Beta,
        ClownSampler_Beta,
        BongSampler,
    )
    from .py.beta.tau_sampler import TauSampler_Beta

    # Register sample_dazzle_rk_beta into ComfyUI's k_diffusion sampling namespace
    # Uses "dazzle_rk_beta" key to avoid collision with RES4LYF's "rk_beta"
    try:
        import sys
        import comfy.k_diffusion.sampling as _k_sampling
        # Find our rk_sampler_beta module from sys.modules (already loaded by samplers.py)
        _rk_mod = None
        for _modname, _mod in sys.modules.items():
            if 'dazzle-ksampler' in _modname and 'rk_sampler_beta' in _modname and hasattr(_mod, 'sample_rk_beta'):
                _rk_mod = _mod
                break
            elif 'dazzle_ksampler' in _modname and 'rk_sampler_beta' in _modname and hasattr(_mod, 'sample_rk_beta'):
                _rk_mod = _mod
                break
        if _rk_mod is not None:
            setattr(_k_sampling, 'sample_dazzle_rk_beta', _rk_mod.sample_rk_beta)
            print(f"[DazzleKSampler] Registered sample_dazzle_rk_beta")
        else:
            # Direct fallback
            from .py.beta.rk_sampler_beta import sample_rk_beta
            setattr(_k_sampling, 'sample_dazzle_rk_beta', sample_rk_beta)
            print(f"[DazzleKSampler] Registered sample_dazzle_rk_beta (direct import)")
    except Exception as e2:
        print(f"[DazzleKSampler] WARNING: Failed to register sample_dazzle_rk_beta: {e2}")
        import traceback
        traceback.print_exc()

    NODE_CLASS_MAPPINGS.update({
        "DazzleKSampler": ClownsharKSampler_Beta,
        "DazzleKSampler_Advanced": ClownSamplerAdvanced_Beta,
        "DazzleKSampler_Chain": ClownsharkChainsampler_Beta,
        "DazzleSharkSampler": SharkSampler_Beta,
        "DazzleClownSampler": ClownSampler_Beta,
        "DazzleBongSampler": BongSampler,
        "DazzleTauSampler": TauSampler_Beta,
    })
except Exception as e:
    print(f"[DazzleKSampler] ERROR loading sampler nodes: {e}")
    import traceback
    traceback.print_exc()

try:
    from .py.beta.samplers_extensions import (
        ClownSamplerSelector_Beta,
        ClownOptions_SDE_Beta,
    )
    # Extension nodes registered with Dazzle prefix
except Exception as e:
    print(f"[DazzleKSampler] WARNING: Extensions not loaded: {e}")

# Build display name mappings from whatever loaded
NODE_DISPLAY_NAME_MAPPINGS = {
    "DazzleKSampler": "Dazzle KSampler (DazzleNodes)",
    "DazzleKSampler_Advanced": "Dazzle KSampler Advanced (DazzleNodes)",
    "DazzleKSampler_Chain": "Dazzle KSampler Chain (DazzleNodes)",
    "DazzleSharkSampler": "Dazzle Shark Sampler (DazzleNodes)",
    "DazzleClownSampler": "Dazzle Clown Sampler (DazzleNodes)",
    "DazzleBongSampler": "Dazzle Bong Sampler (DazzleNodes)",
    "DazzleTauSampler": "Dazzle TauSampler (DazzleNodes)",
}

_load_ok = len(NODE_CLASS_MAPPINGS) > 0

WEB_DIRECTORY = None if _is_duplicate_load else "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

if _is_duplicate_load:
    print(f"[DazzleKSampler] Duplicate skipped (JS disabled)")
elif _load_ok:
    print(f"[DazzleKSampler] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
else:
    print(f"[DazzleKSampler] Failed to load")
