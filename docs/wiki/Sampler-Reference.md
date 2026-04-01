# Sampler Reference

DazzleKSampler includes 100+ sampler methods organized into categories.

## Categories

### multistep/ — Multi-step methods (most common)

These take multiple model evaluations per step for higher accuracy:

| Sampler | Steps | Notes |
|---------|-------|-------|
| `res_2m` | Multi | RES4LYF default. Good all-around. |
| `res_3m` | Multi | Higher order, slightly better quality |
| `dpmpp_2m` | Multi | DPM++ 2M — widely used, fast |
| `dpmpp_3m` | Multi | DPM++ 3M — better quality than 2M |
| `deis_2m/3m/4m` | Multi | DEIS methods — good for low step counts |

### linear/ — Single-step methods

One model evaluation per step. Fast but lower accuracy:

| Sampler | Notes |
|---------|-------|
| `euler` | Simplest first-order method |
| `heun` | Second-order, better than euler |
| `ralston` | Optimized second-order |
| `rk4` | Classic 4th-order Runge-Kutta |
| `dormand-prince` | Adaptive 5th-order |

### exponential/ — Exponential integrators

Specialized methods for diffusion models:

| Sampler | Notes |
|---------|-------|
| `res_2s` | RES4LYF exponential 2-stage |
| `ddim` | DDIM-style denoising |
| `dpmpp` | DPM++ exponential variant |

### diag_implicit/ — Diagonal implicit methods

Implicit solvers with diagonal approximation:

| Sampler | Notes |
|---------|-------|
| `kraaijevanger_spijker` | L-stable, good for stiff problems |
| `qin_zhang` | Second-order L-stable |
| `crouzeix` | Third-order |

### fully_implicit/ — Fully implicit methods

Maximum stability, most expensive per step:

| Sampler | Notes |
|---------|-------|
| `gauss-legendre` | High-order Gauss quadrature |
| `radau_iia` | Stiff-stable, commonly used in scientific computing |
| `lobatto_iiic` | Endpoint-inclusive quadrature |

## Choosing a sampler

**For most users:** Start with `multistep/dpmpp_2m` or `multistep/res_2m`. These are well-tested and produce good results across model types.

**For speed:** Use `linear/euler` with fewer steps (8-15).

**For quality:** Use `multistep/dpmpp_3m` or `exponential/res_2s` with more steps (25-35).

**For img2img:** `multistep/dpmpp_3m` with `denoise=0.3-0.5` tends to preserve source structure well.

**For Flux/Qwen models:** `multistep/dpmpp_3m` with `sgm_uniform` scheduler is a common choice.
