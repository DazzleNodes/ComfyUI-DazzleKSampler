# Scheduler Reference

Schedulers control the **sigma schedule** — the sequence of noise levels that the sampler steps through during denoising. Different schedules allocate denoising "effort" differently across the generation process.

## Available Schedulers

### Standard ComfyUI Schedulers

| Scheduler | Description |
|-----------|-------------|
| `normal` | Linear spacing in noise level. ComfyUI default. |
| `karras` | Cosine-based spacing. More steps at high noise, fewer at low noise. Tends to produce smoother results. |
| `exponential` | Exponential decay spacing. |
| `sgm_uniform` | Uniform spacing in sigma space. Common for Flux and Qwen models. |
| `simple` | Evenly spaced. Predictable and straightforward. |
| `ddim_uniform` | DDIM-style uniform spacing. |
| `linear_quadratic` | Linear-quadratic blend schedule. |

### RES4LYF Custom Schedulers

| Scheduler | Description |
|-----------|-------------|
| `beta57` | RES4LYF's custom schedule using beta distribution (alpha=0.5, beta=0.7). Good general-purpose schedule that allocates more steps to mid-range noise levels where most detail emerges. **DazzleKSampler default.** |

## How Schedulers Affect Output

### High noise region (early steps)
- Establishes overall composition, layout, large-scale structure
- Schedulers that spend more time here: more varied compositions
- Schedulers that rush through: more predictable layouts

### Mid noise region (middle steps)
- Develops details, textures, facial features
- This is where most "quality" comes from
- `beta57` and `karras` allocate extra steps here

### Low noise region (final steps)
- Sharpening, fine details, color accuracy
- Diminishing returns — extra steps here have less impact
- `normal` schedule spends equal time here vs other regions

## Choosing a Scheduler

| Use case | Recommended | Why |
|----------|-------------|-----|
| General SDXL | `beta57` | Good balance across noise levels |
| Flux / Qwen | `sgm_uniform` | These models expect uniform sigma spacing |
| Maximum detail | `karras` | Extra attention to mid-range where details emerge |
| Fast preview | `simple` or `normal` | Predictable, no surprises |
| Scientific/precise | `exponential` | Mathematically clean decay |

## Interaction with Steps

More steps = finer sigma spacing within the chosen schedule. The schedule determines WHERE the steps are allocated, not how many there are.

- 8 steps with `karras` = 8 samples along a cosine curve
- 30 steps with `karras` = 30 samples along the same curve (finer spacing)
- 8 steps with `normal` = 8 evenly spaced samples
- 30 steps with `normal` = 30 evenly spaced samples

Low step counts (8-15) benefit more from careful scheduler choice since each step matters more.
