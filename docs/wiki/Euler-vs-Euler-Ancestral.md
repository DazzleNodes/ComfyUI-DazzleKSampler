# Euler vs Euler Ancestral

## What's the difference?

Both are first-order ODE/SDE solvers for diffusion sampling. The key difference is **noise injection**:

| | Euler | Euler Ancestral |
|---|---|---|
| Method | Deterministic ODE solver | Stochastic SDE solver |
| Noise per step | None | Adds noise, then removes it |
| `eta` | 0.0 (or ignored) | 1.0 |
| Reproducibility | Same seed = identical output | Same seed = identical output |
| Image quality | Cleaner, more predictable | More varied, sometimes more creative |
| Convergence | Converges to a fixed point | Never fully converges (noise prevents it) |

## How ancestral sampling works

At each denoising step, a standard Euler sampler just takes a step toward the predicted clean image. An ancestral sampler does:

1. Take a step toward the predicted clean image (same as Euler)
2. **Add a controlled amount of noise back** (the "ancestral" part)
3. The next step denoises from this noisier state

The amount of noise added is controlled by `eta`:
- `eta = 0.0` = no noise added = standard Euler (deterministic)
- `eta = 1.0` = full ancestral noise = Euler Ancestral
- `0 < eta < 1.0` = partial ancestral (blend between deterministic and stochastic)

## In DazzleKSampler

DazzleKSampler (via the RES4LYF engine) uses the RK framework where "Euler" is the `linear/euler` sampler. To get ancestral behavior:

```
sampler_name: linear/euler
eta: 1.0
noise_mode: hard (via ClownOptions_SDE or baked into preset)
```

**Issue #9** tracks adding `linear/euler_ancestral` as a named preset so you don't need to configure eta manually.

## When to use which

### Use Euler (eta=0) when:
- You want reproducible results
- You're doing img2img (preserving source image structure)
- You want clean, predictable outputs
- You're using few steps (<15) and need stability

### Use Euler Ancestral (eta=1.0) when:
- You want more varied/creative results
- You're exploring compositions and want diversity
- You're doing txt2img and want the model to "surprise" you
- You have enough steps (20+) for the noise to be absorbed

### Use partial ancestral (0 < eta < 1.0) when:
- You want a balance between determinism and creativity
- `eta = 0.5` is a common middle ground
- Good for iterating: deterministic enough to be reproducible, stochastic enough for variety

## Other ancestral samplers

The ancestral concept applies to any sampler, not just Euler. In DazzleKSampler, setting `eta > 0` on any sampler adds ancestral noise injection:

- `multistep/dpmpp_2m` with `eta = 0.5` = partially ancestral DPM++ 2M
- `multistep/res_2m` with `eta = 1.0` = fully ancestral RES 2M
- `exponential/res_2s` with `eta = 0.3` = lightly ancestral exponential solver

The `noise_mode` parameter controls how the noise amplitude scales relative to the sigma schedule. `hard` is the standard ancestral behavior. Other modes (`soft`, `lorentzian`, `sinusoidal`) offer alternative noise scaling curves.

## Relationship to ComfyUI's built-in samplers

| ComfyUI name | DazzleKSampler equivalent |
|---|---|
| `euler` | `linear/euler` with `eta=0` |
| `euler_ancestral` | `linear/euler` with `eta=1.0` (or `linear/euler_ancestral` after #9) |
| `dpmpp_2m` | `multistep/dpmpp_2m` with `eta=0` |
| `dpmpp_2m_sde` | `multistep/dpmpp_2m` with `eta>0` + SDE noise |
| `heun` | `linear/heun` with `eta=0` |
