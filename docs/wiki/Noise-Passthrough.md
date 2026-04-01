# Noise Passthrough (SmartResCalc Integration)

## The -2 Seed Convention

When `noise_seed = -2` and the incoming latent has `use_as_noise = True`, DazzleKSampler uses the input latent directly as the sampling noise instead of generating new noise from the seed.

This enables SmartResCalc's noise shaping features:
- **Spectral blending** — image structure injected into noise via spectral composition
- **img2noise** — input image shapes the noise pattern
- **img2img + img2noise** — VAE-encoded image as starting latent + shaped noise

## How it works

SmartResCalc packages its output as a latent dict:

```python
{
    "samples": tensor,          # The starting latent (VAE-encoded or empty)
    "noise": tensor,            # Shaped noise (spectral blend, img2noise, etc.)
    "use_as_noise": True,       # Flag telling KSampler to use noise key
}
```

DazzleKSampler (in `SharkSampler.main()`) checks:
```python
use_latent_as_noise = latent.get("use_as_noise", False)
if use_latent_as_noise and noise_seed == -2:
    if "noise" in latent:
        noise = latent["noise"]    # Use shaped noise
    else:
        noise = x.clone()         # Use samples as noise
```

Both conditions must be true: the flag AND the -2 seed.

## Setting up the workflow

1. **SmartResCalc** — set `image_purpose` to one of:
   - `img2noise` — image shapes noise, independent of latent
   - `img2img + img2noise` — VAE-encode + shaped noise (most common)
   - `image + noise` — independent image and noise paths

2. **DazzleKSampler** — set `seed = -2` to activate passthrough

3. **Connect** SmartResCalc's `latent` output to DazzleKSampler's `latent_image` input

## Future: Smart Detection (#10)

Issue #10 tracks making this automatic — DazzleKSampler would inspect the latent dict and infer the intent without requiring the `-2` seed. If both `samples` and `noise` keys are present, it assumes passthrough mode.

## ClownsharKSampler compatibility

The `-2` seed convention is inherited from RES4LYF's ClownsharKSampler. Workflows using ClownsharKSampler with `-2` will work identically with DazzleKSampler.

## Troubleshooting

**"both_met=False" in logs:**
The `-2` seed was resolved to a real number before reaching DazzleKSampler. Common cause: an rgthree Seed node connected upstream that resolves special values. Connect the seed directly on DazzleKSampler's widget, not through an external seed node.

**Noise not being used:**
Check that SmartResCalc's `image_purpose` is set to a mode that produces noise (`img2noise`, `img2img + img2noise`, etc.). The `dimensions only` and `img2img` modes don't set `use_as_noise`.
