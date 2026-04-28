# Noise Passthrough (SmartResCalc Integration)

## The -2 Seed Convention

When `noise_seed = -2` and the incoming latent has `use_as_noise = True`, DazzleKSampler uses the input latent directly as the sampling noise instead of generating new noise from the seed.

This enables SmartResCalc's noise shaping features:
- **Spectral blending** — image structure injected into noise via spectral composition
- **img2noise** — input image shapes the noise pattern
- **img2img + img2noise** — VAE-encoded image as starting latent + shaped noise

## The Four-Shape Latent-Dict Protocol

The latent dict can take one of four shapes. DazzleKSampler dispatches on shape, not on knowledge of the upstream node's mode.

| Shape | Dict | Produced by SmartResCalc when |
|---|---|---|
| **A — Pure init** | `{ "samples": vae_encoded_image }` | `image_purpose = img2img` (image + VAE attached) |
| **B — Pure noise** | `{ "samples": shaped_noise, "use_as_noise": True }` | `image_purpose ∈ {dimensions only, img2noise, image + noise}` with non-trivial `fill_type` |
| **C — Layered** | `{ "samples": encoded_image, "noise": shaped_noise, "use_as_noise": True }` | `image_purpose = img2img + img2noise` |
| **D — Empty** | `{ "samples": zeros }` | `fill_type ∈ {black, white, custom_color}` (trivial fill) OR ComfyUI's stock `EmptyLatentImage` |

## DazzleKSampler dispatch with `seed = -2`

```python
# py/beta/_latent_noise_protocol.py
def resolve_latent_as_noise(latent_unbatch, x, noise_seed):
    use_latent_as_noise = latent_unbatch.get("use_as_noise", False)
    if not (use_latent_as_noise and noise_seed == -2):
        return None, x, "not_applied"   # Shapes A, D — fall through to seed-driven noise

    if "noise" in latent_unbatch:
        # Shape C — img2img + img2noise: separate noise key, samples stays as init
        noise = latent_unbatch["noise"].to(device=x.device, dtype=x.dtype)
        return noise, x, "shape_c"

    # Shape B — samples IS the noise; init must be zeroed so samples isn't dual-roled
    noise = x.clone()
    x_init = torch.zeros_like(x)        # <<< prevents dual-role anchoring
    return noise, x_init, "shape_b"
```

### Why the init slot is zeroed in Shape B

Without this, the same tensor occupies BOTH the noise role AND the init-image role passed to the underlying ComfyUI sampler:

```
initial_x = noise * sigmas[0] + latent_image
          = z_norm(samples) * sigmas[0] + samples       # <-- bug: same tensor twice
```

That double-anchoring locks composition to the latent's structure regardless of the upstream noise's character (fill_type, blend_strength). Symptom: changing `fill_type` from `DazNoise:Brown` to `DazNoise:Plasma` to anything else produces near-identical compositions, while changing the seed (away from `-2`) produces dramatic variation.

The fix zeroes the init slot in Shape B so the latent acts purely as noise:

```
initial_x = noise * sigmas[0] + 0
          = z_norm(samples) * sigmas[0]
```

Shape C is unchanged — the separate `noise` key means `samples` still plays its rightful init role, and `noise` plays its rightful noise role. img2img-with-shaped-noise semantics are preserved.

## Setting up the workflow

1. **SmartResCalc** — pick an `image_purpose` that produces a Shape B or Shape C latent:
   - `dimensions only` + non-trivial `fill_type` → Shape B
   - `img2noise` (with image attached) → Shape B
   - `image + noise` → Shape B
   - `img2img + img2noise` → Shape C
2. **DazzleKSampler** — set `seed = -2` to activate passthrough.
3. **Connect** SmartResCalc's `latent` output to DazzleKSampler's `latent_image` input.

## Caveats

> [!important] These configurations do NOT trigger latent-as-noise mode
> - **`fill_type = black / white / custom_color`** — produces Shape D (no `use_as_noise` flag). `seed = -2` falls through to deterministic seed-driven noise generation.
> - **ComfyUI's stock `EmptyLatentImage` node** — also Shape D. Same fall-through behavior.
> - **`image_purpose = img2img` alone** (no img2noise) — produces Shape A. `seed = -2` falls through.
>
> If you need latent-as-noise behavior, ensure the upstream node sets `use_as_noise = True` (Shape B or C). With SmartResCalc, that means using a non-trivial `fill_type` (`noise`, `random`, or any `DazNoise:*`).

## Future: Smart Detection (#10)

Issue #10 tracks making this automatic — DazzleKSampler would inspect the latent dict and infer the intent without requiring the `-2` seed. If both `samples` and `noise` keys are present (Shape C), or if `use_as_noise=True` (Shape B), the sampler could activate passthrough.

## ClownsharKSampler compatibility

The `-2` seed convention is inherited from RES4LYF's ClownsharKSampler. Workflows using ClownsharKSampler with `-2` will work with DazzleKSampler — but note that DazzleKSampler additionally honors the four-shape protocol (Shape B's init-zeroing is a DazzleKSampler-specific fix, added in v0.1.1-alpha).

## Troubleshooting

**"both_met=False" in logs:**
The `-2` seed was resolved to a real number before reaching DazzleKSampler. Common cause: an rgthree Seed node connected upstream that resolves special values. Connect the seed directly on DazzleKSampler's widget, not through an external seed node.

**Compositions look similar across runs even with `seed = -2`:**
Pre-v0.1.1-alpha symptom of the dual-role bug — fixed by zeroing the init slot in Shape B. If the symptom persists after upgrading, verify the upstream is producing varying latents (use SmartResCalc's `[INVEST]` logging or a SmartResCalc → VAE Decode → Preview Image chain to confirm).

**Noise not being used:**
Check that the upstream produces a Shape B or C latent. The `dimensions only` mode with non-trivial `fill_type` (e.g., `DazNoise:Brown`) produces Shape B; `img2img + img2noise` produces Shape C; everything else falls through to seed-driven noise.
