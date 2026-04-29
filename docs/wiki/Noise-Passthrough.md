# Noise Passthrough (SmartResCalc Integration)

## The `latent_role` widget (v0.1.2-alpha+)

DazzleKSampler exposes a **`latent_role`** widget on every Dazzle sampler node that controls how the input latent dict is interpreted. This is the modern, explicit replacement for the legacy `-2` seed convention.

| Role | Behavior |
|---|---|
| **`auto`** *(default)* | Inspect the dict shape and dispatch accordingly. Honors `use_as_noise=True` even without `seed=-2`; emits an informational notice the first step. The legacy `seed=-2` magic still works under `auto` (with a deprecation log). |
| **`noise`** | Use the upstream's noise tensor (whatever it is): Shape B → `samples`, Shape C → `noise` key. Init slot is zeroed in both cases (no img2img anchor — use `noise+latent_image` if you want img2img *with* layered noise). Shape A has no upstream noise and falls back to seed-driven generation with a warning. |
| **`latent_image`** | Force Shape A path: `samples` is used as the init image; noise is generated from `seed`. Any `use_as_noise` flag or `noise` key in the dict is ignored. |
| **`noise+latent_image`** | Force Shape C path: `samples` is the init image; the `noise` key is the noise tensor. Falls back to Shape A with a warning if no `noise` key is present. |
| **`seed_driven`** | True txt2img override: zero the init slot **and** generate noise from `seed`. Ignores upstream `samples` entirely (distinct from `latent_image`, which preserves `samples` as the img2img init). Useful for "what would this prompt produce from scratch?" comparisons. |

The seed widget is auto-hidden when `latent_role ∈ {"noise", "noise+latent_image"}` because the seed is unused on those paths (the noise comes from upstream, not from PRNG).

## The Four-Shape Latent-Dict Protocol

The latent dict can take one of four shapes. DazzleKSampler dispatches on shape, not on knowledge of the upstream node's mode.

| Shape | Dict | Produced by SmartResCalc when |
|---|---|---|
| **A — Pure init** | `{ "samples": vae_encoded_image }` | `image_purpose = img2img` (image + VAE attached) |
| **B — Pure noise** | `{ "samples": shaped_noise, "use_as_noise": True }` | `image_purpose ∈ {dimensions only, img2noise, image + noise}` with non-trivial `fill_type` |
| **C — Layered** | `{ "samples": encoded_image, "noise": shaped_noise, "use_as_noise": True }` | `image_purpose = img2img + img2noise` |
| **D — Empty** | `{ "samples": zeros }` | `fill_type ∈ {black, white, custom_color}` (trivial fill) OR ComfyUI's stock `EmptyLatentImage` |

Shape D is dispatch-equivalent to Shape A (the helper does not GPU-sync to detect zeros).

## Dispatch (modern, v0.1.2-alpha)

```python
# py/beta/_latent_noise_protocol.py
def resolve_latent_as_noise(
    latent_unbatch, x, noise_seed, latent_role="auto"
) -> LatentNoiseDispatch:
    """
    Returns a dataclass with:
        noise          -- noise tensor, or None to fall through to seed-driven
        x              -- init-image tensor (zeroed for Shape B dispatch)
        detected_shape -- "shape_a" / "shape_b" / "shape_c" from dict keys
        applied        -- the dispatch path actually taken
        warning        -- optional console message (mismatch / informational)
        deprecation    -- optional message (emitted on seed=-2 under auto)
    """
```

Dispatch matrix (5 roles × 3 detected shapes):

|              | SHAPE_A (no flags)  | SHAPE_B (use_as_noise) | SHAPE_C (use_as_noise + noise key) |
|--------------|--------------------|------------------------|------------------------------------|
| `auto`       | shape_a (seed)     | shape_b (info notice)  | shape_c (info notice)              |
| `noise`      | shape_a + warn (fallback to seed-driven) | shape_b | shape_b + warn (noise from `noise` key) |
| `latent_image` | shape_a          | shape_a + warn         | shape_a + warn                     |
| `noise+latent_image` | shape_a + warn | shape_a + warn       | shape_c                            |
| `seed_driven` (true txt2img) | seed_driven (init zeroed) + warn | seed_driven + warn | seed_driven + warn |

**Notable behaviors:**
- **`noise` on Shape C** picks the `noise` key, not `samples` (the v0.1.2-alpha Image #4 fix — using `samples` produced "image-as-noise" silhouette artifacts).
- **`noise` on Shape A** falls back to seed-driven instead of treating the encoded image as noise (same artifact class as above).
- **`seed_driven`** zeroes the init slot (true txt2img) — this is what makes it visibly different from `latent_image`, which preserves `samples` as init.

Under `auto`, the dispatch fires on the dict shape regardless of the seed value; `seed=-2` no longer changes behavior except to emit a deprecation log.

### Why the init slot is zeroed in Shape B

Without this, the same tensor occupies BOTH the noise role AND the init-image role passed to the underlying ComfyUI sampler:

```
initial_x = noise * sigmas[0] + latent_image
          = z_norm(samples) * sigmas[0] + samples       # <-- bug: same tensor twice
```

That double-anchoring locks composition to the latent's structure regardless of the upstream noise's character (fill_type, blend_strength). Symptom: changing `fill_type` from `DazNoise:Brown` to `DazNoise:Plasma` to anything else produces near-identical compositions, while changing the seed produces dramatic variation.

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
2. **DazzleKSampler** — leave `latent_role = auto` (default). The dispatch fires automatically based on the upstream dict shape.
3. **Connect** SmartResCalc's `latent` output to DazzleKSampler's `latent_image` input.

To force a specific path regardless of upstream shape, set `latent_role` explicitly. To debug an unintended dispatch, set `latent_role = seed_driven` to bypass any upstream shaping.

## Caveats

> [!important] These configurations do NOT trigger latent-as-noise mode
> - **`fill_type = black / white / custom_color`** — produces Shape D (no `use_as_noise` flag). Falls through to deterministic seed-driven noise generation.
> - **ComfyUI's stock `EmptyLatentImage` node** — also Shape D. Same fall-through behavior.
> - **`image_purpose = img2img` alone** (no img2noise) — produces Shape A. Falls through.
>
> If you need latent-as-noise behavior, ensure the upstream node sets `use_as_noise = True` (Shape B or C). With SmartResCalc, that means using a non-trivial `fill_type` (`noise`, `random`, or any `DazNoise:*`).

## Migration from `seed = -2`

The `-2` seed convention still works under `latent_role = auto` (with a one-time deprecation log per generation). To migrate:

| Pre-v0.1.2-alpha | v0.1.2-alpha+ equivalent |
|---|---|
| `seed = -2`, Shape B upstream | `latent_role = auto` (or `noise`); any seed value |
| `seed = -2`, Shape C upstream | `latent_role = auto` (or `noise+latent_image`); any seed value |
| `seed = -2`, Shape A upstream | `latent_role = auto` (no-op fall-through) — same behavior, no longer needs `-2` |
| `seed = N`, Shape B/C upstream | Pre-v0.1.2 silently fell through to seed-driven; v0.1.2 auto-mode now dispatches and prints an informational notice. To preserve old behavior, set `latent_role = seed_driven`. |

The `-2` magic value will lose its special meaning in a future release. Set `latent_role` explicitly to lock in the dispatch you want.

## ClownsharKSampler compatibility

The `-2` seed convention is inherited from RES4LYF's ClownsharKSampler. Workflows using upstream RES4LYF nodes with `-2` continue to work — but note that:
- DazzleKSampler's `latent_role` widget is a Dazzle-only addition; upstream RES4LYF does not have it.
- Shape B's init-zeroing fix is also DazzleKSampler-specific (added v0.1.1-alpha). A surgical equivalent has been applied to a local RES4LYF copy for upstream PR consideration but is not yet in the upstream `RES4LYF` repository.

## Troubleshooting

**"both_met=False" in logs (legacy):**
Pre-v0.1.2 message — the `-2` seed was resolved to a real number before reaching DazzleKSampler. Common cause: an rgthree Seed node connected upstream that resolves special values. With `latent_role = auto`, this is no longer a concern because dispatch is shape-driven, not seed-driven.

**Compositions look similar across runs even with seed-driven noise on Shape B:**
Pre-v0.1.1-alpha symptom of the dual-role bug — fixed by zeroing the init slot in Shape B. If the symptom persists after upgrading, verify the upstream is producing varying latents (use SmartResCalc's `[INVEST]` logging or a SmartResCalc → VAE Decode → Preview Image chain to confirm).

**Console shows "auto-mode dispatched to Shape B without seed=-2":**
This is the v0.1.2 informational notice. It means `latent_role = auto` saw `use_as_noise=True` in the dict and dispatched accordingly without needing `-2`. To suppress, set `latent_role` explicitly.

**Seed widget vanished:**
Expected behavior when `latent_role ∈ {"noise", "noise+latent_image"}` — the seed is unused on those paths (noise comes from upstream). Switch to `auto`, `latent_image`, or `seed_driven` to bring it back.

**Noise not being used:**
Check that the upstream produces a Shape B or C latent. The `dimensions only` mode with non-trivial `fill_type` (e.g., `DazNoise:Brown`) produces Shape B; `img2img + img2noise` produces Shape C; everything else falls through to seed-driven noise.

## Status of related issues

- **Issue #10** (smart latent input detection) — subsumed by Issue #11 / v0.1.2-alpha. The `latent_role = auto` mode now infers dispatch from the dict shape regardless of seed.
- **Issue #11** (explicit `latent_role` widget) — implemented in v0.1.2-alpha.
