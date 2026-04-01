# Configuration Tips

## Common Presets

### Standard SDXL txt2img
```
sampler_name: multistep/dpmpp_2m
scheduler: beta57
steps: 28
eta: 0.5
cfg: 5.5
denoise: 1.0
seed: (any)
```

### SDXL img2img with SmartResCalc noise
```
sampler_name: multistep/dpmpp_3m
scheduler: sgm_uniform
steps: 19
eta: 0.5
cfg: 2.5
denoise: 0.5
seed: -2  (passthrough mode)
```

### Quick preview (low quality, fast)
```
sampler_name: linear/euler
scheduler: normal
steps: 8
eta: 0.0
cfg: 5.0
denoise: 1.0
```

### High quality (slow, detailed)
```
sampler_name: exponential/res_2s
scheduler: beta57
steps: 30
eta: 0.3
cfg: 7.0
denoise: 1.0
```

## Key Parameters

### eta (noise amount)
- `0.0` = deterministic (no ancestral noise)
- `0.5` = moderate ancestral noise (good default)
- `1.0` = full ancestral (maximum variety)
- Negative values have special meaning in some samplers

### cfg (classifier-free guidance)
- `1.0` = no guidance (model's raw prediction)
- `5.0-7.0` = typical range for SDXL
- `2.0-3.0` = typical for Flux/Qwen models
- Higher = more prompt adherence, less creativity
- Negative values activate channelwise CFG in DazzleKSampler

### denoise
- `1.0` = full denoising (txt2img)
- `0.3-0.7` = partial denoising (img2img)
- Lower = more of the original image preserved
- Negative values set `denoise_alt` (advanced feature)

### bongmath
- `True` (default) = enables richer state tracking (noise_initial, image_initial, state_info)
- `False` = simpler behavior, closer to standard k-diffusion
- Keep True unless you have a specific reason to disable

### sampler_mode
- `standard` = normal denoising
- `unsample` = reverse the diffusion process (add noise)
- `resample` = unsample then resample (style transfer)

## Scheduler Quick Reference

| Scheduler | Best for | Notes |
|-----------|----------|-------|
| `beta57` | General purpose | RES4LYF's custom schedule, alpha=0.5, beta=0.7 |
| `normal` | ComfyUI default | Standard linear schedule |
| `karras` | Smooth results | Cosine-based, good for detailed images |
| `sgm_uniform` | Flux/Qwen models | Uniform spacing in sigma space |
| `simple` | Quick generation | Evenly spaced, predictable |
