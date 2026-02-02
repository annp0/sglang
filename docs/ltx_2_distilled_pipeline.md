# LTX-2 Distilled Pipeline

## Overview

The LTX-2 Distilled Pipeline is a high-performance video+audio generation pipeline that produces quality comparable to the standard LTX-2 pipeline but in only 13 denoising steps (compared to 40+).

## How It Works

The pipeline uses a two-stage distilled approach:

### Stage 1: Low Resolution Generation
- Generates video at **half resolution** (height//2, width//2)
- Uses **9 denoising steps** with fixed sigma values: 
  - `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`
- Produces initial video and audio latents

### Stage 2: Upsample and Refine
- Upsamples video latents **2x** using spatial LatentUpsampler
- Adds noise at sigma=0.909375
- Refines with **4 additional denoising steps**:
  - `[0.909375, 0.725, 0.421875, 0.0]`
- Produces final high-quality video and audio

## Architecture

### Key Components

1. **LatentUpsampler** (`runtime/models/upsamplers/latent_upsampler.py`)
   - Spatial 2x upsampler for video latents
   - Uses PixelShuffle and ResBlock architecture
   - Handles normalization via video encoder statistics

2. **LTX2TwoStageDenoisingStage** (`runtime/pipelines_core/stages/ltx_2_distilled_stages.py`)
   - Implements the complete two-stage denoising process
   - Uses Euler method for ODE integration
   - Operates without CFG (simple denoising) for efficiency

3. **LTX2DistilledPipelineConfig** (`configs/pipeline_configs/ltx_2_distilled.py`)
   - Extends `LTX2PipelineConfig`
   - Adds distilled sigma values for both stages
   - Requires `resolution_divisor = 64` (two-stage constraint)

4. **LTX2DistilledSamplingParams** (`configs/sample/ltx_2_distilled.py`)
   - Default: 1024x1536 resolution, 121 frames
   - 13 total denoising steps
   - Optional image conditioning support

## Usage

### Basic Generation

```python
from sglang.multimodal_gen import DiffGenerator

# Initialize generator
generator = DiffGenerator(
    model_path="path/to/ltx2-distilled",
)

# Generate video+audio
output = generator.generate(
    prompt="A cat playing piano",
    height=1024,
    width=1536,
    num_frames=121,
)
```

### With Image Conditioning

```python
output = generator.generate(
    prompt="A cat playing piano",
    height=1024,
    width=1536,
    num_frames=121,
    images=[
        ("path/to/image.png", 0, 1.0),  # (path, frame_idx, strength)
    ],
)
```

## Model Requirements

The distilled pipeline requires:

1. **Standard LTX-2 components**:
   - Transformer (DiT model)
   - Text encoder (Gemma)
   - Video VAE
   - Audio VAE
   - Vocoder
   - Text connectors

2. **Additional component**:
   - `spatial_upsampler`: LatentUpsampler for 2x spatial upsampling

## Resolution Constraints

Due to the two-stage architecture:
- Both height and width must be divisible by **64**
- Stage 1 operates at half resolution (height//2, width//2)
- Stage 2 operates at full resolution (height, width)

Common resolutions:
- 1024x1536 (default)
- 768x1280
- 512x768

## Performance

### Speed Comparison
- **Standard LTX-2**: 40+ steps
- **Distilled LTX-2**: 13 steps (9 + 4)
- **Speedup**: ~3x faster

### Quality
- Comparable to standard pipeline due to distillation training
- Optimized sigma schedules maintain output quality

## Implementation Details

### Euler Denoising

The pipeline uses the Euler method for flow-based ODE integration:

```python
# Get velocity prediction from model
v = model(x_t, t)

# Convert to denoised latent: x_0 = x_t - sigma * v
x_0 = x_t - sigma * v

# Euler step: x_{t+1} = x_0 + sigma_next * v
x_next = x_0 + sigma_next * v
```

### Latent Upsampling

The upsampler operates in normalized latent space:

```python
# 1. Un-normalize latents
latents = vae.per_channel_statistics.un_normalize(latents)

# 2. Apply 2x spatial upsampler  
latents = upsampler(latents)

# 3. Re-normalize latents
latents = vae.per_channel_statistics.normalize(latents)
```

## Reference

Original implementation: [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)

Key files:
- `packages/ltx-pipelines/src/ltx_pipelines/distilled.py`
- `packages/ltx-core/src/ltx_core/model/upsampler/model.py`

## Files Added/Modified

### New Files
- `python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2_distilled.py`
- `python/sglang/multimodal_gen/configs/sample/ltx_2_distilled.py`
- `python/sglang/multimodal_gen/runtime/models/upsamplers/latent_upsampler.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_distilled_stages.py`
- `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_distilled_pipeline.py`

### Modified Files
- `python/sglang/multimodal_gen/registry.py` (added registration)
- `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py` (added exports)
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py` (added exports)
