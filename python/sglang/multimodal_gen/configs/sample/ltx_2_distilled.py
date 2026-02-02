# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for LTX-2 Distilled Pipeline."""

import dataclasses
from typing import Optional

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams


@dataclasses.dataclass
class LTX2DistilledSamplingParams(LTX2SamplingParams):
    """Sampling parameters for LTX-2 Distilled Pipeline.

    The distilled pipeline operates in two stages:
    - Stage 1: Generate at half resolution (9 steps)
    - Stage 2: Upsample 2x and refine (4 steps)
    Total: 13 denoising steps for high-quality video+audio output.
    """

    # Default to higher resolution since distilled pipeline can handle it efficiently
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    fps: int = 24

    # Audio generation
    generate_audio: bool = True

    # Distilled pipeline uses fixed sigma values, so num_inference_steps is ignored
    # but we set it to 13 (9 + 4) to reflect the actual number of steps
    num_inference_steps: int = 13

    # Distilled pipeline typically doesn't use CFG (simple denoising)
    guidance_scale: float = 1.0

    # Image conditioning: list of (path, frame_idx, strength) tuples
    # Example: [("image.png", 0, 1.0)] to condition first frame
    images: Optional[list[tuple[str, int, float]]] = None

    # Whether to enhance the prompt using Gemma text encoder
    enhance_prompt: bool = False

    # Inherit negative prompt from base LTX2SamplingParams
