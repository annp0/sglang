# SPDX-License-Identifier: Apache-2.0
"""Configuration for LTX-2 Distilled Pipeline."""

import dataclasses
from dataclasses import field
from typing import Optional

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig


# Distilled sigma values from the reference implementation
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


@dataclasses.dataclass
class LTX2DistilledPipelineConfig(LTX2PipelineConfig):
    """Configuration for LTX-2 Distilled Pipeline.
    
    This extends the base LTX2PipelineConfig with distilled-specific settings:
    - Two-stage denoising with fixed sigma values
    - Spatial upsampler for stage 2
    - Resolution divisor of 64 (required for two-stage operation)
    """

    # Distilled sigma values for stage 1 and stage 2
    stage_1_sigmas: list[float] = field(default_factory=lambda: DISTILLED_SIGMA_VALUES)
    stage_2_sigmas: list[float] = field(
        default_factory=lambda: STAGE_2_DISTILLED_SIGMA_VALUES
    )

    # Path to the spatial upsampler weights (can be overridden)
    spatial_upsampler_path: Optional[str] = None

    # Two-stage pipelines require divisibility by 64
    resolution_divisor: int = 64

    def prepare_sigmas(self, sigmas, num_inference_steps):
        """Override to use fixed distilled sigmas instead of computed ones."""
        # For distilled pipeline, we ignore num_inference_steps and always use
        # the fixed distilled sigma values
        if sigmas is not None:
            return sigmas
        # This shouldn't be called for distilled pipeline as sigmas are pre-defined
        return self.stage_1_sigmas
