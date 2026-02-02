# SPDX-License-Identifier: Apache-2.0
"""LTX-2 Distilled Pipeline implementation."""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2TextConnectorStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_distilled_stages import (
    LTX2TwoStageDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2DistilledPipeline(ComposedPipelineBase):
    """
    LTX-2 Distilled Pipeline for fast, high-quality video+audio generation.
    
    This pipeline uses a two-stage distilled approach:
    - Stage 1: Generate at half resolution (9 denoising steps)
    - Stage 2: Upsample 2x and refine (4 denoising steps)
    
    Total: Only 13 steps vs 40+ for standard pipelines, with comparable quality.
    """

    # NOTE: must match `model_index.json`'s `_class_name` for native dispatch.
    pipeline_name = "LTX2DistilledPipeline"

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
        "spatial_upsampler",  # Additional module for distilled pipeline
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages for distilled generation."""

        # 1. Input Validation
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        # 2. Text Encoding
        # LTX-2 needs two contexts (video/audio). We reuse the same
        # underlying Gemma encoder/tokenizer twice.
        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                ],
            ),
        )

        # 3. Text Connector Stage
        self.add_stage(
            stage_name="text_connector_stage",
            stage=LTX2TextConnectorStage(connectors=self.get_module("connectors")),
        )

        # 4. Two-Stage Denoising
        # This stage handles both stage 1 (half resolution) and stage 2 (full resolution)
        # including the upsampling between stages
        self.add_stage(
            stage_name="two_stage_denoising",
            stage=LTX2TwoStageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                upsampler=self.get_module("spatial_upsampler"),
            ),
        )

        # 5. Decoding
        # Reuse the standard LTX2AVDecodingStage since decoding logic is the same
        self.add_stage(
            stage_name="decoding_stage",
            stage=LTX2AVDecodingStage(
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
                pipeline=self,
            ),
        )


EntryClass = LTX2DistilledPipeline
