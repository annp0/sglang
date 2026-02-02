# SPDX-License-Identifier: Apache-2.0
"""Custom pipeline stages for LTX-2 Distilled Pipeline."""

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.upsamplers.latent_upsampler import (
    upsample_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LTX2LatentUpsamplingStage(PipelineStage):
    """
    Stage that upsamples video latents using the LatentUpsampler.
    
    This stage:
    1. Un-normalizes latents using video encoder's per_channel_statistics
    2. Applies the upsampler (2x spatial upsampling)
    3. Re-normalizes latents
    """

    def __init__(self, upsampler, video_encoder):
        super().__init__()
        self.upsampler = upsampler
        self.video_encoder = video_encoder

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Upsample video latents."""
        device = get_local_torch_device()
        self.upsampler = self.upsampler.to(device)
        self.video_encoder = self.video_encoder.to(device)
        
        latents = batch.latents.to(device)
        
        # Upsample using the helper function
        upsampled_latents = upsample_video(latents, self.video_encoder, self.upsampler)
        
        batch.latents = upsampled_latents
        return batch


class LTX2TwoStageDenoisingStage(PipelineStage):
    """
    Two-stage denoising for LTX-2 Distilled Pipeline.
    
    Stage 1: Generate at half resolution with 9 steps
    Stage 2: Upsample 2x and refine with 4 steps
    """

    def __init__(self, transformer, scheduler, vae, audio_vae, upsampler):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.audio_vae = audio_vae
        self.upsampler = upsampler

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        result.add_check("timesteps", batch.timesteps, V.is_tensor)
        return result

    def _euler_step(
        self, noisy: torch.Tensor, denoised: torch.Tensor, sigma: float, sigma_next: float
    ) -> torch.Tensor:
        """Single Euler denoising step."""
        velocity = (noisy - denoised) / sigma
        return denoised + sigma_next * velocity

    def _simple_denoise(
        self,
        noisy_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        timestep: torch.Tensor,
        batch: Req,
        server_args: ServerArgs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple denoising without CFG."""
        device = get_local_torch_device()
        
        # Prepare model inputs
        model_input = torch.cat([noisy_latents, audio_latents], dim=1)
        encoder_hidden_states = torch.cat([video_context, audio_context], dim=1)
        
        # Run transformer
        noise_pred = self.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        
        # Split video and audio predictions
        video_channels = noisy_latents.shape[1]
        video_pred = noise_pred[:, :video_channels]
        audio_pred = noise_pred[:, video_channels:]
        
        return video_pred, audio_pred

    def _stage_1_denoising(
        self,
        batch: Req,
        server_args: ServerArgs,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Generate at half resolution."""
        device = get_local_torch_device()
        config = server_args.pipeline_config
        
        # Get stage 1 sigmas
        stage_1_sigmas = torch.tensor(config.stage_1_sigmas, device=device)
        
        # Prepare latents at half resolution
        batch_size = batch.batch_size
        height_half = batch.height // 2
        width_half = batch.width // 2
        
        # Calculate latent dimensions for half resolution
        latent_height = height_half // config.vae_scale_factor
        latent_width = width_half // config.vae_scale_factor
        latent_num_frames = (batch.num_frames - 1) // config.vae_temporal_compression + 1
        
        # Initialize video latents
        video_shape = (
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )
        video_latents = randn_tensor(
            video_shape,
            generator=batch.generator,
            device=device,
            dtype=video_context.dtype,
        )
        video_latents = video_latents * stage_1_sigmas[0]
        
        # Pack video latents
        video_latents = config.maybe_pack_latents(video_latents, batch_size, batch)
        
        # Initialize audio latents
        audio_shape = config.prepare_audio_latent_shape(
            batch, batch_size, batch.num_frames
        )
        audio_latents = randn_tensor(
            audio_shape,
            generator=batch.generator,
            device=device,
            dtype=audio_context.dtype,
        )
        audio_latents = audio_latents * stage_1_sigmas[0]
        audio_latents = config.maybe_pack_audio_latents(audio_latents, batch_size, batch)
        
        # Euler denoising loop
        for i in range(len(stage_1_sigmas) - 1):
            sigma = stage_1_sigmas[i]
            sigma_next = stage_1_sigmas[i + 1]
            
            # Create timestep tensor
            timestep = torch.tensor([sigma], device=device, dtype=video_context.dtype)
            
            # Denoise
            video_denoised, audio_denoised = self._simple_denoise(
                video_latents,
                audio_latents,
                video_context,
                audio_context,
                timestep,
                batch,
                server_args,
            )
            
            # Euler step
            video_latents = self._euler_step(
                video_latents, video_denoised, sigma.item(), sigma_next.item()
            )
            audio_latents = self._euler_step(
                audio_latents, audio_denoised, sigma.item(), sigma_next.item()
            )
        
        return video_latents, audio_latents

    def _stage_2_denoising(
        self,
        batch: Req,
        server_args: ServerArgs,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        upsampled_video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 2: Upsample and refine at full resolution."""
        device = get_local_torch_device()
        config = server_args.pipeline_config
        
        # Get stage 2 sigmas
        stage_2_sigmas = torch.tensor(config.stage_2_sigmas, device=device)
        noise_scale = stage_2_sigmas[0].item()
        
        # Add noise at noise_scale
        noise = randn_tensor(
            upsampled_video_latents.shape,
            generator=batch.generator,
            device=device,
            dtype=upsampled_video_latents.dtype,
        )
        video_latents = upsampled_video_latents + noise * noise_scale
        
        # Also add noise to audio
        noise_audio = randn_tensor(
            audio_latents.shape,
            generator=batch.generator,
            device=device,
            dtype=audio_latents.dtype,
        )
        audio_latents = audio_latents + noise_audio * noise_scale
        
        # Euler denoising loop
        for i in range(len(stage_2_sigmas) - 1):
            sigma = stage_2_sigmas[i]
            sigma_next = stage_2_sigmas[i + 1]
            
            # Create timestep tensor
            timestep = torch.tensor([sigma], device=device, dtype=video_context.dtype)
            
            # Denoise
            video_denoised, audio_denoised = self._simple_denoise(
                video_latents,
                audio_latents,
                video_context,
                audio_context,
                timestep,
                batch,
                server_args,
            )
            
            # Euler step
            video_latents = self._euler_step(
                video_latents, video_denoised, sigma.item(), sigma_next.item()
            )
            audio_latents = self._euler_step(
                audio_latents, audio_denoised, sigma.item(), sigma_next.item()
            )
        
        return video_latents, audio_latents

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute two-stage denoising."""
        device = get_local_torch_device()
        
        # Load models
        self.transformer = self.transformer.to(device)
        self.vae = self.vae.to(device)
        self.audio_vae = self.audio_vae.to(device)
        self.upsampler = self.upsampler.to(device)
        
        # Get prompt embeddings (video and audio contexts)
        if isinstance(batch.prompt_embeds, list) and len(batch.prompt_embeds) == 2:
            video_context = batch.prompt_embeds[0].to(device)
            audio_context = batch.prompt_embeds[1].to(device)
        else:
            # Fallback: use same context for both
            video_context = batch.prompt_embeds.to(device)
            audio_context = batch.prompt_embeds.to(device)
        
        # Stage 1: Denoise at half resolution
        logger.info("Stage 1: Denoising at half resolution (9 steps)")
        video_latents_s1, audio_latents_s1 = self._stage_1_denoising(
            batch, server_args, video_context, audio_context
        )
        
        # Unpack latents for upsampling
        config = server_args.pipeline_config
        latent_height_half = (batch.height // 2) // config.vae_scale_factor
        latent_width_half = (batch.width // 2) // config.vae_scale_factor
        latent_num_frames = (batch.num_frames - 1) // config.vae_temporal_compression + 1
        
        video_latents_unpacked = config._unpack_latents(
            video_latents_s1,
            latent_num_frames,
            latent_height_half,
            latent_width_half,
            config.patch_size,
            config.patch_size_t,
        )
        
        # Upsample video latents
        logger.info("Upsampling video latents 2x")
        upsampled_latents = upsample_video(
            video_latents_unpacked, self.vae, self.upsampler
        )
        
        # Pack upsampled latents
        upsampled_latents_packed = config.maybe_pack_latents(
            upsampled_latents, batch.batch_size, batch
        )
        
        # Stage 2: Refine at full resolution
        logger.info("Stage 2: Refining at full resolution (4 steps)")
        video_latents_final, audio_latents_final = self._stage_2_denoising(
            batch,
            server_args,
            video_context,
            audio_context,
            upsampled_latents_packed,
            audio_latents_s1,
        )
        
        # Store final latents
        batch.latents = video_latents_final
        batch.audio_latents = audio_latents_final
        
        return batch


class LTX2DistilledDecodingStage(PipelineStage):
    """
    Decoding stage for LTX-2 Distilled Pipeline.
    
    Decodes both video and audio latents.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.pipeline = pipeline

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Decode video and audio latents."""
        device = get_local_torch_device()
        
        # Load models
        self.vae = self.vae.to(device)
        self.vae.eval()
        
        # Get latents
        latents = batch.latents.to(device)
        
        # Decode video
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast
        
        # Unpack and denormalize latents
        config = server_args.pipeline_config
        latents, audio_latents = config._unpad_and_unpack_latents(
            latents, batch.audio_latents, batch, self.vae, self.audio_vae
        )
        
        # Scale and shift for decoding
        scale, shift = config.get_decode_scale_and_shift(
            device, latents.dtype, self.vae
        )
        if shift is not None:
            latents = (latents - shift) * scale
        else:
            latents = latents * scale
        
        # Decode video
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            
            decode_output = self.vae.decode(latents)
            if isinstance(decode_output, tuple):
                video = decode_output[0]
            elif hasattr(decode_output, "sample"):
                video = decode_output.sample
            else:
                video = decode_output
        
        # Post-process video
        from diffusers.video_processor import VideoProcessor
        video_processor = VideoProcessor(vae_scale_factor=32)
        video = video_processor.postprocess_video(video, output_type="np")
        
        # Decode audio
        if batch.audio_latents is not None:
            self.audio_vae = self.audio_vae.to(device)
            self.vocoder = self.vocoder.to(device)
            self.audio_vae.eval()
            self.vocoder.eval()
            
            audio_latents = audio_latents.to(device)
            
            # Decode audio VAE
            with torch.no_grad():
                audio_spec = self.audio_vae.decode(audio_latents).sample
            
            # Vocode to waveform
            with torch.no_grad():
                audio = self.vocoder(audio_spec).cpu().numpy()
            
            batch.audio = audio
        
        batch.output = video
        return batch
