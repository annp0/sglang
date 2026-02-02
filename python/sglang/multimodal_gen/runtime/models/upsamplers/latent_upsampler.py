# SPDX-License-Identifier: Apache-2.0
"""
Latent upsampler for LTX-2 distilled pipeline.

Ported from: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/upsampler/model.py
"""

import torch
from einops import rearrange


class PixelShuffleND(torch.nn.Module):
    """
    N-dimensional pixel shuffle operation.

    Args:
        dims: Number of dimensions to unshuffle (1, 2, or 3)
    """

    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 1:
            # Temporal only: [B, 2*C, F, H, W] -> [B, C, 2*F, H, W]
            b, c, f, h, w = x.shape
            c_out = c // 2
            x = rearrange(x, "b (c s) f h w -> b c (f s) h w", s=2, c=c_out)
        elif self.dims == 2:
            # Spatial only: [B, 4*C, H, W] -> [B, C, 2*H, 2*W]
            b, c, h, w = x.shape
            c_out = c // 4
            x = rearrange(x, "b (c s1 s2) h w -> b c (h s1) (w s2)", s1=2, s2=2, c=c_out)
        elif self.dims == 3:
            # Spatiotemporal: [B, 8*C, F, H, W] -> [B, C, 2*F, 2*H, 2*W]
            b, c, f, h, w = x.shape
            c_out = c // 8
            x = rearrange(
                x, "b (c s1 s2 s3) f h w -> b c (f s1) (h s2) (w s3)", s1=2, s2=2, s3=2, c=c_out
            )
        else:
            raise ValueError(f"Unsupported dims={self.dims}. Must be 1, 2, or 3.")
        return x


class ResBlock(torch.nn.Module):
    """
    Residual block for the upsampler.

    Args:
        channels: Number of channels
        dims: Number of dimensions for convolutions (2 or 3)
    """

    def __init__(self, channels: int, dims: int = 3):
        super().__init__()
        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        
        self.norm1 = torch.nn.GroupNorm(32, channels)
        self.conv1 = conv(channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.conv2 = conv(channels, channels, kernel_size=3, padding=1)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual


class LatentUpsampler(torch.nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.

    Args:
        in_channels: Number of channels in the input latent
        mid_channels: Number of channels in the middle layers
        num_blocks_per_stage: Number of ResBlocks to use in each stage (pre/post upsampling)
        dims: Number of dimensions for convolutions (2 or 3)
        spatial_upsample: Whether to spatially upsample the latent
        temporal_upsample: Whether to temporally upsample the latent
        spatial_scale: Scale factor for spatial upsampling (unused for now)
        rational_resampler: Whether to use a rational resampler for spatial upsampling (unused)
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                raise NotImplementedError("Rational resampler not implemented")
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # remove the first frame after upsampling.
                # This is done because the first frame encodes one pixel frame.
                x = x[:, :, 1:, :, :]
            else:
                # Spatial upsampling
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


def upsample_video(latent: torch.Tensor, video_encoder, upsampler: LatentUpsampler) -> torch.Tensor:
    """
    Apply upsampling to the latent representation using the provided upsampler,
    with normalization and un-normalization based on the video encoder's per-channel statistics.

    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LatentUpsampler module to perform upsampling.

    Returns:
        torch.Tensor: Upsampled and re-normalized latent tensor.
    """
    # Un-normalize using video encoder statistics
    if hasattr(video_encoder, "per_channel_statistics"):
        latent = video_encoder.per_channel_statistics.un_normalize(latent)
    
    # Apply upsampling
    latent = upsampler(latent)
    
    # Re-normalize using video encoder statistics
    if hasattr(video_encoder, "per_channel_statistics"):
        latent = video_encoder.per_channel_statistics.normalize(latent)
    
    return latent
