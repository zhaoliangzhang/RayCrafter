"""Wan RGB-D pipeline components for masked reference RGB/depth slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..diffusion.base_pipeline import PipelineUnit
from .wan_video_rgbd import (
    WanVideoUnit_InputRGBDRasters,
    _raster_to_bchw,
    model_fn_wan_video_rgbd,
)


class WanVideoUnit_ImageEmbedderVAE_RGBDReferenceDepth(PipelineUnit):
    """Encode four known slots and six paired noisy query conditions.

    ``input_validity_masks`` changes only the spatial known-mask supplied to
    the DiT. Query slots remain unknown because they are generated outputs;
    their semantic/dropout masks have already been applied to the raster data.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "num_latent_frames",
                "conditioning_images",
                "input_validity_masks",
            ),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        num_latent_frames=None,
        conditioning_images=None,
        input_validity_masks=None,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        if not pipe.dit.individual_encoding:
            return {}
        if end_image is not None:
            raise ValueError("Reference-depth RGB-D does not support end_image")
        pipe.load_models_to_device(self.onload_model_names)

        known = [
            _raster_to_bchw(pipe, raster, height, width) for raster in input_image
        ]
        num_input = len(known)
        num_total = num_latent_frames if num_latent_frames is not None else num_frames
        num_output = num_total - num_input
        if conditioning_images is None or len(conditioning_images) != num_output:
            raise ValueError(
                f"Expected {num_output} conditioned output rasters, got "
                f"{0 if conditioning_images is None else len(conditioning_images)}"
            )
        unknown = [
            _raster_to_bchw(pipe, raster, height, width)
            for raster in conditioning_images
        ]

        latent_height = height // 8
        latent_width = width // 8
        if input_validity_masks is None:
            known_mask = torch.ones(
                1,
                num_input,
                latent_height,
                latent_width,
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
        else:
            known_mask = input_validity_masks
            if not isinstance(known_mask, torch.Tensor):
                known_mask = torch.as_tensor(known_mask)
            if known_mask.ndim == 4 and known_mask.shape[0] == 1:
                known_mask = known_mask[0]
            if known_mask.shape != (num_input, height, width):
                raise ValueError(
                    "Expected input validity masks "
                    f"{(num_input, height, width)}, got {tuple(known_mask.shape)}"
                )
            known_mask = F.interpolate(
                known_mask.float().unsqueeze(1),
                size=(latent_height, latent_width),
                mode="area",
            )[:, 0]
            # A latent cell touching an exclusion region is not marked known.
            known_mask = (known_mask >= 1.0 - 1e-6).to(
                device=pipe.device, dtype=pipe.torch_dtype
            ).unsqueeze(0)

        output_mask = torch.zeros(
            1,
            num_output,
            latent_height,
            latent_width,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        mask = torch.cat([known_mask, output_mask], dim=1)
        mask = torch.repeat_interleave(mask, repeats=4, dim=1)
        mask = mask.view(
            1, num_total, 4, latent_height, latent_width
        ).transpose(1, 2)[0]

        encoded = []
        for raster in known + unknown:
            latent = pipe.vae.encode(
                [
                    raster[0]
                    .unsqueeze(1)
                    .to(dtype=pipe.torch_dtype, device=pipe.device)
                ],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            encoded.append(latent)
        y = torch.cat(encoded, dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": torch.cat([mask, y]).unsqueeze(0).to(pipe.torch_dtype)}


def install_rgbd_reference_depth_pipeline_components(pipe) -> None:
    """Install the masked 10-slot raster path without changing DiT weights."""
    if len(pipe.units) < 5:
        raise ValueError("Unexpected Wan pipeline unit layout")
    pipe.units[3] = WanVideoUnit_InputRGBDRasters()
    pipe.units[4] = WanVideoUnit_ImageEmbedderVAE_RGBDReferenceDepth()
    pipe.model_fn = model_fn_wan_video_rgbd
