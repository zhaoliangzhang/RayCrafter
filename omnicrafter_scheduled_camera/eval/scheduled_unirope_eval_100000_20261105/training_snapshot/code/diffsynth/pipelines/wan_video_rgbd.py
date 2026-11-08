"""Isolated Wan/FrameCrafter components for float RGB-D raster slots."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from ..diffusion.base_pipeline import PipelineUnit
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d


def _raster_to_bchw(
    pipe,
    raster: Image.Image | torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert a PIL raster or float [-1,1] CHW raster to B,C,H,W."""
    if isinstance(raster, Image.Image):
        return pipe.preprocess_image(raster.resize((width, height))).to(pipe.device)
    if not isinstance(raster, torch.Tensor):
        raise TypeError(f"Unsupported raster type: {type(raster)!r}")
    if raster.ndim == 3:
        raster = raster.unsqueeze(0)
    if raster.ndim != 4 or raster.shape[1] != 3:
        raise ValueError(f"Expected [3,H,W] or [B,3,H,W], got {raster.shape}")
    raster = raster.to(device=pipe.device, dtype=pipe.torch_dtype)
    if raster.shape[-2:] != (height, width):
        raster = F.interpolate(
            raster, size=(height, width), mode="bilinear", align_corners=False
        )
    return raster


class WanVideoUnit_InputRGBDRasters(PipelineUnit):
    """VAE-encode RGB and float depth target slots independently."""

    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None or not pipe.dit.individual_encoding:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        rasters = [
            _raster_to_bchw(
                pipe,
                raster,
                height=noise.shape[-2] * pipe.vae.upsampling_factor,
                width=noise.shape[-1] * pipe.vae.upsampling_factor,
            )
            for raster in input_video
        ]
        input_latents = [
            pipe.vae.encode(
                raster.unsqueeze(2),
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            for raster in rasters
        ]
        input_latents = torch.cat(input_latents, dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "mask_loss": True}
        latents = pipe.scheduler.add_noise(
            input_latents, noise, timestep=pipe.scheduler.timesteps[0]
        )
        return {"latents": latents}


class WanVideoUnit_ImageEmbedderVAE_RGBDIndividualEncoding(PipelineUnit):
    """Build Wan's known/unknown VAE condition from float raster slots."""

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
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        if not pipe.dit.individual_encoding:
            return {}
        if end_image is not None:
            raise ValueError("The RGB-D path does not support end_image conditioning")
        pipe.load_models_to_device(self.onload_model_names)

        known = [
            _raster_to_bchw(pipe, raster, height, width) for raster in input_image
        ]
        num_input = len(known)
        num_total = num_latent_frames if num_latent_frames is not None else num_input + 1
        num_output = num_total - num_input
        if conditioning_images is None or len(conditioning_images) != num_output:
            raise ValueError(
                f"Expected {num_output} RGB-D conditioning rasters, got "
                f"{0 if conditioning_images is None else len(conditioning_images)}"
            )
        unknown = [
            _raster_to_bchw(pipe, raster, height, width)
            for raster in conditioning_images
        ]
        vae_input = known + unknown

        mask = torch.ones(1, num_total, height // 8, width // 8, device=pipe.device)
        mask[:, num_input:] = 0
        mask = torch.repeat_interleave(mask, repeats=4, dim=1)
        mask = mask.view(1, num_total, 4, height // 8, width // 8)
        mask = mask.transpose(1, 2)[0]

        encoded = []
        for raster in vae_input:
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


def install_rgbd_pipeline_components(pipe) -> None:
    """Replace only the two raster units and model function on a new pipeline."""
    if len(pipe.units) < 5:
        raise ValueError("Unexpected Wan pipeline unit layout")
    pipe.units[3] = WanVideoUnit_InputRGBDRasters()
    pipe.units[4] = WanVideoUnit_ImageEmbedderVAE_RGBDIndividualEncoding()
    pipe.model_fn = model_fn_wan_video_rgbd


def model_fn_wan_video_rgbd(
    dit: WanModel,
    latents: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    raymap: torch.Tensor | None = None,
    modality_ids: torch.Tensor | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs: Any,
):
    """Wan forward with an explicit RGB/depth token embedding."""
    if not hasattr(dit, "modality_embedding"):
        raise AttributeError("RGB-D DiT is missing modality_embedding")
    if modality_ids is None:
        modality_ids = getattr(dit, "rgbd_inference_modality_ids", None)
    if modality_ids is None:
        raise ValueError("RGB-D generation requires one modality id per raster slot")

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.cat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.cat([timestep] * context.shape[0], dim=0)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)
    if raymap is not None:
        if raymap.dim() == 4:
            raymap = raymap.permute(1, 0, 2, 3).unsqueeze(0)
        x = torch.cat([x, raymap], dim=1)

    x = dit.patchify(x)
    frame_count, patch_height, patch_width = x.shape[2:]
    modality_ids = modality_ids.to(device=x.device, dtype=torch.long)
    if modality_ids.ndim == 1:
        modality_ids = modality_ids.unsqueeze(0)
    if modality_ids.shape[-1] != frame_count:
        raise ValueError(
            f"Expected {frame_count} modality ids, got {modality_ids.shape[-1]}"
        )
    if modality_ids.shape[0] == 1 and x.shape[0] != 1:
        modality_ids = modality_ids.expand(x.shape[0], -1)
    if modality_ids.shape[0] != x.shape[0]:
        raise ValueError(
            f"Modality batch {modality_ids.shape[0]} does not match latent batch "
            f"{x.shape[0]}"
        )
    modality = F.embedding(
        modality_ids,
        dit.modality_embedding.weight.to(device=x.device, dtype=x.dtype),
    )
    modality = modality.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    x = x + modality.to(dtype=x.dtype)
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    frame_freqs = torch.ones_like(dit.freqs[0][:frame_count])
    freqs = torch.cat(
        [
            frame_freqs.view(frame_count, 1, 1, -1).expand(
                frame_count, patch_height, patch_width, -1
            ),
            dit.freqs[1][:patch_height]
            .view(1, patch_height, 1, -1)
            .expand(frame_count, patch_height, patch_width, -1),
            dit.freqs[2][:patch_width]
            .view(1, 1, patch_width, -1)
            .expand(frame_count, patch_height, patch_width, -1),
        ],
        dim=-1,
    ).reshape(frame_count * patch_height * patch_width, 1, -1).to(x.device)

    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs, use_reentrant=False
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block, x, context, t_mod, freqs, use_reentrant=False
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (frame_count, patch_height, patch_width))
