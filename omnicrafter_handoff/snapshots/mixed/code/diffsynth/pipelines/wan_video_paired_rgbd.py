"""Five-slot paired RGB/depth components for the Wan video pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..diffusion.base_pipeline import PipelineUnit
from .wan_video import model_fn_wan_video
from .wan_video_rgbd import _raster_to_bchw


PAIRED_LATENT_CHANNELS = 32
PAIRED_CONDITION_CHANNELS = 40
PAIRED_DIT_INPUT_CHANNELS = 456


def _encode_raster(pipe, raster, height, width, tiled, tile_size, tile_stride):
    raster = _raster_to_bchw(pipe, raster, height, width)
    return pipe.vae.encode(
        raster.unsqueeze(2),
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    ).to(dtype=pipe.torch_dtype, device=pipe.device)


def _strict_latent_mask(mask, count, height, width, latent_height, latent_width, pipe):
    if mask is None:
        return torch.ones(
            count,
            latent_height,
            latent_width,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != (count, height, width):
        raise ValueError(
            f"Expected validity masks {(count, height, width)}, got {tuple(mask.shape)}"
        )
    mask = F.interpolate(
        mask.float().unsqueeze(1),
        size=(latent_height, latent_width),
        mode="area",
    )[:, 0]
    return (mask >= 1.0 - 1e-6).to(
        device=pipe.device, dtype=pipe.torch_dtype
    )


class WanVideoUnit_PairedRGBDNoiseInitializer(PipelineUnit):
    """Allocate one 32-channel latent for every physical camera slot."""

    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
                "num_latent_frames",
            ),
            output_params=("noise",),
        )

    def process(
        self,
        pipe,
        height,
        width,
        num_frames,
        seed,
        rand_device,
        num_latent_frames=None,
    ):
        length = num_frames if num_latent_frames is None else num_latent_frames
        shape = (
            1,
            PAIRED_LATENT_CHANNELS,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnit_InputPairedRGBDTargets(PipelineUnit):
    """Encode RGB/depth separately, then pair them along the channel axis."""

    def __init__(self):
        super().__init__(
            input_params=(
                "target_rgb_rasters",
                "target_depth_rasters",
                "noise",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        target_rgb_rasters,
        target_depth_rasters,
        noise,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if target_rgb_rasters is None or target_depth_rasters is None:
            return {"latents": noise}
        if len(target_rgb_rasters) != len(target_depth_rasters):
            raise ValueError("RGB and depth target slot counts differ")
        if len(target_rgb_rasters) != noise.shape[2]:
            raise ValueError(
                f"Expected {noise.shape[2]} paired targets, got "
                f"{len(target_rgb_rasters)}"
            )
        pipe.load_models_to_device(self.onload_model_names)
        paired = []
        for rgb, depth in zip(target_rgb_rasters, target_depth_rasters):
            rgb_latent = _encode_raster(
                pipe, rgb, height, width, tiled, tile_size, tile_stride
            )
            depth_latent = _encode_raster(
                pipe, depth, height, width, tiled, tile_size, tile_stride
            )
            paired.append(torch.cat([rgb_latent, depth_latent], dim=1))
        input_latents = torch.cat(paired, dim=2)
        if pipe.scheduler.training:
            return {
                "latents": noise,
                "input_latents": input_latents,
                "mask_loss": True,
            }
        return {
            "latents": pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
        }


class WanVideoUnit_PairedRGBDCondition(PipelineUnit):
    """Build role, paired VAE condition, and depth-presence channels.

    Channel order is ``role[4] | RGB latent[16] | depth latent[16] |
    depth-present[4]``.  Query slots carry a noisy RGB latent and an explicit
    zero/missing depth half.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "reference_rgb",
                "reference_depth",
                "query_rgb_conditioning",
                "reference_rgb_validity",
                "reference_depth_validity",
                "height",
                "width",
                "num_latent_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        reference_rgb,
        reference_depth,
        query_rgb_conditioning,
        reference_rgb_validity,
        reference_depth_validity,
        height,
        width,
        num_latent_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if reference_rgb is None:
            return {}
        num_references = len(reference_rgb)
        num_queries = len(query_rgb_conditioning)
        if len(reference_depth) != num_references:
            raise ValueError("Reference RGB/depth counts differ")
        if num_latent_frames != num_references + num_queries:
            raise ValueError(
                f"Expected {num_latent_frames} physical conditions, got "
                f"{num_references + num_queries}"
            )
        pipe.load_models_to_device(self.onload_model_names)

        latent_height = height // pipe.vae.upsampling_factor
        latent_width = width // pipe.vae.upsampling_factor
        rgb_valid = _strict_latent_mask(
            reference_rgb_validity,
            num_references,
            height,
            width,
            latent_height,
            latent_width,
            pipe,
        )
        depth_valid = _strict_latent_mask(
            reference_depth_validity,
            num_references,
            height,
            width,
            latent_height,
            latent_width,
            pipe,
        )

        paired_conditions = []
        for rgb, depth in zip(reference_rgb, reference_depth):
            rgb_latent = _encode_raster(
                pipe, rgb, height, width, tiled, tile_size, tile_stride
            )
            depth_latent = _encode_raster(
                pipe, depth, height, width, tiled, tile_size, tile_stride
            )
            paired_conditions.append(torch.cat([rgb_latent, depth_latent], dim=1))
        for rgb in query_rgb_conditioning:
            rgb_latent = _encode_raster(
                pipe, rgb, height, width, tiled, tile_size, tile_stride
            )
            paired_conditions.append(torch.cat([rgb_latent, torch.zeros_like(rgb_latent)], dim=1))
        condition = torch.cat(paired_conditions, dim=2)

        role = torch.zeros(
            num_latent_frames,
            latent_height,
            latent_width,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        role[:num_references] = rgb_valid
        depth_presence = torch.zeros_like(role)
        depth_presence[:num_references] = depth_valid
        role = role.unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0)
        depth_presence = (
            depth_presence.unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0)
        )
        y = torch.cat([role, condition, depth_presence], dim=1)
        if y.shape[1] != PAIRED_CONDITION_CHANNELS:
            raise RuntimeError(f"Unexpected paired condition shape: {tuple(y.shape)}")
        return {"y": y.to(dtype=pipe.torch_dtype, device=pipe.device)}


def install_paired_rgbd_pipeline_components(pipe) -> None:
    """Install paired targets/conditioning while retaining Wan's DiT forward."""
    if len(pipe.units) < 5:
        raise ValueError("Unexpected Wan pipeline unit layout")
    pipe.units[1] = WanVideoUnit_PairedRGBDNoiseInitializer()
    pipe.units[3] = WanVideoUnit_InputPairedRGBDTargets()
    pipe.units[4] = WanVideoUnit_PairedRGBDCondition()
    pipe.model_fn = model_fn_wan_video
    pipe.dit.individual_encoding = True


@torch.no_grad()
def generate_paired_rgbd_latents(
    pipe,
    reference_rgb,
    reference_depth,
    query_rgb_conditioning,
    reference_rgb_validity,
    reference_depth_validity,
    raymap,
    height=256,
    width=512,
    num_inference_steps=50,
    sigma_shift=5.0,
    seed=42,
    cfg_scale=1.0,
    prompt="",
    negative_prompt="",
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    progress_bar_cmd=tqdm,
):
    """Denoise five paired slots and return their 32-channel latents.

    Decoding is intentionally left to the caller because each slot must be
    split into 16 RGB and 16 depth channels before invoking the stock VAE.
    """
    pipe.scheduler.set_timesteps(
        num_inference_steps, denoising_strength=1.0, shift=sigma_shift
    )
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {
        "reference_rgb": reference_rgb,
        "reference_depth": reference_depth,
        "query_rgb_conditioning": query_rgb_conditioning,
        "reference_rgb_validity": reference_rgb_validity,
        "reference_depth_validity": reference_depth_validity,
        "target_rgb_rasters": None,
        "target_depth_rasters": None,
        "raymap": raymap,
        "height": height,
        "width": width,
        "num_frames": 5,
        "num_latent_frames": 5,
        "seed": seed,
        "rand_device": "cpu",
        "cfg_scale": cfg_scale,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        prediction_posi = pipe.model_fn(
            **models, **inputs_shared, **inputs_posi, timestep=timestep
        )
        if cfg_scale != 1.0:
            prediction_nega = pipe.model_fn(
                **models, **inputs_shared, **inputs_nega, timestep=timestep
            )
            prediction = prediction_nega + cfg_scale * (
                prediction_posi - prediction_nega
            )
        else:
            prediction = prediction_posi
        inputs_shared["latents"] = pipe.scheduler.step(
            prediction,
            pipe.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )
    return inputs_shared["latents"]
