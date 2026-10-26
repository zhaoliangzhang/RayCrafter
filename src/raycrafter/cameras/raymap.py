"""Plücker ray map on the input channels (FrameCrafter / OmniCrafter conditioning).

Every pixel gets the Plücker ray ``(d, o × d)`` through its center. The
``patch × patch`` pixels of a token are pixel-unshuffled into one
``6 · patch²`` vector (1536 for the 16-px FLUX token), and a zero-initialized
linear layer adds its projection to the token's input embedding. That is the
same as concatenating the ray map to the latent channels in front of the input
projection, so step 0 is the pretrained backbone. FrameCrafter gives Wan the
same rays: pixel-unshuffled by the VAE's 8, then seen by its 2×2 patch embed,
1536 values per token.

Poses use FrameCrafter's gauge (``normalize_w2c_make_cam_last_origin``): the
last camera (the last view) is the identity and translations are divided by
the mean camera distance to it, so the loaders' near-depth unit cancels and
the order of the refs does not matter. Ray directions come from UniRoPE's
generators (pinhole or equirect, the convention OmniCrafter's ray maps use).

This file is backbone-free. ``backbones/flux3.py`` adds the projection.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from .unirope import equirect_directions, pinhole_directions

# Below this mean camera distance a set is a pure rotation: its translations
# are float noise, so they are zeroed instead of scaled up to unit length.
MIN_BASELINE = 1e-6


def raymap_channels(patch: int = 16) -> int:
    """Ray-map values per token: 6 Plücker components per pixel."""
    return 6 * patch * patch


def plucker_raymap(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    patch: int = 16,
    camera_model: str = "pinhole",
) -> torch.Tensor:
    """Pixel-unshuffled Plücker rays, one ``6·patch²`` vector per packed token.

    Args:
        c2w: ``[B, N, 4, 4]`` camera-to-world (OpenCV), ordered as the tokens
            (``[ref..., views...]``).
        fxfycxcy: ``[B, N, 4]`` pinhole intrinsics in pixels of the full image
            (unused for ``equirect``).
        image_hw: ``(H, W)`` of the image given to the VAE. It is assumed
            center-cropped to whole patches, as ``plucker_tokens``.
        latent_hw: ``(h, w)`` token grid.
        camera_model: ``pinhole`` or ``equirect`` (a full 2:1 panorama).

    Returns:
        ``[B, N*h*w, 6*patch²]`` in token order (each image row-major).
        Channel ``c·patch² + dy·patch + dx`` is component ``c`` (direction xyz,
        then moment xyz) of pixel ``(dy, dx)`` of the patch: ``PixelUnshuffle``.
    """
    height, width = image_hw
    patches_y, patches_x = latent_hw
    pixels = (patches_y * patch, patches_x * patch)
    batch, cameras = c2w.shape[:2]

    relative = torch.linalg.inv(c2w[:, -1:].double()) @ c2w.double()
    centers = relative[:, :, :3, 3]
    scale = centers.norm(dim=-1).mean(dim=1)[:, None, None]
    centers = torch.where(scale > MIN_BASELINE, centers / scale.clamp_min(MIN_BASELINE), torch.zeros_like(centers))

    if camera_model == "equirect":
        if pixels != (height, width):
            raise ValueError(f"equirect image {height}x{width} is not a whole number of {patch}px patches")
        directions = equirect_directions(batch, cameras, pixels, c2w.device)
    elif camera_model == "pinhole":
        directions = pinhole_directions(fxfycxcy, image_hw, pixels, patch=1)
    else:
        raise ValueError(f"unknown camera_model {camera_model!r}")

    # [B, N, H*W, 3]: camera-frame pixel rays into the last camera's frame.
    directions = torch.einsum("bnij,bnpj->bnpi", relative[:, :, :3, :3].float(), directions[..., 0, :].float())
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    moments = torch.cross(centers[:, :, None].float().expand_as(directions), directions, dim=-1)
    rays = rearrange(torch.cat((directions, moments), dim=-1), "b n (hh ww) c -> (b n) c hh ww", hh=pixels[0])
    rays = F.pixel_unshuffle(rays, patch)
    return rearrange(rays, "(b n) c hh ww -> b (n hh ww) c", b=batch)
