"""UniRoPE: ray-based relative rotary encoding, independent of the camera model.

A token is one ray: its camera center ``c`` and unit direction ``u``, plus a
point ``p = c + d·u`` at a predicted depth ``d``. Only the ray-generation
function knows the camera model (:func:`pinhole_directions`;
:func:`equirect_directions` for 360° panoramas); everything after it sees rays.

For each query camera, every token's ``c``, ``u`` and ``p`` are moved into that
camera's frame with its SE(3) pose only (no intrinsics, no projection), and
each of the 9 coordinates gets RoPE frequencies. The direction is the full unit
vector, so points behind a camera and wide or 360° views stay distinct. Depth
is an interval ``exp(log d ∓ σ)`` from a per-block depth head, as in RayRoPE;
``p`` is linear in ``d``, so the mean of cos/sin over the interval is exact.

Like RayRoPE, attention runs once per query camera, which makes the scores
exactly invariant to the world frame. :func:`unirope_anchor_attention` is the
cheaper variant: every token is encoded once in camera 0's frame (as SCoPE
anchors its rays), so queries and keys share one encoding and attention runs
once; scores then depend on the anchor camera's orientation. Poses are normalized as in RayRoPE
(camera 0 is the identity, camera 1 at unit distance). Head layout: UniRoPE
takes the whole 128-d head, with no backbone RoPE axis kept (images differ by
their camera center): 9 coordinates × 7 frequencies × 2 = 126 channels are
rotated and the last 2 pass through.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from .rayrope import (
    HEAD_DIM,
    _angles,
    _freqs,
    _mix,
    _trig_interval,
    depth_interval,
    per_camera_attention,
    prepare_geometry,
)

NUM_COORDS = 9  # center (3) + direction (3) + point at depth (3)
NUM_FREQS = HEAD_DIM // (2 * NUM_COORDS)  # 7: 126 rotated channels, 2 pass through
# Longest periods, as RayRoPE's "3d" points (4) and unit directions (8).
CENTER_PERIOD = 4.0
DIRECTION_PERIOD = 8.0
POINT_PERIOD = 4.0
# Longest / shortest period = 81 (5 frequencies at RayRoPE's base 3); the 7
# frequencies fill that range more densely instead of reaching higher.
FREQ_BASE = 81.0 ** (1.0 / (NUM_FREQS - 1))


def pinhole_directions(
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    patch: int = 16,
) -> torch.Tensor:
    """Camera-frame unit ray through each patch center, ``[B, N, h*w, 3]`` (OpenCV).

    The image is assumed center-cropped to a multiple of ``patch`` before
    encoding, so principal points are shifted to match (as ``plucker_tokens``).
    """
    height, width = image_hw
    patches_y, patches_x = latent_hw
    cropped_h, cropped_w = patches_y * patch, patches_x * patch
    top = (height - cropped_h) // 2
    left = (width - cropped_w) // 2
    fx, fy = fxfycxcy[..., 0].float(), fxfycxcy[..., 1].float()
    cx, cy = fxfycxcy[..., 2].float() - left, fxfycxcy[..., 3].float() - top

    device = fxfycxcy.device
    ys = (torch.arange(patches_y, device=device) + 0.5) * patch
    xs = (torch.arange(patches_x, device=device) + 0.5) * patch
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (grid_x.flatten() - cx[..., None]) / fx[..., None]
    y = (grid_y.flatten() - cy[..., None]) / fy[..., None]
    return F.normalize(torch.stack((x, y, torch.ones_like(x)), dim=-1), dim=-1)


def equirect_directions(batch: int, cameras: int, latent_hw: tuple[int, int], device) -> torch.Tensor:
    """Camera-frame unit ray through each patch center of an equirectangular
    panorama, ``[B, N, h*w, 3]`` (OpenCV: x right, y down, z = image center).
    Longitude runs -π..π left to right, latitude π/2..-π/2 top to bottom."""
    patches_y, patches_x = latent_hw
    lon = ((torch.arange(patches_x, device=device) + 0.5) / patches_x * 2.0 - 1.0) * math.pi
    lat = (0.5 - (torch.arange(patches_y, device=device) + 0.5) / patches_y) * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    directions = torch.stack(
        (torch.cos(lat) * torch.sin(lon), -torch.sin(lat), torch.cos(lat) * torch.cos(lon)), dim=-1
    )
    return directions.reshape(1, 1, -1, 3).expand(batch, cameras, -1, -1)


def prepare_unirope(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
) -> dict:
    """Normalized poses (RayRoPE's ``prepare_geometry``) plus pinhole ray directions.

    A different camera model only changes ``directions``.
    """
    geom = prepare_geometry(c2w, fxfycxcy, image_hw, latent_hw)
    geom["directions"] = pinhole_directions(fxfycxcy, image_hw, latent_hw)
    return geom


def _pack_cos_sin(
    centers: torch.Tensor, directions: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``centers`` / ``directions`` ``[B, N, 3]``, ``points`` ``[2, B, N, 3]`` (the
    two ends of the depth interval) → cos/sin ``[B, N, 63]``."""
    device = centers.device
    center_angle = _angles(centers, _freqs(NUM_FREQS, CENTER_PERIOD, device, FREQ_BASE))
    direction_angle = _angles(directions, _freqs(NUM_FREQS, DIRECTION_PERIOD, device, FREQ_BASE))
    point_freqs = _freqs(NUM_FREQS, POINT_PERIOD, device, FREQ_BASE)
    cos_point, sin_point = _trig_interval(_angles(points[0], point_freqs), _angles(points[1], point_freqs))
    cosine = torch.cat((torch.cos(center_angle), torch.cos(direction_angle), cos_point), dim=-1)
    sine = torch.cat((torch.sin(center_angle), torch.sin(direction_angle), sin_point), dim=-1)
    return cosine.flatten(-2, -1), sine.flatten(-2, -1)


@torch.compile
def _coefficients(
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    directions: torch.Tensor,
    raw_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE coefficients of one block.

    ``c2w`` / ``w2c`` ``[B, C, 4, 4]`` normalized poses, ``directions``
    ``[B, C, P, 3]`` camera-frame rays, ``raw_depth`` ``[B, C*P, 2]``. Returns
    query ``cos, sin`` ``[B, img, 63]`` (each token in its own camera) and key
    ``cos, sin`` ``[C, B, img, 63]`` (all tokens, per query camera).
    """
    batch, cameras, patches, _ = directions.shape
    img_len = cameras * patches
    depths = depth_interval(raw_depth.float()).reshape(2, batch, cameras, patches, 1)
    centers_world = c2w[:, :, :3, 3]                                            # [B, C, 3]
    directions_world = torch.einsum("bcij,bcpj->bcpi", c2w[:, :, :3, :3], directions.float())

    cos_k, sin_k, query = [], [], []
    for camera_index in range(cameras):
        rotation = w2c[:, camera_index, :3, :3]
        translation = w2c[:, camera_index, :3, 3]
        centers = torch.einsum("bij,bcj->bci", rotation, centers_world) + translation[:, None]
        centers = centers[:, :, None].expand(-1, -1, patches, -1)               # [B, C, P, 3]
        rays = torch.einsum("bij,bcpj->bcpi", rotation, directions_world)       # [B, C, P, 3]
        points = centers + depths * rays                                        # [2, B, C, P, 3]
        cosine, sine = _pack_cos_sin(
            centers.reshape(batch, img_len, 3), rays.reshape(batch, img_len, 3),
            points.reshape(2, batch, img_len, 3),
        )
        cos_k.append(cosine)
        sin_k.append(sine)
        query.append((centers[:, camera_index], rays[:, camera_index], points[:, :, camera_index]))
    cos_q, sin_q = _pack_cos_sin(
        torch.stack([c for c, _, _ in query], dim=1).reshape(batch, img_len, 3),
        torch.stack([u for _, u, _ in query], dim=1).reshape(batch, img_len, 3),
        torch.stack([p for _, _, p in query], dim=2).reshape(2, batch, img_len, 3),
    )
    return cos_q, sin_q, torch.stack(cos_k), torch.stack(sin_k)


def unirope_attention(
    q_content: torch.Tensor,
    k_content: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    num_txt: int,
    raw_depth: torch.Tensor,
    geom: dict,
    q_extra: Optional[torch.Tensor] = None,
    k_extra: Optional[torch.Tensor] = None,
    rope_axis: int = 0,
) -> torch.Tensor:
    """Full attention with per-camera UniRoPE on image queries.

    Same arguments as :func:`rayrope.rayrope_attention`; ``geom`` comes from
    :func:`prepare_unirope`. ``rope_axis`` is accepted for the shared call
    signature and unused: UniRoPE replaces the whole head on image tokens.
    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    dim = q_content.shape[-1]
    if dim != HEAD_DIM:
        raise RuntimeError(f"UniRoPE expects head dim {HEAD_DIM}, got {dim}")
    batch, cameras, patches, _ = geom["directions"].shape
    if q_content.shape[2] - num_txt != cameras * patches:
        raise RuntimeError(
            f"UniRoPE image tokens {q_content.shape[2] - num_txt} != {cameras} cameras × {patches} patches"
        )
    if raw_depth.shape[1] != cameras * patches:
        raise RuntimeError(f"Depth head length {raw_depth.shape[1]} != image tokens {cameras * patches}")
    # Geometry in fp32 (see rayrope_attention).
    with torch.autocast(device_type=q_content.device.type, enabled=False):
        cos_q, sin_q, cos_k, sin_k = _coefficients(geom["c2w"], geom["w2c"], geom["directions"], raw_depth)
    return per_camera_attention(
        q_content, k_content, v, q_rope, k_rope, num_txt, cos_q, sin_q, cos_k, sin_k,
        patches, q_extra, k_extra, rope_axis=None,
    )


@torch.compile(dynamic=False)
def _anchor_coefficients(
    c2w: torch.Tensor, directions: torch.Tensor, raw_depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE coefficients ``[B, img, 63]`` of every token in camera 0's frame
    (the frame of the normalized poses). ``dynamic=False``: FLUX 3 calls it with
    2, 3 and 5 cameras, and an automatic dynamic-shape recompile saved different
    tensors for backward than the static graph, breaking gradient-checkpoint
    recomputation. One static graph per camera count avoids that."""
    batch, cameras, patches, _ = directions.shape
    img_len = cameras * patches
    depths = depth_interval(raw_depth.float()).reshape(2, batch, cameras, patches, 1)
    centers = c2w[:, :, None, :3, 3].expand(-1, -1, patches, -1)               # [B, C, P, 3]
    rays = torch.einsum("bcij,bcpj->bcpi", c2w[:, :, :3, :3], directions.float())
    points = centers + depths * rays                                            # [2, B, C, P, 3]
    return _pack_cos_sin(
        centers.reshape(batch, img_len, 3), rays.reshape(batch, img_len, 3), points.reshape(2, batch, img_len, 3)
    )


def unirope_anchor_attention(
    q_content: torch.Tensor,
    k_content: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    num_txt: int,
    raw_depth: torch.Tensor,
    geom: dict,
    q_extra: Optional[torch.Tensor] = None,
    k_extra: Optional[torch.Tensor] = None,
    rope_axis: int = 0,
) -> torch.Tensor:
    """UniRoPE with every token placed once in camera 0's frame: one attention pass.

    Same arguments as :func:`unirope_attention` (``rope_axis`` unused).
    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    dim = q_content.shape[-1]
    if dim != HEAD_DIM:
        raise RuntimeError(f"UniRoPE expects head dim {HEAD_DIM}, got {dim}")
    batch, cameras, patches, _ = geom["directions"].shape
    if q_content.shape[2] - num_txt != cameras * patches or raw_depth.shape[1] != cameras * patches:
        raise RuntimeError(
            f"UniRoPE image tokens {q_content.shape[2] - num_txt} / depth {raw_depth.shape[1]} != "
            f"{cameras} cameras × {patches} patches"
        )
    with torch.autocast(device_type=q_content.device.type, enabled=False):
        cosine, sine = _anchor_coefficients(geom["c2w"], geom["directions"], raw_depth)
    q_img = _mix(q_rope[:, :, num_txt:], q_content[:, :, num_txt:], cosine, sine, None)
    k_img = _mix(k_rope[:, :, num_txt:], k_content[:, :, num_txt:], cosine, sine, None)
    if q_extra is not None:
        q_img = q_img + q_extra[:, :, num_txt:]
    if k_extra is not None:
        k_img = k_img + k_extra[:, :, num_txt:]
    q = torch.cat((q_rope[:, :, :num_txt], q_img), dim=2)
    k = torch.cat((k_rope[:, :, :num_txt], k_img), dim=2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return rearrange(out, "b h s d -> b s (h d)")
