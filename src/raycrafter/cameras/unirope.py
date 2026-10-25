"""UniRoPE: ray-based relative rotary encoding, independent of the camera model.

A token is one ray: its camera center ``c`` and unit direction ``u``, plus a
point ``p = c + d·u`` at a predicted depth ``d``. Only the ray-generation
function knows the camera model (:func:`pinhole_directions`;
:func:`equirect_directions` for 360° panoramas); everything after it sees rays.

For each query camera, every token's ``c``, ``u`` and ``p`` are moved into that
camera's frame with its SE(3) pose only (no intrinsics, no projection). ``p`` is
encoded as RayRoPE encodes its points, as where it is seen from the query
camera and how far: the unit direction ``p / |p|`` and the log distance
``log |p|`` (clipped to the depth head's range). Directions are full unit
vectors, so points behind a camera and wide or 360° views stay distinct.

Depth is an interval ``exp(log d ∓ σ)`` from a per-block depth head, as in
RayRoPE, and cos/sin are averaged between the two ends. Each coordinate's
longest period is 4× its range (RayRoPE's rule: 4 for camera centers, 8 for
unit vectors, ``4 · 2 · log(MAX_DEPTH)`` ≈ 37 for log distances), so an
unknown depth only blurs the fine frequencies. Encoding ``p`` as raw xyz with
a camera-center period instead (the first version) wiped every point channel
at the initial σ = 3.

Like RayRoPE, attention runs once per query camera, which makes the scores
exactly invariant to the world frame. :func:`unirope_anchor_attention` is the
cheaper variant: every token is encoded once in camera 0's frame (as SCoPE
anchors its rays), so queries and keys share one encoding and attention runs
once; scores then depend on the anchor camera's orientation. Poses are normalized as in RayRoPE
(camera 0 is the identity, camera 1 at unit distance).

Footprint (``footprint=True``): a token is the bundle of rays through its patch,
not just the center ray. The ray generator also returns rays through three
patch corners (as RayRoPE's three rays), and the depth point ``p`` becomes the
three corner points at depth ``d`` along their rays, so the encoding sees the
patch's size, in-plane rotation and lens distortion (a panorama's poles, a
fisheye's edge) as a surface patch in 3D.

Head layout: UniRoPE takes the whole 128-d head, with no backbone RoPE axis
kept (images differ by their camera center). Center ray: c, u, the point's
direction and log distance are 10 coordinates × 6 frequencies × 2 = 120
channels rotated, 8 pass through. Footprint: c and each corner point's
direction and log distance are 15 coordinates × 4 frequencies × 2 = 120, 8 pass
through; the ray direction u is left out (the corner points carry it). Either
way the frequencies span a period ratio of 81.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from .rayrope import (
    HEAD_DIM,
    MAX_DEPTH,
    _angles,
    _freqs,
    _mix,
    _rotate_vo,
    _trig_interval,
    depth_interval,
    per_camera_attention,
    prepare_geometry,
)

# Longest periods: 4× each coordinate's range, as in RayRoPE (camera centers
# about ±0.5 after pose normalization; unit vectors; log distances clipped to
# the depth head's ±log(MAX_DEPTH)).
CENTER_PERIOD = 4.0
DIRECTION_PERIOD = 8.0
LOG_DISTANCE_MAX = math.log(MAX_DEPTH)
DISTANCE_PERIOD = 4.0 * 2.0 * LOG_DISTANCE_MAX
# Longest / shortest period (5 frequencies at RayRoPE's base 3). However many
# frequencies fit the head, they fill this range instead of reaching higher.
PERIOD_RATIO = 81.0
# Pixel offsets in a patch, in patches: the center, then RayRoPE's three corners.
CENTER = (0.5, 0.5)
CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _offsets(footprint: bool) -> tuple[tuple[float, float], ...]:
    return (CENTER, *CORNERS) if footprint else (CENTER,)


def pinhole_directions(
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    footprint: bool = False,
    patch: int = 16,
) -> torch.Tensor:
    """Camera-frame unit rays per patch, ``[B, N, h*w, R, 3]`` (OpenCV): the center
    ray, plus the three corner rays when ``footprint`` (R = 1 or 4).

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
    rays = []
    for ox, oy in _offsets(footprint):
        ys = (torch.arange(patches_y, device=device) + oy) * patch
        xs = (torch.arange(patches_x, device=device) + ox) * patch
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x = (grid_x.flatten() - cx[..., None]) / fx[..., None]
        y = (grid_y.flatten() - cy[..., None]) / fy[..., None]
        rays.append(F.normalize(torch.stack((x, y, torch.ones_like(x)), dim=-1), dim=-1))
    return torch.stack(rays, dim=-2)


def equirect_directions(
    batch: int, cameras: int, latent_hw: tuple[int, int], device, footprint: bool = False
) -> torch.Tensor:
    """Camera-frame unit rays per patch of an equirectangular panorama,
    ``[B, N, h*w, R, 3]`` (OpenCV: x right, y down, z = image center), center
    first, then the three corners when ``footprint``. Longitude runs -π..π left
    to right, latitude π/2..-π/2 top to bottom."""
    patches_y, patches_x = latent_hw
    rays = []
    for ox, oy in _offsets(footprint):
        lon = ((torch.arange(patches_x, device=device) + ox) / patches_x * 2.0 - 1.0) * math.pi
        lat = (0.5 - (torch.arange(patches_y, device=device) + oy) / patches_y) * math.pi
        lat, lon = torch.meshgrid(lat, lon, indexing="ij")
        rays.append(torch.stack(
            (torch.cos(lat) * torch.sin(lon), -torch.sin(lat), torch.cos(lat) * torch.cos(lon)), dim=-1
        ).reshape(-1, 3))
    return torch.stack(rays, dim=-2).expand(batch, cameras, -1, -1, -1)


def prepare_unirope(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    footprint: bool = False,
    camera_model: str = "pinhole",
) -> dict:
    """Normalized poses (RayRoPE's ``prepare_geometry``) plus camera-frame rays
    ``[B, C, P, R, 3]`` from ``camera_model``'s generator (``fxfycxcy`` is unused
    for ``equirect``). The camera model only changes ``directions``.
    """
    geom = prepare_geometry(c2w, fxfycxcy, image_hw, latent_hw, camera_model=camera_model)
    if camera_model == "equirect":
        batch, cameras = c2w.shape[:2]
        geom["directions"] = equirect_directions(batch, cameras, latent_hw, c2w.device, footprint)
    else:
        geom["directions"] = pinhole_directions(fxfycxcy, image_hw, latent_hw, footprint)
    return geom


def _pack_cos_sin(
    centers: torch.Tensor, directions: Optional[torch.Tensor], point_dirs: torch.Tensor, distance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``centers`` ``[B, N, 3]``; ``directions`` ``[B, N, 3]`` or ``None`` (left
    out); ``point_dirs`` ``[2, B, N, 3K]`` and ``distance`` ``[2, B, N, K]`` at
    the two ends of the depth interval (K points per token) → cos/sin
    ``[B, N, F * D]`` with D = 3 (+ 3) + 4K coordinates: as many frequencies F as
    fit the head."""
    device = centers.device
    num_coords = 3 + (0 if directions is None else 3) + point_dirs.shape[-1] + distance.shape[-1]
    num_freqs = HEAD_DIM // (2 * num_coords)
    base = PERIOD_RATIO ** (1.0 / (num_freqs - 1))
    center_angle = _angles(centers, _freqs(num_freqs, CENTER_PERIOD, device, base))
    direction_freqs = _freqs(num_freqs, DIRECTION_PERIOD, device, base)
    cos_parts, sin_parts = [torch.cos(center_angle)], [torch.sin(center_angle)]
    if directions is not None:
        direction_angle = _angles(directions, direction_freqs)
        cos_parts.append(torch.cos(direction_angle))
        sin_parts.append(torch.sin(direction_angle))
    for coords, freqs in ((point_dirs, direction_freqs), (distance, _freqs(num_freqs, DISTANCE_PERIOD, device, base))):
        cosine, sine = _trig_interval(_angles(coords[0], freqs), _angles(coords[1], freqs))
        cos_parts.append(cosine)
        sin_parts.append(sine)
    return torch.cat(cos_parts, dim=-1).flatten(-2, -1), torch.cat(sin_parts, dim=-1).flatten(-2, -1)


def _token_geometry(rotation, translation, centers_world, directions_world, depths):
    """Every token's center, center ray and depth points in one camera's frame.

    ``directions_world`` is ``[B, C, P, R, 3]``: with R = 1 the point is the
    center ray at depth, with R = 4 the three corner rays are. Returns centers
    ``[B, C, P, 3]``, center rays ``[B, C, P, 3]`` (``None`` for the
    footprint, whose corner points carry them), the points' unit directions
    from this camera ``[2, B, C, P, 3K]`` and their log distances
    ``[2, B, C, P, K]``.
    """
    patches, rays_per = directions_world.shape[2], directions_world.shape[3]
    centers = torch.einsum("bij,bcj->bci", rotation, centers_world) + translation[:, None]
    centers = centers[:, :, None].expand(-1, -1, patches, -1)                   # [B, C, P, 3]
    rays = torch.einsum("bij,bcprj->bcpri", rotation, directions_world)         # [B, C, P, R, 3]
    point_rays = rays if rays_per == 1 else rays[:, :, :, 1:]
    points = centers[:, :, :, None] + depths * point_rays                       # [2, B, C, P, K, 3]
    norm = points.norm(dim=-1, keepdim=True)
    point_dirs = points / norm.clamp(min=1e-6)
    distance = norm.clamp(min=1e-6).log().clamp(-LOG_DISTANCE_MAX, LOG_DISTANCE_MAX)
    center_rays = rays[:, :, :, 0] if rays_per == 1 else None
    return centers, center_rays, point_dirs.flatten(-2, -1), distance.flatten(-2, -1)


@torch.compile
def _coefficients(
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    directions: torch.Tensor,
    raw_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE coefficients of one block.

    ``c2w`` / ``w2c`` ``[B, C, 4, 4]`` normalized poses, ``directions``
    ``[B, C, P, R, 3]`` camera-frame rays, ``raw_depth`` ``[B, C*P, 2]``. Returns
    query ``cos, sin`` ``[B, img, half]`` (each token in its own camera) and key
    ``cos, sin`` ``[C, B, img, half]`` (all tokens, per query camera).
    """
    batch, cameras, patches = directions.shape[:3]
    img_len = cameras * patches
    depths = depth_interval(raw_depth.float()).reshape(2, batch, cameras, patches, 1, 1)
    centers_world = c2w[:, :, :3, 3]                                            # [B, C, 3]
    directions_world = torch.einsum("bcij,bcprj->bcpri", c2w[:, :, :3, :3], directions.float())

    cos_k, sin_k, query = [], [], []
    for camera_index in range(cameras):
        centers, rays, point_dirs, distance = _token_geometry(
            w2c[:, camera_index, :3, :3], w2c[:, camera_index, :3, 3], centers_world, directions_world, depths
        )
        cosine, sine = _pack_cos_sin(
            centers.reshape(batch, img_len, 3), None if rays is None else rays.reshape(batch, img_len, 3),
            point_dirs.reshape(2, batch, img_len, -1), distance.reshape(2, batch, img_len, -1),
        )
        cos_k.append(cosine)
        sin_k.append(sine)
        query.append((centers[:, camera_index], None if rays is None else rays[:, camera_index],
                      point_dirs[:, :, camera_index], distance[:, :, camera_index]))
    cos_q, sin_q = _pack_cos_sin(
        torch.stack([q[0] for q in query], dim=1).reshape(batch, img_len, 3),
        None if rays is None else torch.stack([q[1] for q in query], dim=1).reshape(batch, img_len, 3),
        torch.stack([q[2] for q in query], dim=2).reshape(2, batch, img_len, -1),
        torch.stack([q[3] for q in query], dim=2).reshape(2, batch, img_len, -1),
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
    apply_vo: bool = False,
) -> torch.Tensor:
    """Full attention with per-camera UniRoPE on image queries.

    Same arguments as :func:`rayrope.rayrope_attention`; ``geom`` comes from
    :func:`prepare_unirope`. ``rope_axis`` is accepted for the shared call
    signature and unused: UniRoPE replaces the whole head on image tokens.
    ``apply_vo`` also rotates V and the output, as in RayRoPE.
    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    dim = q_content.shape[-1]
    if dim != HEAD_DIM:
        raise RuntimeError(f"UniRoPE expects head dim {HEAD_DIM}, got {dim}")
    batch, cameras, patches = geom["directions"].shape[:3]
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
        patches, q_extra, k_extra, rope_axis=None, apply_vo=apply_vo,
    )


@torch.compile(dynamic=False)
def _anchor_coefficients(
    c2w: torch.Tensor, directions: torch.Tensor, raw_depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE coefficients ``[B, img, half]`` of every token in camera 0's frame
    (the frame of the normalized poses). ``dynamic=False``: FLUX 3 calls it with
    2, 3 and 5 cameras, and an automatic dynamic-shape recompile saved different
    tensors for backward than the static graph, breaking gradient-checkpoint
    recomputation. One static graph per camera count avoids that."""
    batch, cameras, patches = directions.shape[:3]
    img_len = cameras * patches
    depths = depth_interval(raw_depth.float()).reshape(2, batch, cameras, patches, 1, 1)
    identity = torch.eye(3, device=c2w.device, dtype=c2w.dtype).expand(batch, -1, -1)
    directions_world = torch.einsum("bcij,bcprj->bcpri", c2w[:, :, :3, :3], directions.float())
    centers, rays, point_dirs, distance = _token_geometry(
        identity, torch.zeros_like(c2w[:, 0, :3, 3]), c2w[:, :, :3, 3], directions_world, depths
    )
    return _pack_cos_sin(
        centers.reshape(batch, img_len, 3), None if rays is None else rays.reshape(batch, img_len, 3),
        point_dirs.reshape(2, batch, img_len, -1), distance.reshape(2, batch, img_len, -1),
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
    apply_vo: bool = False,
) -> torch.Tensor:
    """UniRoPE with every token placed once in camera 0's frame: one attention pass.

    Same arguments as :func:`unirope_attention` (``rope_axis`` unused).
    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    dim = q_content.shape[-1]
    if dim != HEAD_DIM:
        raise RuntimeError(f"UniRoPE expects head dim {HEAD_DIM}, got {dim}")
    batch, cameras, patches = geom["directions"].shape[:3]
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
    if apply_vo:
        v = torch.cat((v[:, :, :num_txt], _rotate_vo(v[:, :, num_txt:], cosine, sine, None, False)), dim=2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    if apply_vo:
        out = torch.cat((out[:, :, :num_txt], _rotate_vo(out[:, :, num_txt:], cosine, sine, None, True)), dim=2)
    return rearrange(out, "b h s d -> b s (h d)")
