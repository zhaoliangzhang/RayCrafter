"""RayRoPE: projective-ray rotary encoding on the attention heads.

Head layout (FLUX Klein heads are 128-d): RayRoPE (``d_pj+0_3d``: camera
center in the query frame, plus three rays at a predicted depth) takes the
whole head on image tokens, with no backbone RoPE axis kept. Its 12
coordinates need 24 channels per frequency: 5 frequencies (adjacent periods ×3,
as in the reference) rotate 120 channels and the last 8 pass through.

Q and K of image tokens are rotated; V is not. Each camera is a query in turn:
every image token is re-expressed in that query camera, then that camera's
queries attend to all tokens. Text keeps the backbone RoPE.

As in the reference RayRoPE, poses are normalized so camera 0 is the identity
and camera 1 is at unit distance; the frequency ranges and depth clamps below
assume that scale.

This file is backbone-free: it works on ``[B, heads, txt+img, head_dim]``
tensors. ``backbones/flux2.py`` and ``backbones/flux3.py`` wire it into FLUX
blocks; ``cameras/prope.py`` shares the geometry and the channel helpers for
keeping one backbone RoPE axis (``rope_axis``: ``t`` = 0 for FLUX.2, ``l`` = 3
for FLUX.3), and ``cameras/unirope.py`` the per-camera attention.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# FLUX head layout: dims [0:32] = t, [32:64] = h, [64:96] = w, [96:128] = l.
ROPE_DIM = 32
RAY_DIM = 96
HEAD_DIM = ROPE_DIM + RAY_DIM
NUM_COORDS = 12  # p0 xyz (3) + 3 rays * (dir xy + depth)
NUM_FREQS = HEAD_DIM // (2 * NUM_COORDS)  # 5: 120 rotated channels, 8 pass through

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_D_F = 10.0
FREQ_BASE = 3.0

_OFFSETS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


class DepthHead(nn.Module):
    """Per-token log-depth and uncertainty. Zero weights leave depth at the bias."""

    def __init__(self, dim: int, init_depth: float = 0.0, init_sigma: float = 3.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dim))
        self.bias = nn.Parameter(torch.tensor([init_depth, init_sigma]))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``[B, S, dim]`` image tokens. Returns ``[B, S, 2]``."""
        return F.linear(tokens, self.weight, self.bias)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation = transforms[..., :3, :3]
    rotation_inv = rotation.transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rotation_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_k(k3: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(*k3.shape[:-2], 4, 4, device=k3.device, dtype=k3.dtype)
    out[..., :3, :3] = k3
    out[..., 3, 3] = 1.0
    return out


def _invert_k(k3: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(k3)
    out[..., 0, 0] = 1.0 / k3[..., 0, 0]
    out[..., 1, 1] = 1.0 / k3[..., 1, 1]
    out[..., 0, 2] = -k3[..., 0, 2] / k3[..., 0, 0]
    out[..., 1, 2] = -k3[..., 1, 2] / k3[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _normalize_k(k3: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    height, width = image_hw
    out = torch.zeros_like(k3)
    out[..., 0, 0] = k3[..., 0, 0] / width
    out[..., 1, 1] = k3[..., 1, 1] / height
    out[..., 0, 2] = k3[..., 0, 2] / width - 0.5
    out[..., 1, 2] = k3[..., 1, 2] / height - 0.5
    out[..., 2, 2] = 1.0
    return out


def _intrinsics(fxfycxcy: torch.Tensor) -> torch.Tensor:
    k3 = torch.zeros(*fxfycxcy.shape[:-1], 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype)
    k3[..., 0, 0] = fxfycxcy[..., 0]
    k3[..., 1, 1] = fxfycxcy[..., 1]
    k3[..., 0, 2] = fxfycxcy[..., 2]
    k3[..., 1, 2] = fxfycxcy[..., 3]
    k3[..., 2, 2] = 1.0
    return k3


def _freqs(num_freqs: int, max_period: float, device: torch.device, base: float = FREQ_BASE) -> torch.Tensor:
    """``num_freqs`` log-spaced frequencies; adjacent periods differ by ``base``."""
    min_period = max_period / (base ** (num_freqs - 1))
    log_min = math.log(2.0 * math.pi / max_period)
    log_max = math.log(2.0 * math.pi / min_period)
    return torch.exp(torch.linspace(log_min, log_max, num_freqs, device=device, dtype=torch.float32))


def _trig_interval(angle_lo: torch.Tensor, angle_hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of (cos, sin) over a uniform draw between the two angles."""
    same = torch.isclose(angle_lo, angle_hi, atol=1e-2, rtol=0.0)
    cos_lo, cos_hi = torch.cos(angle_lo), torch.cos(angle_hi)
    sin_lo, sin_hi = torch.sin(angle_lo), torch.sin(angle_hi)
    delta = angle_hi - angle_lo
    delta_safe = torch.where(same, torch.ones_like(delta), delta)
    cos_mean = (sin_hi - sin_lo) / delta_safe
    sin_mean = (cos_lo - cos_hi) / delta_safe
    return torch.where(same, cos_lo, cos_mean), torch.where(same, sin_lo, sin_mean)


def _angles(coords: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``coords`` ``[..., D]``, ``freqs`` ``[F]`` → ``[..., F, D]``."""
    return coords.unsqueeze(-2) * freqs.view(*([1] * (coords.ndim - 1)), -1, 1)


def prepare_geometry(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
) -> dict:
    """Camera matrices shared by every block. Poses are relative to camera 0.

    Cameras are ordered ``[ref..., views...]``, the same order as packed tokens.
    Translations are scaled so camera 1 is at unit distance from camera 0
    (RayRoPE's ``_normalize_poses_identity_unit_distance``); a pair closer than
    1e-2 is left unscaled.
    """
    c2w = c2w.float()
    relative = torch.linalg.inv(c2w[:, :1]) @ c2w
    if relative.shape[1] > 1:
        dist = relative[:, 1, :3, 3].norm(dim=-1)
        scale = torch.where(dist > 1e-2, dist, torch.ones_like(dist))
        relative = relative.clone()
        relative[:, :, :3, 3] /= scale[:, None, None]
    w2c = _invert_se3(relative)
    k_norm = _normalize_k(_intrinsics(fxfycxcy.float()), image_hw)
    projection = torch.matmul(_lift_k(k_norm), w2c)
    projection_inv = torch.matmul(relative, _lift_k(_invert_k(k_norm)))
    patches_y, patches_x = latent_hw
    return {
        "c2w": relative,
        "w2c": w2c,
        "projection": projection,
        "projection_inv": projection_inv,
        "patches_y": int(patches_y),
        "patches_x": int(patches_x),
        "num_cameras": int(c2w.shape[1]),
    }


def _patch_uv(patches_x: int, patches_y: int, device: torch.device) -> torch.Tensor:
    """Normalized patch-corner coordinates, width fastest, matching packed tokens."""
    columns = torch.arange(patches_x, device=device, dtype=torch.float32)
    rows = torch.arange(patches_y, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(columns, rows, indexing="xy")
    corners = []
    for ox, oy in _OFFSETS:
        u = (grid_x + ox) / patches_x - 0.5
        v = (grid_y + oy) / patches_y - 0.5
        corners.append(torch.stack((u, v), dim=-1).reshape(-1, 2))
    return torch.stack(corners, dim=1)  # (patches, rays, 2)


def depth_interval(raw: torch.Tensor) -> torch.Tensor:
    """Depth-head output ``[..., 2]`` (log depth, sigma) → ``[2, ...]`` depths ``exp(log d ∓ sigma)``."""
    log_depth = raw[..., 0]
    sigma = raw[..., 1]
    depth_lo = torch.exp(torch.clamp(log_depth - sigma, max=MAX_LOG_DEPTH))
    depth_hi = torch.exp(torch.clamp(log_depth + sigma, max=MAX_LOG_DEPTH))
    return torch.stack((depth_lo, depth_hi), dim=0).clamp(min=1e-2, max=MAX_DEPTH)


def _depths(raw: torch.Tensor, num_cameras: int, num_patches: int) -> torch.Tensor:
    """``raw`` ``[B, S, 2]`` → two depths ``[2, B, cameras, patches, rays, 1]``."""
    depths = depth_interval(raw.reshape(raw.shape[0], num_cameras, num_patches, 2))
    return depths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, len(_OFFSETS), 1)


def _world_points(projection_inv: torch.Tensor, uv: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """Back-project ``[u, v, 1, 1/d]`` with ``P^{-1}``. ``depths`` is ``[2, B, C, P, R, 1]``."""
    disparity = 1.0 / depths
    ones = torch.ones_like(disparity)
    uv = uv.to(device=depths.device, dtype=depths.dtype)
    uv = uv.view(1, 1, 1, *uv.shape).expand(2, depths.shape[1], depths.shape[2], -1, -1, -1)
    homogeneous = torch.cat((uv, ones, disparity), dim=-1)
    return torch.einsum("bcij,ebcprj->ebcpri", projection_inv, homogeneous)


def _query_positions(
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    projection: torch.Tensor,
    world: torch.Tensor,
    camera_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Points of every camera, expressed in camera ``camera_index``.

    ``world`` is ``[2, B, C, P, R, 4]`` from :func:`_world_points`. Returns
    centers ``[B, C, P, 3]`` and rays ``[2, B, C, P, 9]`` (3 rays × dir xy and
    depth, two depths).
    """
    num_patches = world.shape[3]
    w2c_q = w2c[:, camera_index]
    projection_q = projection[:, camera_index]
    # Camera center in the camera-0 frame is the translation column of c2w.
    centers = c2w[:, :, :, 3]
    centers = centers.unsqueeze(2).expand(-1, -1, num_patches, -1)
    centers_q = torch.einsum("bij,bcpj->bcpi", w2c_q, centers)
    centers_q = centers_q[..., :3] / centers_q[..., 3:].clamp(min=1e-4)

    projected = torch.einsum("bij,ebcprj->ebcpri", projection_q, world)
    # sqrt(z^2 + eps) rather than |z|: smooth gradient at z = 0.
    z = torch.sqrt(projected[..., 2:3].pow(2) + 1e-9).clamp(min=1e-4)
    homogeneous_w = projected[..., 3:4].clamp(min=1e-4)
    direction = projected[..., :3] / (projected[..., :3].norm(dim=-1, keepdim=True) + 1e-6)
    depth = (z / homogeneous_w).clamp(min=-MAX_D_F, max=MAX_D_F)
    direction_xy = direction[..., :2].flatten(-2, -1)
    depth = depth.flatten(-2, -1)
    rays = torch.cat((direction_xy, depth), dim=-1)
    return centers_q, rays


def _pack_cos_sin(
    centers: torch.Tensor,
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``centers [B, N, 3]``, ``rays [2, B, N, 9]`` → cos/sin ``[B, N, 48]``."""
    device = centers.device
    freq_center = _freqs(NUM_FREQS, 1.0 * 4, device)
    freq_dir = _freqs(NUM_FREQS, 2.0 * 4, device)
    freq_depth = _freqs(NUM_FREQS, MAX_D_F * 2 * 4, device)

    center_angle = _angles(centers, freq_center)
    cos_center, sin_center = torch.cos(center_angle), torch.sin(center_angle)

    direction = rays[..., :6]
    depth = rays[..., 6:]
    dir_lo, dir_hi = _angles(direction[0], freq_dir), _angles(direction[1], freq_dir)
    dep_lo, dep_hi = _angles(depth[0], freq_depth), _angles(depth[1], freq_depth)
    cos_dir, sin_dir = _trig_interval(dir_lo, dir_hi)
    cos_dep, sin_dep = _trig_interval(dep_lo, dep_hi)

    cosine = torch.cat((cos_center, cos_dir, cos_dep), dim=-1).flatten(-2, -1)
    sine = torch.cat((sin_center, sin_dir, sin_dep), dim=-1).flatten(-2, -1)
    return cosine, sine


@torch.compile
def _coefficients(
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    projection: torch.Tensor,
    projection_inv: torch.Tensor,
    raw_depth: torch.Tensor,
    patches_x: int,
    patches_y: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE coefficients of one block (RayRoPE's ``_prepare_apply_fns``).

    World points are back-projected once, then expressed in each query camera.
    Returns query ``cos, sin`` ``[B, img, 48]`` (each token in its own camera)
    and key ``cos, sin`` ``[C, B, img, 48]`` (all tokens, per query camera).
    """
    batch = raw_depth.shape[0]
    num_cameras = c2w.shape[1]
    num_patches = patches_y * patches_x
    img_len = num_cameras * num_patches
    depths = _depths(raw_depth.float(), num_cameras, num_patches)
    uv = _patch_uv(patches_x, patches_y, raw_depth.device)
    world = _world_points(projection_inv, uv, depths)

    cos_k, sin_k, query_centers, query_rays = [], [], [], []
    for camera_index in range(num_cameras):
        centers, rays = _query_positions(c2w, w2c, projection, world, camera_index)
        cosine, sine = _pack_cos_sin(
            centers.reshape(batch, img_len, 3), rays.reshape(2, batch, img_len, 9)
        )
        cos_k.append(cosine)
        sin_k.append(sine)
        query_centers.append(centers[:, camera_index])
        query_rays.append(rays[:, :, camera_index])
    cos_q, sin_q = _pack_cos_sin(
        torch.stack(query_centers, dim=1).reshape(batch, img_len, 3),
        torch.stack(query_rays, dim=2).reshape(2, batch, img_len, 9),
    )
    return cos_q, sin_q, torch.stack(cos_k), torch.stack(sin_k)


def _rotate(feats: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Split-half rotation of the first ``2 * cos.shape[-1]`` of the 96 camera
    channels (all of them for RayRoPE); the rest pass through. ``feats`` is
    ``[B, H, S, 96]``, ``cos`` / ``sin`` ``[B, S, half]``."""
    half = cos.shape[-1]
    x1, x2, rest = feats[..., :half], feats[..., half : 2 * half], feats[..., 2 * half :]
    cos = cos[:, None].to(dtype=feats.dtype)
    sin = sin[:, None].to(dtype=feats.dtype)
    rotated_1 = x1 * cos + x2 * sin
    rotated_2 = -x1 * sin + x2 * cos
    return torch.cat((rotated_1, rotated_2, rest), dim=-1)


@torch.compile
def _mix(
    rope_feats: torch.Tensor, content: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rope_axis: Optional[int],
) -> torch.Tensor:
    """Keep the pretrained RoPE of ``rope_axis`` and replace the other 96 channels;
    with ``rope_axis=None`` the camera encoding takes the whole head."""
    if rope_axis is None:
        return _rotate(content, cos, sin)
    _, camera_channels = split_channels(content, rope_axis)
    kept, _ = split_channels(rope_feats, rope_axis)
    return merge_channels(kept, _rotate(camera_channels, cos, sin), rope_axis)


def split_channels(feats: torch.Tensor, rope_axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., 128]`` → the kept backbone-RoPE channels ``[..., 32]`` and the
    camera channels ``[..., 96]`` (the other three axes, in order)."""
    start = rope_axis * ROPE_DIM
    end = start + ROPE_DIM
    return feats[..., start:end], torch.cat((feats[..., :start], feats[..., end:]), dim=-1)


def merge_channels(kept: torch.Tensor, camera_channels: torch.Tensor, rope_axis: int) -> torch.Tensor:
    """Inverse of :func:`split_channels`."""
    start = rope_axis * ROPE_DIM
    return torch.cat((camera_channels[..., :start], kept, camera_channels[..., start:]), dim=-1)


def subset_geometry(geom: dict, start: int, end: int) -> dict:
    """Geometry of cameras ``start:end``, in the same normalized frame."""
    out = dict(geom)
    for key in ("c2w", "w2c", "projection", "projection_inv", "directions"):
        if key in geom:
            out[key] = geom[key][:, start:end]
    out["num_cameras"] = end - start
    return out




def rayrope_attention(
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
    """Full attention with per-camera RayRoPE on image queries.

    ``*_content`` are Q/K before any RoPE and ``*_rope`` after the backbone
    RoPE; both are ``[B, heads, txt+img, HEAD_DIM]``. Images are packed in
    camera order (``[ref..., views...]``). ``raw_depth`` is ``[B, img, 2]``
    from a :class:`DepthHead`. ``q_extra`` / ``k_extra`` (e.g. SCoPE) are
    added after the ray rotation and must be zero on text. ``rope_axis`` is
    accepted for the shared call signature and unused: RayRoPE replaces the
    whole head on image tokens.

    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """

    batch, heads, seqlen, dim = q_content.shape
    if dim != HEAD_DIM:
        raise RuntimeError(f"RayRoPE expects head dim {HEAD_DIM}, got {dim}")
    num_cameras = geom["num_cameras"]
    num_patches = geom["patches_y"] * geom["patches_x"]
    img_len = num_cameras * num_patches
    if seqlen - num_txt != img_len:
        raise RuntimeError(
            f"RayRoPE image tokens {seqlen - num_txt} != "
            f"{num_cameras} cameras × {num_patches} patches"
        )
    if raw_depth.shape[1] != img_len:
        raise RuntimeError(f"Depth head length {raw_depth.shape[1]} != image tokens {img_len}")

    # Geometry in fp32: under bf16 autocast the einsums would round the
    # coordinates, which the highest frequencies turn into ~0.5 rad of phase noise.
    with torch.autocast(device_type=q_content.device.type, enabled=False):
        cos_q, sin_q, cos_k, sin_k = _coefficients(
            geom["c2w"], geom["w2c"], geom["projection"], geom["projection_inv"],
            raw_depth, geom["patches_x"], geom["patches_y"],
        )
    return per_camera_attention(
        q_content, k_content, v, q_rope, k_rope, num_txt, cos_q, sin_q, cos_k, sin_k,
        num_patches, q_extra, k_extra, rope_axis=None,
    )


def per_camera_attention(
    q_content: torch.Tensor,
    k_content: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    num_txt: int,
    cos_q: torch.Tensor,
    sin_q: torch.Tensor,
    cos_k: torch.Tensor,
    sin_k: torch.Tensor,
    num_patches: int,
    q_extra: Optional[torch.Tensor] = None,
    k_extra: Optional[torch.Tensor] = None,
    rope_axis: Optional[int] = 0,
) -> torch.Tensor:
    """Each camera's queries attend to all tokens rotated into that camera's frame.

    ``cos_q`` / ``sin_q`` ``[B, img, half]`` place every image token in its own
    camera; ``cos_k`` / ``sin_k`` ``[C, B, img, half]`` place all image tokens in
    query camera ``c``. ``rope_axis`` is the backbone RoPE axis kept, or ``None``
    when the coefficients cover the whole head. Shared by RayRoPE and UniRoPE.
    """
    from einops import rearrange

    outputs = []
    if num_txt:
        outputs.append(
            F.scaled_dot_product_attention(q_rope[:, :, :num_txt], k_rope, v, is_causal=False)
        )

    q_img_content = q_content[:, :, num_txt:]
    k_img_content = k_content[:, :, num_txt:]
    q_img_rope = q_rope[:, :, num_txt:]
    k_img_rope = k_rope[:, :, num_txt:]
    k_txt = k_rope[:, :, :num_txt]
    q_img = _mix(q_img_rope, q_img_content, cos_q, sin_q, rope_axis)
    if q_extra is not None:
        q_img = q_img + q_extra[:, :, num_txt:]
    image_outs = []
    for camera_index in range(cos_k.shape[0]):
        start = camera_index * num_patches
        end = start + num_patches
        k_img = _mix(k_img_rope, k_img_content, cos_k[camera_index], sin_k[camera_index], rope_axis)
        if k_extra is not None:
            k_img = k_img + k_extra[:, :, num_txt:]
        k_all = torch.cat((k_txt, k_img), dim=2) if num_txt else k_img
        q_i = q_img[:, :, start:end]
        image_outs.append(F.scaled_dot_product_attention(q_i, k_all, v, is_causal=False))

    outputs.append(torch.cat(image_outs, dim=2))
    out = torch.cat(outputs, dim=2)
    return rearrange(out, "b h s d -> b s (h d)")
