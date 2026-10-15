"""RayRoPE: projective-ray rotary encoding on part of each attention head.

Head layout (FLUX Klein heads are 128-d): channels ``[0:ROPE_DIM]`` keep the
backbone's own RoPE (for FLUX, the ``t`` axis = which image), and channels
``[ROPE_DIM:]`` are replaced by RayRoPE (``d_pj+0_3d``: camera center in the
query frame, plus three rays at a predicted depth). RayRoPE's 12 projective
coordinates need 24 channels per frequency, so ``RAY_DIM=96`` gives 4
frequencies.

Q and K of image tokens are rotated; V is not. Each camera is a query in turn:
every image token is re-expressed in that query camera, then that camera's
queries attend to all tokens. Text keeps the backbone RoPE.

This file is backbone-free: it works on ``[B, heads, txt+img, head_dim]``
tensors. ``backbones/flux2.py`` wires it into FLUX blocks.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# FLUX head layout: dims [0:32] = t, [32:128] = h, w, l.
ROPE_DIM = 32
RAY_DIM = 96
HEAD_DIM = ROPE_DIM + RAY_DIM
NUM_COORDS = 12  # p0 xyz (3) + 3 rays * (dir xy + depth)
NUM_FREQS = RAY_DIM // (2 * NUM_COORDS)  # 4

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


def _freqs(num_freqs: int, max_period: float, device: torch.device) -> torch.Tensor:
    min_period = max_period / (FREQ_BASE ** (num_freqs - 1))
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
    """
    c2w = c2w.float()
    relative = torch.linalg.inv(c2w[:, :1]) @ c2w
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


def _depths(raw: torch.Tensor, num_cameras: int, num_patches: int) -> torch.Tensor:
    """``raw`` ``[B, S, 2]`` → two depths ``[2, B, cameras, patches, rays, 1]``."""
    batch = raw.shape[0]
    raw = raw.reshape(batch, num_cameras, num_patches, 2)
    log_depth = raw[..., 0]
    sigma = raw[..., 1]
    depth_lo = torch.exp(torch.clamp(log_depth - sigma, max=MAX_LOG_DEPTH))
    depth_hi = torch.exp(torch.clamp(log_depth + sigma, max=MAX_LOG_DEPTH))
    depths = torch.stack((depth_lo, depth_hi), dim=0)
    depths = depths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, len(_OFFSETS), 1)
    return depths.clamp(min=1e-2, max=MAX_DEPTH)


def _world_points(projection_inv: torch.Tensor, uv: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """Back-project ``[u, v, 1, 1/d]`` with ``P^{-1}``. ``depths`` is ``[2, B, C, P, R, 1]``."""
    disparity = 1.0 / depths
    ones = torch.ones_like(disparity)
    uv = uv.to(device=depths.device, dtype=depths.dtype)
    uv = uv.view(1, 1, 1, *uv.shape).expand(2, depths.shape[1], depths.shape[2], -1, -1, -1)
    homogeneous = torch.cat((uv, ones, disparity), dim=-1)
    return torch.einsum("bcij,ebcprj->ebcpri", projection_inv, homogeneous)


def _query_positions(
    geom: dict,
    raw_depth: torch.Tensor,
    camera_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Points of every camera, expressed in camera ``camera_index``.

    Returns centers ``[B, C, P, 3]``, rays ``[2, B, C, P, 9]`` (3 rays × dir xy
    and depth, two depths), and the same tensors sliced to the query camera.
    """
    patches_y = geom["patches_y"]
    patches_x = geom["patches_x"]
    num_patches = patches_y * patches_x
    num_cameras = geom["num_cameras"]
    depths = _depths(raw_depth.float(), num_cameras, num_patches)
    uv = _patch_uv(patches_x, patches_y, raw_depth.device)
    world = _world_points(geom["projection_inv"], uv, depths)

    w2c_q = geom["w2c"][:, camera_index]
    projection_q = geom["projection"][:, camera_index]
    # Camera center in the camera-0 frame is the translation column of c2w.
    centers = geom["c2w"][:, :, :, 3]
    centers = centers.unsqueeze(2).expand(-1, -1, num_patches, -1)
    centers_q = torch.einsum("bij,bcpj->bcpi", w2c_q, centers)
    centers_q = centers_q[..., :3] / centers_q[..., 3:].clamp(min=1e-4)

    projected = torch.einsum("bij,ebcprj->ebcpri", projection_q, world)
    z = projected[..., 2:3].abs().clamp(min=1e-4)
    homogeneous_w = projected[..., 3:4].clamp(min=1e-4)
    direction = projected[..., :3] / (projected[..., :3].norm(dim=-1, keepdim=True) + 1e-6)
    depth = (z / homogeneous_w).clamp(min=-MAX_D_F, max=MAX_D_F)
    direction_xy = direction[..., :2].flatten(-2, -1)
    depth = depth.flatten(-2, -1)
    rays = torch.cat((direction_xy, depth), dim=-1)

    query_centers = centers_q[:, camera_index]
    query_rays = rays[:, :, camera_index]
    return centers_q, rays, query_centers, query_rays


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


def _rotate(feats: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RayRoPE rotation on the last 96 channels. ``feats`` is ``[B, H, S, 96]``."""
    half = RAY_DIM // 2
    x1, x2 = feats[..., :half], feats[..., half:]
    cos = cos[:, None].to(dtype=feats.dtype)
    sin = sin[:, None].to(dtype=feats.dtype)
    rotated_1 = x1 * cos + x2 * sin
    rotated_2 = -x1 * sin + x2 * cos
    return torch.cat((rotated_1, rotated_2), dim=-1)


def _mix(rope_feats: torch.Tensor, content: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Keep the pretrained ``t`` RoPE; replace the other 96 channels."""
    rotated = _rotate(content[..., ROPE_DIM:], cos, sin)
    return torch.cat((rope_feats[..., :ROPE_DIM], rotated), dim=-1)




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
) -> torch.Tensor:
    """Full attention with per-camera RayRoPE on image queries.

    ``*_content`` are Q/K before any RoPE and ``*_rope`` after the backbone
    RoPE; both are ``[B, heads, txt+img, HEAD_DIM]``. Images are packed
    ``[ref..., views...]``. ``raw_depth`` is ``[B, img, 2]`` from a
    :class:`DepthHead`. ``q_extra`` / ``k_extra`` (e.g. SCoPE) are added after
    the ray rotation and must be zero on text.

    Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    from einops import rearrange

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
    image_outs = []
    for camera_index in range(num_cameras):
        centers, rays, query_centers, query_rays = _query_positions(geom, raw_depth, camera_index)
        cos_k, sin_k = _pack_cos_sin(
            centers.reshape(batch, img_len, 3), rays.reshape(2, batch, img_len, 9)
        )
        cos_q, sin_q = _pack_cos_sin(
            query_centers.reshape(batch, num_patches, 3),
            query_rays.reshape(2, batch, num_patches, 9),
        )
        start = camera_index * num_patches
        end = start + num_patches
        k_img = _mix(k_img_rope, k_img_content, cos_k, sin_k)
        if k_extra is not None:
            k_img = k_img + k_extra[:, :, num_txt:]
        k_all = torch.cat((k_txt, k_img), dim=2) if num_txt else k_img
        q_i = _mix(q_img_rope[:, :, start:end], q_img_content[:, :, start:end], cos_q, sin_q)
        if q_extra is not None:
            q_i = q_i + q_extra[:, :, num_txt + start : num_txt + end]
        image_outs.append(F.scaled_dot_product_attention(q_i, k_all, v, is_causal=False))

    outputs.append(torch.cat(image_outs, dim=2))
    out = torch.cat(outputs, dim=2)
    return rearrange(out, "b h s d -> b s (h d)")
