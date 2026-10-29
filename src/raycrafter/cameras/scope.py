"""SCoPE: Plücker-ray positional encoding added to attention Q/K/V.

Each image token gets one Plücker ray (direction, moment) computed at the
center of its patch. After RoPE, a projection of that ray is added to Q and K
(K sees direction and moment swapped). An optional frame-level residual is
added to V. Nothing is appended to the token sequence.

The Q/K output layers start at zero, so step 0 matches the pretrained
attention. See SCoPE ``scope/encoding.py`` for the Normalize-Gate-Inject form.
Defaults follow the released SCoPE recipe: a 7→64→dim MLP per projection,
alpha = 1, ``plucker_eps = 1e-3``, a log-scale shift on the gate input during
training, and no V residual. Translations are expected in near-depth units
(``near_depth_json`` in the data config).

This file is backbone-free: it works on ``[B, heads, txt+img, head_dim]``
tensors. ``backbones/flux2.py`` decides where in a block it is called.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .unirope import equirect_directions

# Fourier direction features (``dir_freqs``): periods over a unit-vector component.
# Longest = 4x its range [-1, 1] (RayRoPE's rule); shortest resolves adjacent tokens
# (about 0.035 rad apart in a 288x512 view, 0.012-0.04 in a 256x512 panorama).
DIR_MAX_PERIOD = 8.0
DIR_MIN_PERIOD = 0.05


def _pinhole_directions(fxfycxcy, cropped_h, cropped_w, top, left, latent_hw, device,
                        offset: tuple[float, float] = (0.5, 0.5)) -> torch.Tensor:
    """Camera-frame unit rays through the token centers (or another ``offset`` in
    the patch, in patches), ``[B, N, h, w, 3]``."""
    patches_y, patches_x = latent_hw
    fx = fxfycxcy[..., 0].float()
    fy = fxfycxcy[..., 1].float()
    cx = fxfycxcy[..., 2].float() - left
    cy = fxfycxcy[..., 3].float() - top

    ys = (torch.arange(patches_y, device=device) + offset[1]) * (cropped_h / patches_y)
    xs = (torch.arange(patches_x, device=device) + offset[0]) * (cropped_w / patches_x)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # [B, N, h, w]
    x = (grid_x - cx[:, :, None, None]) / fx[:, :, None, None]
    y = (grid_y - cy[:, :, None, None]) / fy[:, :, None, None]
    directions = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return F.normalize(directions, dim=-1, eps=1e-8)


def plucker_tokens(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    patch: int = 16,
    camera_model: str = "pinhole",
    patch_size: bool = False,
) -> torch.Tensor:
    """One Plücker ray per packed image token.

    Cameras are ``[ref..., views...]``, the same order the tokens are packed
    in (each image row-major over the latent grid). Poses are rewritten
    relative to camera 0. The image is assumed center-cropped to a multiple of
    ``patch`` before encoding, so principal points are shifted to match.

    Args:
        c2w: ``[B, N, 4, 4]`` camera-to-world (OpenCV convention).
        fxfycxcy: ``[B, N, 4]`` pinhole intrinsics in pixels of the full image
            (unused for ``equirect``).
        image_hw: ``(H, W)`` of the image given to the VAE.
        latent_hw: ``(h, w)`` token grid, one token per ``patch × patch`` pixels.
        camera_model: ``pinhole`` (released SCoPE) or ``equirect``: the ray
            through each token center of a full panorama (``cameras.unirope``
            convention).
        patch_size: append the patch's angular size, a 7th channel (see
            :func:`_patch_log_size`).

    Returns:
        ``[B, N*h*w, 6]`` as ``(direction, moment)``, or ``[B, N*h*w, 7]`` with
        ``patch_size``.
    """
    height, width = image_hw
    patches_y, patches_x = latent_hw
    cropped_h, cropped_w = patches_y * patch, patches_x * patch
    top = (height - cropped_h) // 2
    left = (width - cropped_w) // 2

    c2w = c2w.float()
    relative = torch.linalg.inv(c2w[:, :1]) @ c2w

    device = c2w.device
    if camera_model == "equirect":
        if (cropped_h, cropped_w) != (height, width):
            raise ValueError(f"equirect image {height}x{width} is not a whole number of {patch}px patches")
        batch, cameras = c2w.shape[:2]
        rays = equirect_directions(batch, cameras, latent_hw, device, footprint=patch_size)
        rays = rays.reshape(batch, cameras, patches_y, patches_x, -1, 3)
        directions = rays[..., 0, :]
        corners = rays[..., 1:, :] if patch_size else None
    elif camera_model == "pinhole":
        directions = _pinhole_directions(fxfycxcy, cropped_h, cropped_w, top, left, latent_hw, device)
        corners = None
        if patch_size:
            corners = torch.stack([
                _pinhole_directions(fxfycxcy, cropped_h, cropped_w, top, left, latent_hw, device, offset)
                for offset in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            ], dim=-2)
    else:
        raise ValueError(f"unknown camera_model {camera_model!r}")

    rotation = relative[:, :, :3, :3]
    directions = torch.einsum("bnij,bnhwj->bnhwi", rotation, directions)
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    origins = relative[:, :, None, None, :3, 3].expand_as(directions)
    moments = torch.cross(origins, directions, dim=-1)

    batch = c2w.shape[0]
    channels = [directions.reshape(batch, -1, 3), moments.reshape(batch, -1, 3)]
    if corners is not None:
        channels.append(_patch_log_size(corners).reshape(batch, -1, 1))
    return torch.cat(channels, dim=-1)


def _patch_log_size(corners: torch.Tensor) -> torch.Tensor:
    """Angular size of each patch from its corner rays ``[..., 3, 3]`` ((0,0),
    (1,0), (0,1)): the log of the geometric mean of its two side angles,
    relative to 0.05 rad (about a 16 px patch of a 60° view). Frame-free, so it
    needs no pose; a panorama's polar patches, whose top side shrinks to a
    point, are clamped at 1e-3 rad a side."""
    def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)).clamp(min=1e-3)

    side_x = angle(corners[..., 0, :], corners[..., 1, :])
    side_y = angle(corners[..., 0, :], corners[..., 2, :])
    return (0.5 * (side_x.log() + side_y.log()) - math.log(0.05)).unsqueeze(-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale * self.weight).to(dtype=x.dtype)


def _projection(dim: int, hidden: int, bias: bool = False, in_dim: int = 7) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden, bias=bias), nn.GELU(), nn.Linear(hidden, dim, bias=False))


class SightlineCoordinatePE(nn.Module):
    """Add scale-gated Plücker features to Q and K, and optionally a per-frame residual to V."""

    def __init__(
        self,
        dim: int,
        mlp_hidden: int = 64,
        plucker_eps: float = 1e-3,
        cam_residual: bool = False,
        log_scale_aug_prob: float = 0.3,
        log_scale_aug_range: tuple[float, float] = (-1.2, 1.6),
        dir_freqs: int = 0,
        patch_size: bool = False,
    ):
        """``dir_freqs`` > 0 adds sin/cos of each direction component at that many
        log-spaced periods from ``DIR_MAX_PERIOD`` to ``DIR_MIN_PERIOD`` (the
        components are continuous on the sphere, so there is no seam or pole);
        ``patch_size`` reads a 7th Plücker channel (the patch's angular size).
        Both are extra MLP inputs beside the released 7: at the defaults this is
        the released module."""
        super().__init__()
        self.dir_freqs = int(dir_freqs)
        self.patch_size = bool(patch_size)
        in_dim = 7 + 6 * self.dir_freqs + int(self.patch_size)
        if self.dir_freqs:
            ratio = DIR_MAX_PERIOD / DIR_MIN_PERIOD
            periods = DIR_MAX_PERIOD / ratio ** (torch.arange(self.dir_freqs) / max(self.dir_freqs - 1, 1))
            self.register_buffer("dir_omega", 2 * math.pi / periods, persistent=False)
        self.plucker_eps = float(plucker_eps)
        self.log_scale_aug_prob = float(log_scale_aug_prob)
        self.log_scale_aug_range = (float(log_scale_aug_range[0]), float(log_scale_aug_range[1]))
        self.eq = _projection(dim, mlp_hidden, in_dim=in_dim)
        self.ek = _projection(dim, mlp_hidden, in_dim=in_dim)
        self.norm_pe_q = _RMSNorm(dim)
        self.norm_pe_k = _RMSNorm(dim)
        hidden = max(dim // 4, 1)
        self.scale_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.alpha_q = nn.Parameter(torch.ones(1))
        self.alpha_k = nn.Parameter(torch.ones(1))
        self.cam_residual = cam_residual
        if cam_residual:
            self.ev = _projection(dim, mlp_hidden, in_dim=in_dim)
            self.gate_proj = _projection(dim, mlp_hidden, bias=True, in_dim=in_dim)
            self.gate_logit = nn.Parameter(torch.full((dim,), -2.0))
        self._init_weights()

    def _init_weights(self) -> None:
        # Zero only the output layer: zeroing both would leave GELU(0) = 0 and no gradient.
        nn.init.kaiming_uniform_(self.eq[0].weight, a=5**0.5)
        nn.init.zeros_(self.eq[2].weight)
        self.ek.load_state_dict(self.eq.state_dict())  # tied init, then trained independently
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)
        if self.cam_residual:
            nn.init.normal_(self.ev[0].weight, 0.0, 0.01)
            nn.init.normal_(self.ev[2].weight, 0.0, 0.01)
            nn.init.xavier_uniform_(self.gate_proj[0].weight)
            nn.init.zeros_(self.gate_proj[0].bias)
            nn.init.zeros_(self.gate_proj[2].weight)

    def decompose(self, plucker_6d: torch.Tensor):
        expected = 6 + int(self.patch_size)
        if plucker_6d.shape[-1] != expected:
            raise RuntimeError(f"SCoPE expects {expected} ray channels, got {plucker_6d.shape[-1]}")
        direction = plucker_6d[..., :3]
        moment = plucker_6d[..., 3:6]
        moment_norm = moment.norm(dim=-1, keepdim=True).clamp(min=self.plucker_eps)
        moment_hat = moment / moment_norm
        log_scale = torch.log(moment_norm)
        # Q and K swap direction and moment: their product carries the rays' reciprocal product.
        feat_q = [direction, moment_hat, log_scale]
        feat_k = [moment_hat, direction, log_scale]
        extra = []
        if self.dir_freqs:
            angle = direction.unsqueeze(-1) * self.dir_omega  # [..., 3, F]
            extra.append(torch.cat((angle.sin(), angle.cos()), dim=-1).flatten(-2))
        if self.patch_size:
            extra.append(plucker_6d[..., 6:7])
        return torch.cat(feat_q + extra, dim=-1), torch.cat(feat_k + extra, dim=-1), log_scale

    def _gate_log_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        """In training, shift the gate's log-scale by one uniform draw per sample
        (a global rescale of the translations). E_q / E_k keep the true value."""
        if not self.training or self.log_scale_aug_prob <= 0.0:
            return log_scale
        if torch.rand(()).item() > self.log_scale_aug_prob:
            return log_scale
        low, high = self.log_scale_aug_range
        shift = torch.empty(log_scale.shape[0], 1, 1, device=log_scale.device, dtype=log_scale.dtype)
        return log_scale + shift.uniform_(low, high)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``q``/``k`` are ``[B, S, dim]`` image tokens. Returns ``(q, k, v_residual or None)``."""
        dtype = q.dtype
        feat_q, feat_k, log_scale = self.decompose(plucker_6d)
        feat_q = feat_q.to(dtype)
        feat_k = feat_k.to(dtype)
        gate = self.scale_gate(self._gate_log_scale(log_scale).to(dtype))
        pe_q = self.alpha_q * gate * self.norm_pe_q(self.eq(feat_q))
        pe_k = self.alpha_k * gate * self.norm_pe_k(self.ek(feat_k))
        if not self.cam_residual:
            return q + pe_q, k + pe_k, None

        batch, seq, channels = feat_q.shape
        spatial = seq // num_frames
        feat_frame = feat_q.reshape(batch, num_frames, spatial, channels).mean(dim=2, keepdim=True)
        feat_frame = feat_frame.expand(batch, num_frames, spatial, channels).reshape(batch, seq, channels)
        cam_gate = torch.sigmoid(self.gate_logit + self.gate_proj(feat_frame))
        cam_residual = cam_gate.to(dtype) * self.ev(feat_frame)
        return q + pe_q, k + pe_k, cam_residual


def scope_qkv(
    pe: SightlineCoordinatePE,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_txt: int,
    plucker: torch.Tensor,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SCoPE to the image part of ``[B, heads, txt+img, head_dim]`` Q/K/V.

    Text tokens pass through unchanged. ``plucker`` is ``[B, img, 6]``.
    """
    img_len = q.shape[2] - num_txt
    if img_len != plucker.shape[1]:
        raise RuntimeError(f"SCoPE has {plucker.shape[1]} rays for {img_len} image tokens")
    heads = q.shape[1]
    q_img = rearrange(q[:, :, num_txt:], "b h s d -> b s (h d)")
    k_img = rearrange(k[:, :, num_txt:], "b h s d -> b s (h d)")
    q_img, k_img, cam = pe(q_img, k_img, plucker, num_frames=num_frames)

    def split(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b s (h d) -> b h s d", h=heads)

    q = torch.cat((q[:, :, :num_txt], split(q_img)), dim=2)
    k = torch.cat((k[:, :, :num_txt], split(k_img)), dim=2)
    if cam is not None:
        v = torch.cat((v[:, :, :num_txt], v[:, :, num_txt:] + split(cam)), dim=2)
    return q, k, v
