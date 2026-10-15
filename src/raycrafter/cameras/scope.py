"""SCoPE: Plücker-ray positional encoding added to attention Q/K/V.

Each image token gets one Plücker ray (direction, moment) computed at the
center of its patch. After RoPE, a projection of that ray is added to Q and K
(K sees direction and moment swapped), and a frame-level residual is added to V.
Nothing is appended to the token sequence.

All output projections start at zero, so step 0 matches the pretrained
attention. See SCoPE ``scope/encoding.py`` for the Normalize-Gate-Inject form.

This file is backbone-free: it works on ``[B, heads, txt+img, head_dim]``
tensors. ``backbones/flux2.py`` decides where in a block it is called.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def plucker_tokens(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    patch: int = 16,
) -> torch.Tensor:
    """One Plücker ray per packed image token.

    Cameras are ``[ref..., views...]``, the same order the tokens are packed
    in (each image row-major over the latent grid). Poses are rewritten
    relative to camera 0. The image is assumed center-cropped to a multiple of
    ``patch`` before encoding, so principal points are shifted to match.

    Args:
        c2w: ``[B, N, 4, 4]`` camera-to-world (OpenCV convention).
        fxfycxcy: ``[B, N, 4]`` pinhole intrinsics in pixels of the full image.
        image_hw: ``(H, W)`` of the image given to the VAE.
        latent_hw: ``(h, w)`` token grid, one token per ``patch × patch`` pixels.

    Returns:
        ``[B, N*h*w, 6]`` as ``(direction, moment)``.
    """
    height, width = image_hw
    patches_y, patches_x = latent_hw
    cropped_h, cropped_w = patches_y * patch, patches_x * patch
    top = (height - cropped_h) // 2
    left = (width - cropped_w) // 2

    c2w = c2w.float()
    relative = torch.linalg.inv(c2w[:, :1]) @ c2w

    fx = fxfycxcy[..., 0].float()
    fy = fxfycxcy[..., 1].float()
    cx = fxfycxcy[..., 2].float() - left
    cy = fxfycxcy[..., 3].float() - top

    device = c2w.device
    ys = (torch.arange(patches_y, device=device) + 0.5) * (cropped_h / patches_y)
    xs = (torch.arange(patches_x, device=device) + 0.5) * (cropped_w / patches_x)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # [B, N, h, w]
    x = (grid_x - cx[:, :, None, None]) / fx[:, :, None, None]
    y = (grid_y - cy[:, :, None, None]) / fy[:, :, None, None]
    directions = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    directions = F.normalize(directions, dim=-1, eps=1e-8)

    rotation = relative[:, :, :3, :3]
    directions = torch.einsum("bnij,bnhwj->bnhwi", rotation, directions)
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    origins = relative[:, :, None, None, :3, 3].expand_as(directions)
    moments = torch.cross(origins, directions, dim=-1)

    batch = c2w.shape[0]
    return torch.cat(
        (directions.reshape(batch, -1, 3), moments.reshape(batch, -1, 3)),
        dim=-1,
    )


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale * self.weight).to(dtype=x.dtype)


class SightlineCoordinatePE(nn.Module):
    """Add scale-gated Plücker features to Q and K, and a per-frame residual to V."""

    def __init__(self, dim: int, plucker_eps: float = 1e-6):
        super().__init__()
        self.plucker_eps = float(plucker_eps)
        self.eq = nn.Linear(7, dim, bias=False)
        self.ek = nn.Linear(7, dim, bias=False)
        self.ev = nn.Linear(7, dim, bias=False)
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
        self.gate_proj = nn.Linear(7, dim, bias=False)
        self.gate_logit = nn.Parameter(torch.full((dim,), -2.0))
        self._zero_outputs()

    def _zero_outputs(self) -> None:
        nn.init.zeros_(self.eq.weight)
        nn.init.zeros_(self.ek.weight)
        nn.init.zeros_(self.ev.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)

    def decompose(self, plucker_6d: torch.Tensor):
        direction = plucker_6d[..., :3]
        moment = plucker_6d[..., 3:]
        moment_norm = moment.norm(dim=-1, keepdim=True).clamp(min=self.plucker_eps)
        moment_hat = moment / moment_norm
        log_scale = torch.log(moment_norm)
        feat_q = torch.cat((direction, moment_hat, log_scale), dim=-1)
        feat_k = torch.cat((moment_hat, direction, log_scale), dim=-1)
        return feat_q, feat_k, log_scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``q``/``k`` are ``[B, S, dim]`` image tokens. Returns ``(q, k, v_residual)``."""
        dtype = q.dtype
        feat_q, feat_k, log_scale = self.decompose(plucker_6d)
        feat_q = feat_q.to(dtype)
        feat_k = feat_k.to(dtype)
        gate = self.scale_gate(log_scale.to(dtype))
        pe_q = self.alpha_q * gate * self.norm_pe_q(self.eq(feat_q))
        pe_k = self.alpha_k * gate * self.norm_pe_k(self.ek(feat_k))

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
    v = torch.cat((v[:, :, :num_txt], v[:, :, num_txt:] + split(cam)), dim=2)
    return q, k, v
