"""PRoPE: relative projective positional encoding on part of each attention head.

From "Cameras as Relative Positional Encoding" (https://github.com/liruilong940607/prope,
``prope/torch.py``). Each image token's camera channels get a block-diagonal
transform: Q by ``P_iᵀ``, K and V by ``P_j⁻¹``, and the attention output by
``P_i``, where ``P = lift(K) @ w2c`` is camera ``i``'s image←world projection.
The score between two tokens then depends only on the relative projective
transform ``P_i P_j⁻¹``, so one attention pass covers every camera pair.

Head layout matches RayRoPE: the 32 channels of the backbone RoPE axis
``rope_axis`` are kept, and the other 96 are PRoPE as in the reference split:
half projective (12 blocks of 4), a quarter x-RoPE and a quarter y-RoPE on the
patch grid (base 100). Text tokens keep the backbone RoPE and are not
transformed. Poses come from ``rayrope.prepare_geometry`` (camera 0 is the
identity and camera 1 is at unit distance, as in the PRoPE NVS code).

This file is backbone-free: it works on ``[B, heads, txt+img, head_dim]``
tensors.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from .rayrope import HEAD_DIM, RAY_DIM, merge_channels, split_channels

PROJ_DIM = RAY_DIM // 2   # 48: 12 tiled 4×4 projections
XY_DIM = RAY_DIM // 4     # 24 each for x and y
FREQ_BASE = 100.0


def _xy_coeffs(geom: dict, device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """cos/sin ``[img, XY_DIM // 2]`` of the patch column and row, row-major per camera."""
    patches_x, patches_y, cameras = geom["patches_x"], geom["patches_y"], geom["num_cameras"]
    num_freqs = XY_DIM // 2
    freqs = FREQ_BASE ** (-torch.arange(num_freqs, device=device, dtype=torch.float32) / num_freqs)
    columns = torch.arange(patches_x, device=device, dtype=torch.float32).repeat(patches_y * cameras)
    rows = torch.arange(patches_y, device=device, dtype=torch.float32).repeat_interleave(patches_x)
    rows = rows.repeat(cameras)
    return tuple(
        (torch.cos(pos[:, None] * freqs), torch.sin(pos[:, None] * freqs)) for pos in (columns, rows)
    )


def _rope(feats: torch.Tensor, coeffs: tuple[torch.Tensor, torch.Tensor], inverse: bool) -> torch.Tensor:
    """Split-half RoPE (the reference's ``_rope_apply_coeffs``)."""
    cos, sin = (c.to(feats.dtype) for c in coeffs)
    x, y = feats.chunk(2, dim=-1)
    if inverse:
        return torch.cat((cos * x - sin * y, sin * x + cos * y), dim=-1)
    return torch.cat((cos * x + sin * y, -sin * x + cos * y), dim=-1)


def _projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply camera ``c``'s 4×4 ``matrix[:, c]`` to every 4-channel block of its tokens."""
    batch, heads, seqlen, dim = feats.shape
    cameras = matrix.shape[1]
    blocks = feats.reshape(batch, heads, cameras, seqlen // cameras, dim // 4, 4)
    return torch.einsum("bcij,bhcpkj->bhcpki", matrix.to(feats.dtype), blocks).reshape(feats.shape)


def _transform(feats: torch.Tensor, matrix: torch.Tensor, coeffs, inverse: bool) -> torch.Tensor:
    """Block-diagonal PRoPE transform of the 96 camera channels."""
    proj, x, y = feats.split((PROJ_DIM, XY_DIM, XY_DIM), dim=-1)
    coeffs_x, coeffs_y = coeffs
    return torch.cat(
        (_projmat(proj, matrix), _rope(x, coeffs_x, inverse), _rope(y, coeffs_y, inverse)), dim=-1
    )


def prope_attention(
    q_content: torch.Tensor,
    k_content: torch.Tensor,
    v: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    num_txt: int,
    geom: dict,
    q_extra: Optional[torch.Tensor] = None,
    k_extra: Optional[torch.Tensor] = None,
    rope_axis: int = 0,
) -> torch.Tensor:
    """Full attention with PRoPE on the image tokens.

    Same arguments as :func:`rayrope.rayrope_attention`, without depth: Q/K
    before (``*_content``) and after (``*_rope``) the backbone RoPE, all
    ``[B, heads, txt+img, HEAD_DIM]``, images packed in camera order.
    ``q_extra`` / ``k_extra`` are added after the transform and must be zero
    on text. Returns ``[B, txt+img, heads*HEAD_DIM]``.
    """
    dim = q_content.shape[-1]
    if dim != HEAD_DIM:
        raise RuntimeError(f"PRoPE expects head dim {HEAD_DIM}, got {dim}")
    img_len = geom["num_cameras"] * geom["patches_y"] * geom["patches_x"]
    if q_content.shape[2] - num_txt != img_len:
        raise RuntimeError(
            f"PRoPE image tokens {q_content.shape[2] - num_txt} != {geom['num_cameras']} cameras × "
            f"{geom['patches_y'] * geom['patches_x']} patches"
        )
    coeffs = _xy_coeffs(geom, q_content.device)
    projection, projection_inv = geom["projection"], geom["projection_inv"]

    def encode(rope_feats, content, matrix):
        kept, _ = split_channels(rope_feats[:, :, num_txt:], rope_axis)
        _, camera_channels = split_channels(content[:, :, num_txt:], rope_axis)
        return merge_channels(kept, _transform(camera_channels, matrix, coeffs, inverse=False), rope_axis)

    q_img = encode(q_rope, q_content, projection.transpose(-1, -2))
    k_img = encode(k_rope, k_content, projection_inv)
    v_img = encode(v, v, projection_inv)
    if q_extra is not None:
        q_img = q_img + q_extra[:, :, num_txt:]
    if k_extra is not None:
        k_img = k_img + k_extra[:, :, num_txt:]
    q = torch.cat((q_rope[:, :, :num_txt], q_img), dim=2)
    k = torch.cat((k_rope[:, :, :num_txt], k_img), dim=2)
    v = torch.cat((v[:, :, :num_txt], v_img), dim=2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    kept, camera_channels = split_channels(out[:, :, num_txt:], rope_axis)
    out_img = merge_channels(kept, _transform(camera_channels, projection, coeffs, inverse=True), rope_axis)
    out = torch.cat((out[:, :, :num_txt], out_img), dim=2)
    return rearrange(out, "b h s d -> b s (h d)")
