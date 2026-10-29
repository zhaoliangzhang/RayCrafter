"""FLUX 3 Image Klein backbone: weights, token packing, and a camera-aware forward.

Same functions as ``flux2.py``. FLUX 3 ships the FLUX.2 autoencoder unchanged
(identical ae.safetensors), so the VAE, the 16-px token patch, and the camera
tensors come from ``flux2.py``.

The DiT runs per-mode blocks (image / image_cond / text, each attending only
within its mode), then shared single blocks over all tokens. Here the views
are the ``image`` mode (timestep t, RoPE l = 0..V-1), the refs are
``image_cond`` (timestep 0, RoPE l = -1, -2, ..., as in FLUX 3 editing), and
there is no text, so the text stream never runs and stays frozen.

Camera modules sit in every block that sees image tokens. In the per-mode
blocks each mode uses only its own cameras (refs or views). RayRoPE, PRoPE and
UniRoPE keep the ``l`` RoPE axis (channels 96:128, which tells images apart in FLUX 3)
and replace ``t, h, w``. ``plucker_concat`` instead adds a zero-initialized
projection of the pixel ray map (``cameras/raymap.py``) to each mode's input
embedding (``ray_in``), as if the rays were concatenated to the 128 latent channels.

FLUX 3 code comes from https://github.com/black-forest-labs/flux-3-image-klein (see README).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as load_sft
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    import flux3.dit  # noqa: F401
except ImportError:
    # Fall back to the third_party/flux3 submodule when it is not pip-installed.
    _FLUX3_SRC = Path(__file__).parents[3] / "third_party" / "flux3" / "src"
    if not _FLUX3_SRC.is_dir():
        raise ImportError(
            "flux3 not found. `pip install --no-deps -e third_party/flux3` "
            "after `git submodule update --init` (see README)."
        )
    sys.path.insert(0, str(_FLUX3_SRC))

from flux3.autoencoder import latent_ids
from flux3.dit import DEPTH, HIDDEN, DiT, layernorm_modulation, rope_embed, timestep_embedding

from ..cameras.prope import prope_attention
from ..cameras.raymap import raymap_channels
from ..cameras.rayrope import DepthHead, subset_geometry
from ..cameras.scope import SightlineCoordinatePE, scope_qkv
from .flux2 import (  # noqa: F401
    DEPTH_ATTENTION, DEPTH_HEADS, PATCH, _hf_download, camera_inputs, load_vae, set_camera_context, unpack_views,
)

# flux3.pipeline.IMAGE_ROPE_SCALE: image RoPE (h, w) ids are the latent grid × 0.5.
IMAGE_ROPE_SCALE = 0.5
# RoPE axis (t, h, w, l) that RayRoPE / PRoPE keep: l numbers the images.
ROPE_AXIS = 3

VARIANTS = {
    # Undistilled base. The distilled flux-3-image-klein is a flow map (it also
    # takes the time to jump to) and is not supported.
    "flux.3-image-klein-base": {
        "repo_id": "black-forest-labs/flux-3-image-klein-base",
        "filename": "model.safetensors",
    },
}

# Weights that only touch text tokens. Without text they get no gradient
# (DDP runs with find_unused_parameters=False), so they are frozen.
_TEXT_PREFIXES = (
    "txt_in.", "txt_mode_blocks.", "early_stream_modulations.txt.", "single_stream_modulations.txt.",
)


def _norm_rope(x: torch.Tensor, scale: torch.Tensor, freqs: torch.Tensor, eps: float = 1e-6):
    """``flux3.dit.rmsnorm_rope`` (same ops), also returning the normalized
    input before RoPE. Returns ``(content, rotated)`` in ``x.dtype``."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        b, h, l, d = x.shape
        x_f = x.float()
        x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        x_r, freqs = x_norm.reshape(b, h, l, d // 2, 1, 2), freqs.float()
        rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
        return x_norm.to(x.dtype), rotated.reshape(b, h, l, d).to(x.dtype)


def _block_forward(
    block: nn.Module, x, freqs, mod, plucker, num_frames: int, geom, rope_type: str, apply_vo: bool = False
) -> torch.Tensor:
    """``flux3.dit.Block.forward`` on batched ``(B, L, D)`` tokens, with SCoPE
    (when ``plucker`` is given) and RayRoPE / PRoPE / UniRoPE (``rope_type``) for the
    cameras in ``geom``."""
    shift, scale, gate = mod
    x_mod = layernorm_modulation(x, scale, shift)
    split = lambda t: t.unflatten(-1, (block.heads, -1)).transpose(1, 2)  # B L (H D) -> B H L D
    q, k, v = split(block.q_proj(x_mod)), split(block.k_proj(x_mod)), split(block.v_proj(x_mod))
    if rope_type == "flux":
        q, k = block.norm(q, k, freqs)
        v = v.to(q.dtype)
        if plucker is not None:
            q, k, v = scope_qkv(block.scope_pe, q, k, v, 0, plucker, num_frames)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    else:
        q_content, q_rope = _norm_rope(q, block.norm.query_norm.scale, freqs)
        k_content, k_rope = _norm_rope(k, block.norm.key_norm.scale, freqs)
        v = v.to(q_rope.dtype)
        q_extra = k_extra = None
        if plucker is not None:
            # SCoPE shifts computed on zeros, then added after the camera RoPE.
            q_extra, k_extra, v = scope_qkv(
                block.scope_pe, torch.zeros_like(q_rope), torch.zeros_like(k_rope), v, 0, plucker, num_frames
            )
        if rope_type == "prope":
            attn = prope_attention(q_content, k_content, v, q_rope, k_rope, 0, geom, q_extra, k_extra, ROPE_AXIS)
        else:
            # The depth head reads the normalized, modulated QKV input.
            depth = getattr(block, DEPTH_HEADS[rope_type])(x_mod)
            attn = DEPTH_ATTENTION[rope_type](
                q_content, k_content, v, q_rope, k_rope, 0, depth, geom, q_extra, k_extra, ROPE_AXIS, apply_vo
            )
    gate_in, up = block.mlp_in(x_mod).chunk(2, dim=-1)
    return x + gate * (block.attn_out(attn) + block.mlp_out(F.silu(gate_in) * up))


class Flux3(DiT):
    """BFL's DiT (same parameters and checkpoint keys) with a batched forward for
    views + refs without text."""

    def __init__(self) -> None:
        super().__init__(flow_map=False)
        self.gradient_checkpointing = False
        self._camera: Optional[dict] = None
        self._rope_type = "flux"
        self._rope_apply_vo = False

    def _block(self, block: nn.Module, *args) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(_block_forward, block, *args, use_reentrant=False)
        return _block_forward(block, *args)

    def forward(self, x_views, view_ids, x_ref, ref_ids, timesteps) -> torch.Tensor:
        """Velocity of the view tokens ``(B, V*h*w, 128)``. Ids are ``(B, L, 4)``."""
        num_views, num_refs = x_views.shape[1], x_ref.shape[1]
        has_scope = hasattr(self.single_blocks[0], "scope_pe")
        has_raymap = hasattr(self, "ray_in")
        rope_type = self._rope_type
        if (has_scope or has_raymap or rope_type != "flux") and self._camera is None:
            raise RuntimeError("camera conditioning is installed but no cameras were set for this forward")
        plucker = self._camera["plucker"] if has_scope else None
        raymap = self._camera["raymap"] if has_raymap else None
        if has_raymap and raymap is None:
            raise RuntimeError("ray_in is installed but the cameras of this forward have no raymap")
        geom = self._camera["geom"] if rope_type != "flux" else None
        ref_rays = view_rays = ref_geom = view_geom = None
        frames = r_frames = v_frames = 0
        if self._camera is not None:
            # Cameras are ordered [ref..., views...]; every frame has the same token count.
            frames = self._camera["num_frames"]
            per_frame = (num_refs + num_views) // frames
            r_frames, v_frames = num_refs // per_frame, num_views // per_frame
        if plucker is not None:
            ref_rays, view_rays = plucker[:, :num_refs], plucker[:, num_refs:]
        if geom is not None:
            ref_geom, view_geom = subset_geometry(geom, 0, r_frames), subset_geometry(geom, r_frames, frames)

        vec_img = self.time_in(timestep_embedding(timesteps))
        vec_ref = self.time_in(timestep_embedding(torch.zeros_like(timesteps)))  # refs are clean
        mod = lambda m, vec: tuple(p.unsqueeze(1) for p in m(vec))
        early_img = mod(self.early_stream_modulations["image"], vec_img)
        early_ref = mod(self.early_stream_modulations["image_cond"], vec_ref)

        img = self.emb_in["image"](x_views)
        ref = self.emb_in["image_cond"](x_ref)
        if raymap is not None:
            # emb_in is bias-free, so this is emb_in on [latent | rays] with zero-initialized ray columns.
            img = img + self.ray_in["image"](raymap[:, num_refs:].to(x_views.dtype))
            ref = ref + self.ray_in["image_cond"](raymap[:, :num_refs].to(x_ref.dtype))
        freqs_img, freqs_ref = rope_embed(view_ids), rope_embed(ref_ids)
        for d in range(DEPTH):
            img = self._block(self.content_mode_blocks["image"][d], img, freqs_img, early_img,
                              view_rays, v_frames, view_geom, rope_type, self._rope_apply_vo)
            ref = self._block(self.content_mode_blocks["image_cond"][d], ref, freqs_ref, early_ref,
                              ref_rays, r_frames, ref_geom, rope_type, self._rope_apply_vo)

        # Single stream: [refs | views], the camera order. Position comes from the
        # RoPE ids, so the order does not change attention.
        h = torch.cat((ref, img), dim=1)
        freqs = torch.cat((freqs_ref, freqs_img), dim=2)
        single = tuple(
            torch.cat((r.expand(-1, num_refs, -1), i.expand(-1, num_views, -1)), dim=1)
            for r, i in zip(mod(self.single_stream_modulations["image_cond"], vec_ref),
                            mod(self.single_stream_modulations["image"], vec_img))
        )
        for block in self.single_blocks:
            h = self._block(block, h, freqs, single, plucker, frames, geom, rope_type, self._rope_apply_vo)
        return self.final_layer["image"](h[:, num_refs:], vec_img.unsqueeze(1))


def load_backbone(variant: str, path: Optional[str], device: torch.device) -> Flux3:
    v = VARIANTS[variant]
    with torch.device("meta"):
        model = Flux3().to(torch.bfloat16)
    sd = load_sft(path or _hf_download(v["repo_id"], v["filename"]), device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)
    for name, p in model.named_parameters():
        if name.startswith(_TEXT_PREFIXES):
            p.requires_grad_(False)
    return model.to(device)


# ── Token packing ────────────────────────────────────────────────────────────


def pack_views(
    z: torch.Tensor, l_start: int = 0, l_step: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, V, C, h, w)`` → tokens ``(B, V*h*w, C)`` and RoPE ids ``(B, V*h*w, 4)``.

    Ids are FLUX 3's float ``(t, h, w, l)``: ``t = 0``, ``(h, w)`` the latent
    grid × 0.5, row-major, and image ``i`` gets ``l = l_start + i * l_step``.
    """
    B, V, _, h, w = z.shape
    tokens = rearrange(z, "b v c hh ww -> b (v hh ww) c")
    ids = latent_ids(h, w, z.device).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    ids = ids.repeat(V, 1)
    ids[:, 3] = l_start + l_step * torch.arange(V, device=z.device).repeat_interleave(h * w)
    return tokens, ids.expand(B, -1, -1)


def ref_index(num_views: int, ref_t_scale: Optional[int]) -> Tuple[int, int]:
    """(start, step) of the ref RoPE l coordinates: -1, -2, ... as in FLUX 3
    editing. ``ref_t_scale`` is FLUX.2's ref offset and unused here."""
    return -1, -1


def run_transformer(transformer: nn.Module, x_ref, ref_ids, x_views, view_ids, t) -> torch.Tensor:
    """One forward of the (possibly DDP-wrapped) transformer; returns the view velocity."""
    return transformer(x_views, view_ids, x_ref, ref_ids, t)


# ── Camera conditioning ─────────────────────────────────────────────────────


def install_camera_attention(
    transformer: Flux3, scope: bool, rope_type: str, apply_vo: bool = False, raymap: bool = False,
    scope_kwargs: Optional[dict] = None,
) -> None:
    """Add camera modules to every block that sees image tokens.

    ``rope_type`` is ``flux``, ``rayrope``, ``prope`` (no parameters),
    ``unirope`` or ``unirope_anchor``. ``raymap`` adds ``ray_in``, the ray-map
    input projection of ``plucker_concat``. Module names (``scope_pe``,
    ``ray_in`` and the ``DEPTH_HEADS`` values) are checkpoint keys.
    """
    transformer._rope_type = rope_type
    transformer._rope_apply_vo = apply_vo
    ref = next(transformer.parameters())
    if raymap:
        transformer.ray_in = nn.ModuleDict({
            m: nn.Linear(raymap_channels(PATCH), HIDDEN, bias=False) for m in ("image", "image_cond")
        }).to(device=ref.device, dtype=ref.dtype)
        # Zero: step 0 is the pretrained model.
        for layer in transformer.ray_in.values():
            nn.init.zeros_(layer.weight)
    if not scope and rope_type not in DEPTH_HEADS:
        return
    blocks = [*transformer.content_mode_blocks["image"], *transformer.content_mode_blocks["image_cond"],
              *transformer.single_blocks]
    for block in blocks:
        if scope:
            block.scope_pe = SightlineCoordinatePE(HIDDEN, **(scope_kwargs or {}))
            block.scope_pe.to(device=ref.device, dtype=ref.dtype)
        if rope_type in DEPTH_HEADS:
            setattr(block, DEPTH_HEADS[rope_type], DepthHead(HIDDEN).to(device=ref.device, dtype=ref.dtype))


def enable_gradient_checkpointing(transformer: Flux3) -> None:
    """Checkpoint every block."""
    transformer.gradient_checkpointing = True
