"""FLUX 3 Image Klein backbone: weights, token packing, and a camera-aware forward.

Same functions as ``flux2.py``. FLUX 3 ships the FLUX.2 autoencoder unchanged
(identical ae.safetensors), so the VAE, the 16-px token patch, and the camera
tensors come from ``flux2.py``.

The DiT runs per-mode blocks (image / image_cond / text, each attending only
within its mode), then shared single blocks over all tokens. Here the views
are the ``image`` mode (timestep t, RoPE l = 0..V-1), the refs are
``image_cond`` (timestep 0, RoPE l = -1, -2, ..., as in FLUX 3 editing), and
there is no text, so the text stream never runs and stays frozen.

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

from ..cameras.scope import SightlineCoordinatePE, scope_qkv
from .flux2 import PATCH, _hf_download, camera_inputs, load_vae, set_camera_context, unpack_views  # noqa: F401

# flux3.pipeline.IMAGE_ROPE_SCALE: image RoPE (h, w) ids are the latent grid × 0.5.
IMAGE_ROPE_SCALE = 0.5

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


def _block_forward(block: nn.Module, x, freqs, mod, plucker, num_frames: int) -> torch.Tensor:
    """``flux3.dit.Block.forward`` on batched ``(B, L, D)`` tokens, with SCoPE
    added to Q/K/V after RoPE when ``plucker`` is given."""
    shift, scale, gate = mod
    x_mod = layernorm_modulation(x, scale, shift)
    split = lambda t: t.unflatten(-1, (block.heads, -1)).transpose(1, 2)  # B L (H D) -> B H L D
    q, k = block.norm(split(block.q_proj(x_mod)), split(block.k_proj(x_mod)), freqs)
    v = split(block.v_proj(x_mod)).to(q.dtype)
    if plucker is not None:
        q, k, v = scope_qkv(block.scope_pe, q, k, v, 0, plucker, num_frames)
    attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    gate_in, up = block.mlp_in(x_mod).chunk(2, dim=-1)
    return x + gate * (block.attn_out(attn) + block.mlp_out(F.silu(gate_in) * up))


class Flux3(DiT):
    """BFL's DiT (same parameters and checkpoint keys) with a batched forward for
    views + refs without text."""

    def __init__(self) -> None:
        super().__init__(flow_map=False)
        self.gradient_checkpointing = False
        self._camera: Optional[dict] = None

    def _block(self, block: nn.Module, *args) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(_block_forward, block, *args, use_reentrant=False)
        return _block_forward(block, *args)

    def forward(self, x_views, view_ids, x_ref, ref_ids, timesteps) -> torch.Tensor:
        """Velocity of the view tokens ``(B, V*h*w, 128)``. Ids are ``(B, L, 4)``."""
        num_views, num_refs = x_views.shape[1], x_ref.shape[1]
        has_scope = hasattr(self.single_blocks[0], "scope_pe")
        plucker = self._camera["plucker"] if self._camera is not None else None
        if has_scope and plucker is None:
            raise RuntimeError("SCoPE is installed but no cameras were set for this forward")
        if has_scope:
            # Rays are ordered [ref..., views...]; every frame has the same token count.
            frames = self._camera["num_frames"]
            per_frame = (num_refs + num_views) // frames
            ref_rays, view_rays = plucker[:, :num_refs], plucker[:, num_refs:]
            r_frames, v_frames = num_refs // per_frame, num_views // per_frame
        else:
            plucker = ref_rays = view_rays = None
            frames = r_frames = v_frames = 0

        vec_img = self.time_in(timestep_embedding(timesteps))
        vec_ref = self.time_in(timestep_embedding(torch.zeros_like(timesteps)))  # refs are clean
        mod = lambda m, vec: tuple(p.unsqueeze(1) for p in m(vec))
        early_img = mod(self.early_stream_modulations["image"], vec_img)
        early_ref = mod(self.early_stream_modulations["image_cond"], vec_ref)

        img = self.emb_in["image"](x_views)
        ref = self.emb_in["image_cond"](x_ref)
        freqs_img, freqs_ref = rope_embed(view_ids), rope_embed(ref_ids)
        for d in range(DEPTH):
            img = self._block(self.content_mode_blocks["image"][d], img, freqs_img, early_img, view_rays, v_frames)
            ref = self._block(self.content_mode_blocks["image_cond"][d], ref, freqs_ref, early_ref, ref_rays, r_frames)

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
            h = self._block(block, h, freqs, single, plucker, frames)
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


def install_camera_attention(transformer: Flux3, scope: bool, rayrope: bool) -> None:
    """Add a zero-init SCoPE module to every block that sees image tokens.

    The module name (``scope_pe``) is a checkpoint key.
    """
    if rayrope:
        raise NotImplementedError("RayRoPE is FLUX.2-only; use rope_type: flux with backbone: flux3")
    if not scope:
        return
    ref = next(transformer.parameters())
    blocks = [*transformer.content_mode_blocks["image"], *transformer.content_mode_blocks["image_cond"],
              *transformer.single_blocks]
    for block in blocks:
        block.scope_pe = SightlineCoordinatePE(HIDDEN).to(device=ref.device, dtype=ref.dtype)


def enable_gradient_checkpointing(transformer: Flux3) -> None:
    """Checkpoint every block."""
    transformer.gradient_checkpointing = True
