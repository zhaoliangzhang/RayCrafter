"""FLUX.2 backbone: weights, VAE, token packing, and camera-aware block forwards.

Everything that knows about FLUX.2 internals lives in this file. To add a new
backbone, provide the same functions (``load_backbone``, ``load_vae``,
``pack_views``, ``unpack_views``, ``ref_index``, ``run_transformer``,
``camera_inputs``, ``install_camera_attention``, ``set_camera_context``,
``enable_gradient_checkpointing``, ``PATCH``) for it and register it in
``model.BACKBONES``.

FLUX.2 code comes from https://github.com/black-forest-labs/flux2 (see README).
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

try:
    import flux2.model  # noqa: F401  (a bare `import flux2` also matches any flux2/ dir on the path)
except ImportError:
    # Fall back to the third_party/flux2 submodule when it is not pip-installed.
    _FLUX2_SRC = Path(__file__).parents[3] / "third_party" / "flux2" / "src"
    if not _FLUX2_SRC.is_dir():
        raise ImportError(
            "flux2 not found. `pip install --no-deps -e third_party/flux2` "
            "after `git submodule update --init` (see README)."
        )
    sys.path.insert(0, str(_FLUX2_SRC))

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.model import Flux2, Flux2Params, Klein4BParams, Klein9BParams, apply_rope
from flux2.sampling import prc_img

from ..cameras.prope import prope_attention
from ..cameras.rayrope import DepthHead, prepare_geometry, rayrope_attention
from ..cameras.scope import SightlineCoordinatePE, plucker_tokens, scope_qkv
from ..cameras.unirope import prepare_unirope, unirope_anchor_attention, unirope_attention

# Depth-head attribute (a checkpoint key) per rope_type that predicts depth.
DEPTH_HEADS = {"rayrope": "rayrope_depth", "unirope": "unirope_depth", "unirope_anchor": "unirope_depth"}
# Attention of each depth-predicting rope_type.
DEPTH_ATTENTION = {
    "rayrope": rayrope_attention, "unirope": unirope_attention, "unirope_anchor": unirope_anchor_attention,
}

# VAE: 16× downsampling, 128 latent channels → one 128-ch token per 16×16 patch.
PATCH = 16

VARIANTS = {
    "flux.2-klein-4b": {
        "params": Klein4BParams(),
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
    },
    "flux.2-klein-9b": {
        "params": Klein9BParams(),
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
    },
    # Undistilled base models: same architecture as the distilled klein models.
    "flux.2-klein-base-4b": {
        "params": Klein4BParams(),
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "filename": "flux-2-klein-base-4b.safetensors",
    },
    "flux.2-klein-base-9b": {
        "params": Klein9BParams(),
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "filename": "flux-2-klein-base-9b.safetensors",
    },
    "flux.2-dev": {
        "params": Flux2Params(),
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
    },
}
AE_REPO_ID, AE_FILENAME = "black-forest-labs/FLUX.2-dev", "ae.safetensors"


def _hf_download(repo_id: str, filename: str) -> str:
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)


def load_backbone(variant: str, path: Optional[str], device: torch.device) -> Flux2:
    params = VARIANTS[variant]["params"]
    with torch.device("meta"):
        model = Flux2(params).to(torch.bfloat16)
    sd = load_sft(path or _hf_download(VARIANTS[variant]["repo_id"], VARIANTS[variant]["filename"]),
                  device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device)


def load_vae(path: Optional[str], device: torch.device) -> AutoEncoder:
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    sd = load_sft(path or _hf_download(AE_REPO_ID, AE_FILENAME), device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


# ── Token packing ────────────────────────────────────────────────────────────


def pack_views(
    z: torch.Tensor, t_start: int = 0, t_step: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, V, C, h, w)`` → tokens ``(B, V*h*w, C)`` and 4-D RoPE ids ``(B, V*h*w, 4)``.

    Image ``i`` gets RoPE ``t = t_start + i * t_step``; ``(h, w)`` ids are the
    latent grid, row-major.
    """
    B, V = z.shape[:2]
    tokens, ids = [], []
    for i in range(V):
        t_coord = torch.tensor([t_start + i * t_step], dtype=torch.long)
        packed = [prc_img(z[b, i], t_coord=t_coord) for b in range(B)]
        tokens.append(torch.stack([p[0] for p in packed]))
        ids.append(torch.stack([p[1] for p in packed]))
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=1)


def unpack_views(tokens: torch.Tensor, num_views: int, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`pack_views` for the token values: ``(B, V*h*w, C)`` → ``(B, V, C, h, w)``."""
    return rearrange(tokens, "b (v hh ww) c -> b v c hh ww", v=num_views, hh=h, ww=w)


def empty_text(batch_size: int, ctx_dim: int, device: torch.device, dtype: torch.dtype):
    """No text: zero-length context, so double blocks reduce to image self-attention."""
    txt = torch.zeros(batch_size, 0, ctx_dim, device=device, dtype=dtype)
    txt_ids = torch.zeros(batch_size, 0, 4, device=device, dtype=torch.long)
    return txt, txt_ids


def ref_index(num_views: int, ref_t_scale: Optional[int]) -> Tuple[int, int]:
    """(start, step) of the ref RoPE t coordinates; views use t = 0..V-1."""
    return (int(num_views), 1) if ref_t_scale is None else (int(ref_t_scale), int(ref_t_scale))


def run_transformer(transformer: nn.Module, x_ref, ref_ids, x_views, view_ids, t) -> torch.Tensor:
    """One forward of the (possibly DDP-wrapped) transformer over ``[refs | views]``
    without text; returns the view velocity."""
    txt, txt_ids = empty_text(x_views.shape[0], _unwrap(transformer).txt_in.in_features,
                              x_views.device, x_views.dtype)
    pred = transformer(
        x=torch.cat([x_ref, x_views], dim=1),
        x_ids=torch.cat([ref_ids, view_ids], dim=1),
        timesteps=t,
        ctx=txt,
        ctx_ids=txt_ids,
        guidance=None,
    )
    return pred[:, x_ref.shape[1]:]


# ── Camera conditioning inside FLUX blocks ──────────────────────────────────


def _unwrap(transformer: nn.Module) -> nn.Module:
    return transformer.module if hasattr(transformer, "module") else transformer


def camera_inputs(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    image_hw: Tuple[int, int],
    latent_hw: Tuple[int, int],
    scope: bool,
    rope_type: str,
) -> dict:
    """Per-forward camera tensors, shared by every block."""
    geometry = {
        "flux": None, "rayrope": prepare_geometry, "prope": prepare_geometry,
        "unirope": prepare_unirope, "unirope_anchor": prepare_unirope,
    }
    with torch.no_grad():
        prepare = geometry[rope_type]
        return {
            "plucker": plucker_tokens(c2w, fxfycxcy, image_hw, latent_hw, PATCH) if scope else None,
            "geom": prepare(c2w, fxfycxcy, image_hw, latent_hw) if prepare else None,
            "num_frames": int(c2w.shape[1]),
        }


def set_camera_context(transformer: nn.Module, ctx: Optional[dict]) -> None:
    host = _unwrap(transformer)
    if hasattr(host, "_camera"):
        host._camera = ctx


def _attention(block: nn.Module, q, k, v, pe, num_txt: int, depth_tokens: Optional[torch.Tensor]) -> torch.Tensor:
    """FLUX attention with SCoPE and/or RayRoPE / PRoPE / UniRoPE. Returns ``[B, txt+img, heads*dim]``.

    ``depth_tokens`` are the image tokens the depth head reads (``None``
    without RayRoPE / UniRoPE).
    """
    ctx = block._camera_host._camera
    if ctx is None:
        raise RuntimeError("camera conditioning is installed but no cameras were set for this forward")
    q_rope, k_rope = apply_rope(q, k, pe)
    has_scope = hasattr(block, "scope_pe")
    rope_type = block._camera_host._rope_type

    if rope_type == "flux":
        q_rope, k_rope, v = scope_qkv(
            block.scope_pe, q_rope, k_rope, v, num_txt, ctx["plucker"], ctx["num_frames"]
        )
        out = F.scaled_dot_product_attention(q_rope, k_rope, v, is_causal=False)
        return rearrange(out, "b h n d -> b n (h d)")

    q_extra = k_extra = None
    if has_scope:
        # SCoPE shifts computed on zeros, then added after the camera RoPE.
        q_extra, k_extra, v = scope_qkv(
            block.scope_pe, torch.zeros_like(q_rope), torch.zeros_like(k_rope), v,
            num_txt, ctx["plucker"], ctx["num_frames"],
        )
    if rope_type == "prope":
        return prope_attention(q, k, v, q_rope, k_rope, num_txt, ctx["geom"], q_extra, k_extra)
    depth = getattr(block, DEPTH_HEADS[rope_type])(depth_tokens)
    return DEPTH_ATTENTION[rope_type](q, k, v, q_rope, k_rope, num_txt, depth, ctx["geom"], q_extra, k_extra)


# The depth head reads the normalized, modulated tokens that feed the QKV
# projection, as RayRoPE's depth head reads its pre-norm attention input.
# Recomputed here because the FLUX block does not return them.


def _double_forward(self, img, txt, pe, pe_ctx, mod_img, mod_txt, num_ref_tokens):
    q, k, v, pe_full, num_txt, mods = self._prepare_qkv(img, txt, pe, pe_ctx, mod_img, mod_txt)
    depth_tokens = None
    if self._camera_host._rope_type in DEPTH_HEADS:
        shift, scale, _ = mod_img[0]
        depth_tokens = (1 + scale) * self.img_norm1(img) + shift
    attn = _attention(self, q, k, v, pe_full, num_txt, depth_tokens)
    img, txt = self._apply_residuals(img, txt, attn[:, num_txt:], attn[:, :num_txt], mods)
    return img, txt, {}


def _single_forward(self, x, pe, mod, num_txt_tokens, num_ref_tokens):
    q, k, v, mlp, mod_gate = self._qkv(x, mod)
    depth_tokens = None
    if self._camera_host._rope_type in DEPTH_HEADS:
        shift, scale, _ = mod
        depth_tokens = ((1 + scale) * self.pre_norm(x) + shift)[:, num_txt_tokens:]
    attn = _attention(self, q, k, v, pe, num_txt_tokens, depth_tokens)
    return self._out(x, attn, mlp, mod_gate), {}


def install_camera_attention(transformer: Flux2, scope: bool, rope_type: str) -> None:
    """Add camera modules to every block and patch its attention.

    ``rope_type`` is ``flux`` (backbone RoPE only), ``rayrope``, ``prope`` (no
    parameters), ``unirope`` or ``unirope_anchor``. Module names (``scope_pe`` and the
    ``DEPTH_HEADS`` values) are checkpoint keys.
    """
    if not scope and rope_type == "flux":
        return
    ref = next(transformer.parameters())
    hidden = transformer.hidden_size
    if rope_type != "flux" and hidden // transformer.num_heads != 128:
        raise RuntimeError(f"{rope_type} needs 128-d heads, got {hidden // transformer.num_heads}")
    transformer._camera = None
    transformer._rope_type = rope_type

    def bind(block: nn.Module, forward) -> None:
        # object.__setattr__: a normal assignment would register the transformer
        # as a child of its own block and make .train() recurse forever.
        object.__setattr__(block, "_camera_host", transformer)
        if scope:
            block.scope_pe = SightlineCoordinatePE(hidden).to(device=ref.device, dtype=ref.dtype)
        if rope_type in DEPTH_HEADS:
            setattr(block, DEPTH_HEADS[rope_type], DepthHead(hidden).to(device=ref.device, dtype=ref.dtype))
        block.forward_kv_extract = forward.__get__(block, block.__class__)

    for block in transformer.double_blocks:
        bind(block, _double_forward)
    for block in transformer.single_blocks:
        bind(block, _single_forward)


def enable_gradient_checkpointing(transformer: Flux2) -> None:
    """Checkpoint every block. ``Flux2.forward`` calls ``block.forward_kv_extract``,
    so wrap that per instance (after :func:`install_camera_attention`)."""
    from torch.utils.checkpoint import checkpoint

    blocks = list(transformer.double_blocks) + list(transformer.single_blocks)
    for block in blocks:
        def wrapped(*args, _fn=block.forward_kv_extract, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_fn, *args, use_reentrant=False, **kwargs)
            return _fn(*args, **kwargs)
        block.forward_kv_extract = wrapped
