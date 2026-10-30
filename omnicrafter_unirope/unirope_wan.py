"""UniRoPE camera encoding for the OmniCrafter Wan2.1 DiT.

Same place as ``omnicrafter_scopeff/scope_ff.py``: every self-attention, Q/K after
their norms. The encoding is RayCrafter's ``unirope_attention`` as
``configs/flux3_mix_unirope_v3.yaml`` uses it: it takes the whole 128-d head on
every token, so Wan's RoPE is not applied (images differ by their camera center),
attention runs once per camera (5 passes per layer), and V and the attention output
are rotated too. Each block has a zero-initialized ``DepthHead`` reading the
modulated attention input. No camera channels are concatenated to the latents.
"""
import types

import torch
from einops import rearrange

import unirope_paths  # noqa: F401
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from raycrafter.cameras.rayrope import HEAD_DIM, DepthHead
from raycrafter.cameras.unirope import unirope_attention
from scope_model import scope_block_forward


def unirope_attention_forward(self, x, freqs, geom):
    """``x`` is the modulated block input; ``freqs`` is unused (``None``), for
    ``scope_block_forward``'s call signature."""
    q = rearrange(self.norm_q(self.q(x)), 'b s (n d) -> b n s d', n=self.num_heads)
    k = rearrange(self.norm_k(self.k(x)), 'b s (n d) -> b n s d', n=self.num_heads)
    v = rearrange(self.v(x), 'b s (n d) -> b n s d', n=self.num_heads)
    # No text tokens and no backbone RoPE, so Q and K also fill the "after RoPE" arguments.
    out = unirope_attention(q, k, v, q, k, 0, self.unirope_depth(x), geom, apply_vo=self.unirope_apply_vo)
    return self.o(out)


def install_unirope(dit, apply_vo=True):
    if tuple(dit.patch_size) != (1, 2, 2):
        raise ValueError(f'Unsupported patch grid: {dit.patch_size}')
    for block in dit.blocks:
        attention = block.self_attn
        if hasattr(attention, 'unirope_depth'):
            raise RuntimeError('UniRoPE already installed')
        if attention.head_dim != HEAD_DIM:
            raise ValueError(f'UniRoPE needs {HEAD_DIM}-d heads, got {attention.head_dim}')
        parameter = next(attention.parameters())
        attention.unirope_depth = DepthHead(dit.dim).to(device=parameter.device, dtype=parameter.dtype)
        attention.unirope_apply_vo = apply_vo
        attention.forward = types.MethodType(unirope_attention_forward, attention)
        block.forward = types.MethodType(scope_block_forward, block)


def unirope_geometry(c2w, w2c, directions, patch_angle, device):
    """The ``geom`` ``unirope_attention`` reads, in fp32 whatever the DiT's dtype."""
    geom = {'c2w': c2w, 'w2c': w2c, 'directions': directions}
    if patch_angle is not None:
        geom['patch_angle'] = patch_angle
    return {k: v.to(device=device, dtype=torch.float32) for k, v in geom.items()}


def model_fn_unirope(dit, latents=None, timestep=None, context=None, clip_feature=None, y=None,
                     unirope_c2w=None, unirope_w2c=None, unirope_directions=None, unirope_patch_angle=None,
                     scope_rays=None, raymap=None, use_gradient_checkpointing=False,
                     use_gradient_checkpointing_offload=False, **kwargs):
    """``scope_ff.model_fn_scope_ff`` with UniRoPE geometry in place of the rays and Wan's RoPE."""
    if raymap is not None or scope_rays is not None:
        raise ValueError('UniRoPE is the only camera input: no raymap, no scope_rays')
    if unirope_c2w is None or unirope_w2c is None or unirope_directions is None:
        raise ValueError('Expected UniRoPE geometry')
    if use_gradient_checkpointing_offload:
        raise ValueError('CPU activation offload is disabled for this experiment')
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents
    if x.shape[0] != context.shape[0]:
        raise ValueError('Batch dimension mismatch')
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if x.shape[1] != dit.in_dim:
        raise ValueError(f'Image-only channels {x.shape[1]} != DiT in_dim={dit.in_dim}')
    if clip_feature is not None and dit.require_clip_embedding:
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    # Frames in camera order [ref, ref, query x3], each row-major: the order of the geometry's rays.
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    if tuple(unirope_directions.shape[:3]) != (x.shape[0], f, h*w):
        raise ValueError(f'Geometry/token mismatch: {unirope_directions.shape}, grid={(f,h,w)}')
    geom = unirope_geometry(unirope_c2w, unirope_w2c, unirope_directions, unirope_patch_angle, x.device)
    for block in dit.blocks:
        if use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(block, x, context, t_mod, None, geom, use_reentrant=False)
        else:
            x = block(x, context, t_mod, None, geom)
    return dit.unpatchify(dit.head(x, t), (f, h, w))
