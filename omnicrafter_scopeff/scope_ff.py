"""scope-ff camera injection for the OmniCrafter Wan2.1 DiT.

Same place as the near-depth run's ``snapshot/scope_model.py``: Q/K of every
self-attention, after the Q/K norm and RoPE. The module is RayCrafter's
``SightlineCoordinatePE`` as ``configs/flux3_mix_scope_ff.yaml`` uses it instead
of the vendored TencentARC one: a 64-wide MLP over the 7 released inputs plus
sin/cos direction features (``dir_freqs``) and the patch's angular size
(``patch_size``), ``plucker_eps`` 1e-3, the log-scale gate shift in training and
no V residual. No camera channels are concatenated to the latents.
"""
import types

import torch
from einops import rearrange

import scopeff_paths  # noqa: F401
from diffsynth.models.wan_video_dit import rope_apply, sinusoidal_embedding_1d
from raycrafter.cameras.scope import SightlineCoordinatePE
from scope_model import scope_block_forward


class WanScopePE(SightlineCoordinatePE):
    """Keeps the Fourier frequencies (``dir_omega``) in fp32 through ``.to(bf16)``
    and DeepSpeed's ``module.bfloat16()``, as in flux3's fp32 module; only the
    learned weights follow the DiT's dtype."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omega_fp32 = self.dir_omega.clone() if self.dir_freqs else None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self._omega_fp32 is not None:
            self._buffers['dir_omega'] = self._omega_fp32.to(self._buffers['dir_omega'].device)
        return self


def scope_ff_attention_forward(self, x, freqs, scope_rays):
    q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
    v = self.v(x)
    # num_frames only sizes the V residual, which scope-ff does not use.
    q, k, _ = self.scope_pe(q, k, scope_rays, num_frames=1)
    return self.o(self.attn(q, k, v))


def install_scope_ff(dit, mlp_hidden=64, dir_freqs=8, patch_size=True):
    if tuple(dit.patch_size) != (1, 2, 2):
        raise ValueError(f'Unsupported patch grid: {dit.patch_size}')
    for block in dit.blocks:
        attention = block.self_attn
        if hasattr(attention, 'scope_pe'):
            raise RuntimeError('SCoPE already installed')
        parameter = next(attention.parameters())
        pe = WanScopePE(dit.dim, mlp_hidden=mlp_hidden, dir_freqs=dir_freqs, patch_size=patch_size)
        attention.scope_pe = pe.to(device=parameter.device, dtype=parameter.dtype)
        attention.forward = types.MethodType(scope_ff_attention_forward, attention)
        block.forward = types.MethodType(scope_block_forward, block)


def ray_channels(dit):
    return 6 + int(dit.blocks[0].self_attn.scope_pe.patch_size)


def model_fn_scope_ff(dit, latents=None, timestep=None, context=None, clip_feature=None,
                      y=None, scope_rays=None, raymap=None, use_gradient_checkpointing=False,
                      use_gradient_checkpointing_offload=False, **kwargs):
    """``snapshot/scope_model.model_fn_scope`` with rays of 6 + patch_size channels."""
    if raymap is not None:
        raise ValueError('SCoPE path must not receive channel-concatenated raymaps')
    channels = ray_channels(dit)
    if scope_rays is None or scope_rays.ndim != 3 or scope_rays.shape[-1] != channels:
        raise ValueError(f'Expected explicit [batch, tokens, {channels}] Pluecker rays')
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
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    if tuple(scope_rays.shape) != (x.shape[0], f*h*w, channels):
        raise ValueError(f'Ray/token mismatch: {scope_rays.shape}, grid={(f,h,w)}')
    scope_rays = scope_rays.to(device=x.device, dtype=torch.float32)
    # OmniCrafter's neutral temporal frequencies and spatial RoPE.
    f_freqs = torch.ones_like(dit.freqs[0][:f])
    freqs = torch.cat([
        f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f*h*w, 1, -1).to(x.device)
    for block in dit.blocks:
        if use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(block, x, context, t_mod, freqs, scope_rays, use_reentrant=False)
        else:
            x = block(x, context, t_mod, freqs, scope_rays)
    return dit.unpatchify(dit.head(x, t), (f, h, w))
