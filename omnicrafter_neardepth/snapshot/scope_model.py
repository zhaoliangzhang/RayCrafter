"""SCoPE Q/K-only camera injection for the existing WAN panorama backbone.

Based on TencentARC/SCoPE, commit 2fa984036004fb564fc2b0e888ce2bd95ab03735.
The official Normalize-Gate-Inject module is vendored, unchanged. We use its
Q/K-only option, with FP32 RMSNorm reductions for bf16 stability. No RayRoPE,
frame-average V residual, or concatenated camera channels are used.
"""
import sys
import types
import torch
from torch import nn
from einops import rearrange
from common import CODE, REPO
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CODE / 'vendor'))
from scope.encoding import SightlineCoordinatePE
from diffsynth.models.wan_video_dit import rope_apply, modulate, sinusoidal_embedding_1d


def stable_rms_forward(self, x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            * self.weight.float()).to(x.dtype)


def scope_attention_forward(self, x, freqs, scope_rays):
    q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
    v = self.v(x)
    q, k = self.scope_pe.apply_to_qk(q, k, scope_rays)
    return self.o(self.attn(q, k, v))


def scope_block_forward(self, x, context, t_mod, freqs, scope_rays):
    has_seq = t_mod.ndim == 4
    chunk_dim = 2 if has_seq else 1
    mods = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        mods = tuple(m.squeeze(2) for m in mods)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, scope_rays))
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    return self.gate(x, gate_mlp, self.ffn(input_x))


def install_scope(dit):
    if tuple(dit.patch_size) != (1, 2, 2):
        raise ValueError(f'Unsupported patch grid: {dit.patch_size}')
    for block in dit.blocks:
        attention = block.self_attn
        if hasattr(attention, 'scope_pe'):
            raise RuntimeError('SCoPE already installed')
        parameter = next(attention.parameters())
        pe = SightlineCoordinatePE(dim=dit.dim, plucker_init='zero', plucker_mlp_hidden=0,
            plucker_scale=1.0, enable_cam_residual=False, scale_gate_hidden=0,
            tied_init=True, log_scale_aug_prob=0.0, plucker_eps=1e-6)
        pe.norm_pe_q.forward = types.MethodType(stable_rms_forward, pe.norm_pe_q)
        pe.norm_pe_k.forward = types.MethodType(stable_rms_forward, pe.norm_pe_k)
        attention.scope_pe = pe.to(device=parameter.device, dtype=parameter.dtype)
        attention.forward = types.MethodType(scope_attention_forward, attention)
        block.forward = types.MethodType(scope_block_forward, block)


def model_fn_scope(dit, latents=None, timestep=None, context=None, clip_feature=None,
                   y=None, scope_rays=None, raymap=None, use_gradient_checkpointing=False,
                   use_gradient_checkpointing_offload=False, **kwargs):
    if raymap is not None:
        raise ValueError('SCoPE path must not receive channel-concatenated raymaps')
    if scope_rays is None or scope_rays.ndim != 3 or scope_rays.shape[-1] != 6:
        raise ValueError('Expected explicit [batch, tokens, 6] Pluecker rays')
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
    if tuple(scope_rays.shape) != (x.shape[0], f*h*w, 6):
        raise ValueError(f'Ray/token mismatch: {scope_rays.shape}, grid={(f,h,w)}')
    scope_rays = scope_rays.to(device=x.device, dtype=torch.float32)
    # Preserve OmniCrafter's neutral temporal frequencies and spatial RoPE.
    f_freqs = torch.ones_like(dit.freqs[0][:f])
    freqs = torch.cat([
        f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f*h*w, 1, -1).to(x.device)
    for block in dit.blocks:
        if use_gradient_checkpointing:
            # Explicit immutable camera input is reused during recomputation;
            # no module-global mutable ray state can leak between batches.
            x = torch.utils.checkpoint.checkpoint(block, x, context, t_mod, freqs, scope_rays, use_reentrant=False)
        else:
            x = block(x, context, t_mod, freqs, scope_rays)
    return dit.unpatchify(dit.head(x, t), (f, h, w))
