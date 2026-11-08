"""Scheduled, exactly bypassable camera adapters on the original concat WAN."""
import types
import torch
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import bootstrap
from config import VARIANT
from contract import CAMERA_CONTRACT
from diffsynth.models.wan_video_dit import SelfAttention, rope_apply, sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video import model_fn_wan_video
from vendor.raycrafter_camera.scope import SightlineCoordinatePE
from vendor.raycrafter_camera.rayrope import DepthHead
from vendor.raycrafter_camera.unirope import unirope_attention

def attention_forward(self, x, payload):
    if torch.is_tensor(payload):
        return SelfAttention.forward(self, x, payload)
    freqs, cameras, strength = payload
    if strength == 0.:
        return SelfAttention.forward(self, x, freqs)
    q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(x)), self.v(x)
    if VARIANT == 'scope_ff':
        q_base, k_base = rope_apply(q,freqs,self.num_heads), rope_apply(k,freqs,self.num_heads)
        dq, dk, unused = self.scope_pe(torch.zeros_like(q),torch.zeros_like(k),cameras['plucker'],5)
        if unused is not None:
            raise RuntimeError('SCoPE V residual is disabled')
        return self.o(self.attn(q_base + strength*dq, k_base + strength*dk, v))
    split = lambda t: rearrange(t,'b s (h d) -> b h s d',h=self.num_heads)
    qs,ks,vs = map(split,(q,k,v))
    new = unirope_attention(qs,ks,vs,qs,ks,0,self.unirope_depth(x),cameras['geom'],apply_vo=True)
    if strength == 1.:
        return self.o(new)
    baseline = self.attn(rope_apply(q,freqs,self.num_heads),rope_apply(k,freqs,self.num_heads),v)
    return self.o((1.-strength)*baseline + strength*new)

def batched_head_forward(self, x, t):
    if t.ndim == 2:
        t = t[:,None,:]
    shift,scale = (self.modulation[None].to(t)+t.unsqueeze(2)).chunk(2,dim=2)
    return self.head(self.norm(x)*(1+scale.squeeze(2))+shift.squeeze(2))

def install_camera_attention(dit):
    if dit.in_dim != 420 or tuple(dit.patch_size) != (1,2,2):
        raise ValueError('Expected the original 420-channel concat architecture')
    for block in dit.blocks:
        a = block.self_attn
        if a.head_dim != 128 or hasattr(a,'scope_pe') or hasattr(a,'unirope_depth'):
            raise ValueError('Expected unmodified WAN attention')
        parameter = next(a.parameters())
        if VARIANT == 'scope_ff':
            a.scope_pe = SightlineCoordinatePE(dit.dim,mlp_hidden=64,dir_freqs=8,patch_size=True).to(parameter)
        else:
            a.unirope_depth = DepthHead(dit.dim,init_depth=0.,init_sigma=3.).to(parameter)
        a.forward = types.MethodType(attention_forward,a)
    dit.head.forward = types.MethodType(batched_head_forward,dit.head)
    dit.camera_contract = dict(CAMERA_CONTRACT)
    dit.injection_strength = 0.

def model_fn(dit, latents=None, timestep=None, context=None, clip_feature=None, y=None,
             camera_context=None, use_gradient_checkpointing=False,
             use_gradient_checkpointing_offload=False, **kwargs):
    strength = float(dit.injection_strength)
    if not 0. <= strength <= 1. or camera_context is None or use_gradient_checkpointing_offload:
        raise ValueError('Invalid gate/context or unsupported activation offload')
    raymap = camera_context['raymap'].to(latents)
    if strength == 0.:
        # Exact frozen baseline implementation, including original operation
        # order, spatial RoPE and attention kernel. New modules are not called.
        return model_fn_wan_video(dit,latents,timestep,context,clip_feature,y,raymap,
                                  use_gradient_checkpointing,False)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim,timestep))
    t_mod = dit.time_projection(t).unflatten(1,(6,dit.dim))
    context = dit.text_embedding(context)
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x]*context.shape[0],dim=0)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x,y],dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        context = torch.cat([dit.img_emb(clip_feature),context],dim=1)
    x = dit.patchify(torch.cat([x,raymap],dim=1))
    f,h,w = x.shape[2:]
    if (f,h,w) != (5,*camera_context['latent_hw']):
        raise ValueError('Camera/token layout mismatch')
    x = rearrange(x,'b c f h w -> b (f h w) c').contiguous()
    freqs = torch.cat([
        torch.ones_like(dit.freqs[0][:f]).view(f,1,1,-1).expand(f,h,w,-1),
        dit.freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),
        dit.freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1),
    ],dim=-1).reshape(f*h*w,1,-1).to(x.device)
    payload = (freqs,camera_context,strength)  # Immutable gate for checkpoint recomputation.
    for block in dit.blocks:
        x = checkpoint(block,x,context,t_mod,payload,use_reentrant=False) if use_gradient_checkpointing else block(x,context,t_mod,payload)
    return dit.unpatchify(dit.head(x,t),(f,h,w))

def optimizer_groups(dit, main_lr=1e-5, weight_decay=.01):
    inherited, new = [], []
    for name,p in dit.named_parameters():
        if p.requires_grad:
            (new if '.scope_pe.' in name or '.unirope_depth.' in name else inherited).append(p)
    if not inherited or not new:
        raise ValueError('Expected inherited LoRA/patch and new camera parameters')
    return [dict(params=inherited,lr=main_lr,weight_decay=weight_decay,group_name='inherited_concat_lora_patch'),
            dict(params=new,lr=main_lr if VARIANT=='scope_ff' else 4e-4,
                 weight_decay=weight_decay if VARIANT=='scope_ff' else .5,group_name=VARIANT)]

def depth_statistics(dit):
    if VARIANT != 'unirope':
        return {}
    bias = torch.stack([b.self_attn.unirope_depth.bias.float() for b in dit.blocks]).mean(0)
    return {'depth/log_depth_bias':bias[0].item(),'depth/sigma_bias':bias[1].item()}
