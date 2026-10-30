"""OmniCrafter training module with scope-ff, and the RayCrafter-batch adapter.

``ScopeFFTrainingModule`` is the near-depth run's ``MixedTrainingModule`` (same
pipeline units, flow-matching loss on the 3 query latents, batched head) with
scope-ff installed instead of the plain SCoPE module. ``to_omnicrafter`` turns
one batch of RayCrafter's ``dl3dv_points`` / panorama loaders into the set dict
``encode_set`` reads, with the scope-ff rays flux3_mix_scope_ff would build for it.
"""
import numpy as np
import torch
from PIL import Image

import scopeff_paths  # noqa: F401
from mixed_model import MixedTrainingModule, install_batch_head, target_loss
from model_training.train import WanTrainingModule
from raycrafter.cameras.scope import plucker_tokens
from scope_ff import install_scope_ff, model_fn_scope_ff

# Pixels per DiT token: the Wan VAE downsamples 8x, the (1, 2, 2) patchify 2x more.
PATCH = 16


class ScopeFFTrainingModule(MixedTrainingModule):
    def __init__(self, *args, scope_kwargs, **kwargs):
        # Skip MixedTrainingModule.__init__, which installs the plain SCoPE module.
        WanTrainingModule.__init__(self, *args, **kwargs)
        # 'scope' routes encode_set's explicit [tokens, C] rays to the model function.
        self.architecture = 'scope'
        assert self.pipe.dit.in_dim == 36
        install_batch_head(self.pipe.dit)
        install_scope_ff(self.pipe.dit, **scope_kwargs)
        self.pipe.model_fn = model_fn_scope_ff
        self.last_batch_shapes = None

    @torch.no_grad()
    def val_loss(self, data, t_index, generator):
        """``forward`` at a fixed timestep and seeded noise, without the log-scale
        gate shift (eval mode) and without activation checkpointing."""
        dit = self.pipe.dit
        training = dit.training
        dit.eval()
        try:
            packed = self.encode_set(data)
            clean = packed.pop('input_latents')
            t = self.pipe.scheduler.timesteps[t_index:t_index+1].to(self.pipe.device, dtype=self.pipe.torch_dtype)
            noise = torch.randn(clean.shape, generator=generator).to(clean)
            noisy = self.pipe.scheduler.add_noise(clean, noise, t)
            target = self.pipe.scheduler.training_target(clean, noise, t)
            weight = self.pipe.scheduler.training_weight(t)
            prediction = self.pipe.model_fn(dit=dit, latents=noisy, timestep=t, **packed,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
            return target_loss(prediction, target, weight)
        finally:
            dit.train(training)


def to_pil(image):
    """``[3, H, W]`` in [-1, 1] → PIL. Exact for the loaders' uint8 sources."""
    pixels = ((image.float() + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def sort_refs_by_distance(ref, c2w, fxfycxcy):
    """RayCrafterModel._sort_refs for one set: refs near→far by mean camera-center
    distance to the queries. ``ref`` is ``[R, 3, H, W]``, cameras ``[1, R+V, ...]``."""
    count = ref.shape[0]
    centers = c2w[0, :, :3, 3].float()
    order = torch.cdist(centers[:count], centers[count:]).mean(dim=-1).argsort()
    cameras = torch.cat([order, torch.arange(count, c2w.shape[1])])
    return ref[order], c2w[:, cameras], fxfycxcy[:, cameras]


def to_omnicrafter(batch, patch_size=True, sort_refs=True, black_queries=False):
    """One RayCrafter batch (batch size 1) → ``encode_set``'s set dict.

    Frames are [ref, ref, query x3]. ``scope_rays`` is ``plucker_tokens`` on the
    loaders' near-depth-scaled c2w, relative to the first (nearest) ref, exactly
    flux3_mix_scope_ff's rays. ``black_queries`` turns the set into NVS.
    """
    if batch['ref'].shape[0] != 1:
        raise ValueError('OmniCrafter trains one set per GPU')
    ref, src, tgt = batch['ref'][0], batch['src'][0], batch['tgt'][0]
    c2w, fxfycxcy = batch['camera_c2w'], batch['camera_fxfycxcy']
    if ref.shape[0] != 2 or src.shape[0] != 3 or c2w.shape[1] != 5:
        raise ValueError('Need two references and three queries')
    if sort_refs:
        ref, c2w, fxfycxcy = sort_refs_by_distance(ref, c2w, fxfycxcy)
    if black_queries:
        src = torch.full_like(src, -1.0)
    height, width = ref.shape[-2:]
    if height % PATCH or width % PATCH:
        raise ValueError(f'{height}x{width} is not a whole number of {PATCH}px tokens')
    camera_model = batch.get('camera_model', 'pinhole')
    rays = plucker_tokens(c2w, fxfycxcy, (height, width), (height // PATCH, width // PATCH), PATCH,
                          camera_model=camera_model, patch_size=patch_size)
    refs = [to_pil(image) for image in ref]
    return {'input_images': refs, 'conditioning_images': [to_pil(image) for image in src],
            'target_images': refs + [to_pil(image) for image in tgt], 'prompt': '',
            'scope_rays': rays[0].contiguous(), 'camera_model': camera_model, 'key': batch['key'][0]}


def black_fraction(data):
    """Fraction of the queries that are entirely black (NVS)."""
    return float(np.mean([not np.asarray(image).any() for image in data['conditioning_images']]))
