"""OmniCrafter training module with FrameCrafter's Plücker channel concat on Wan's own weights.

``ConcatWanTrainingModule`` takes the near-depth run's non-SCoPE path (``raymap`` through
``model_fn_wan_video``) and scope-ff's ``val_loss``. ``expand_patch_embedding`` grows Wan's
36-channel patch embedding to FrameCrafter's 420 with the new channels at zero, so step 0
is exactly Wan. ``to_concat_set`` turns one batch of RayCrafter's ``dl3dv_points`` / panorama
loaders into the set dict ``encode_set`` reads.
"""
from datetime import timedelta

import torch
import torch.distributed.distributed_c10d as c10d
from einops import rearrange
from torch import nn

import concat_wan_paths  # noqa: F401
from mixed_model import install_batch_head
from model_training.train import WanTrainingModule
from raycrafter.cameras.raymap import plucker_raymap
from scopeff_model import PATCH, ScopeFFTrainingModule, sort_refs_by_distance, to_pil

WAN_CHANNELS = 36
# Pixels per latent: the Wan VAE downsamples 8x. Six Plücker components per pixel.
LATENT_PATCH = 8
RAY_CHANNELS = 6 * LATENT_PATCH * LATENT_PATCH
IN_CHANNELS = WAN_CHANNELS + RAY_CHANNELS
# Data reads over NFS have stalled for over 10 minutes. A rank waiting that long in a collective
# should wait out the stall, not kill the run.
COLLECTIVE_TIMEOUT = timedelta(hours=2)


def extend_collective_timeout():
    """Groups made without a timeout, as ZeRO's gradient group is, take torch's 10-minute NCCL
    default. The world group's timeout is given to ``Accelerator`` separately."""
    c10d.default_pg_nccl_timeout = COLLECTIVE_TIMEOUT


def expand_patch_embedding(dit):
    """Wan's patch embedding for its first 36 channels, zero for the 384 ray channels."""
    old = dit.patch_embedding
    assert dit.in_dim == WAN_CHANNELS and old.in_channels == WAN_CHANNELS
    new = nn.Conv3d(IN_CHANNELS, dit.dim, kernel_size=dit.patch_size, stride=dit.patch_size).to(
        device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :WAN_CHANNELS] = old.weight
        new.bias.copy_(old.bias)
    dit.patch_embedding = new
    dit.in_dim = IN_CHANNELS


class ConcatWanTrainingModule(ScopeFFTrainingModule):
    def __init__(self, *args, **kwargs):
        # Skip ScopeFFTrainingModule.__init__, which installs scope-ff.
        WanTrainingModule.__init__(self, *args, **kwargs)
        # Anything but 'scope' routes encode_set's raymap to model_fn_wan_video.
        self.architecture = 'concat'
        install_batch_head(self.pipe.dit)
        expand_patch_embedding(self.pipe.dit)
        self.last_batch_shapes = None


def to_concat_set(batch, sort_refs=True, black_queries=False):
    """One RayCrafter batch (batch size 1) → ``encode_set``'s set dict.

    Frames are [ref, ref, query x3]. ``raymap`` is ``[5, 384, h, w]``, one 8×8 pixel-unshuffled
    Plücker map per frame in FrameCrafter's gauge (last camera at the origin, unit mean camera
    distance), so the loaders' near-depth unit cancels. ``black_queries`` turns the set into NVS.
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
    grid = (height // LATENT_PATCH, width // LATENT_PATCH)
    rays = plucker_raymap(c2w, fxfycxcy, (height, width), grid, LATENT_PATCH, camera_model=camera_model)
    raymap = rearrange(rays[0], '(n h w) c -> n c h w', n=5, h=grid[0], w=grid[1])
    refs = [to_pil(image) for image in ref]
    return {'input_images': refs, 'conditioning_images': [to_pil(image) for image in src],
            'target_images': refs + [to_pil(image) for image in tgt], 'prompt': '',
            'raymap': raymap.contiguous(), 'camera_model': camera_model, 'key': batch['key'][0]}
