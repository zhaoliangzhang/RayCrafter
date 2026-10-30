"""OmniCrafter training module with UniRoPE, and the RayCrafter-batch adapter.

``UniRoPETrainingModule`` is scope-ff's ``ScopeFFTrainingModule`` (same pipeline
units, flow-matching loss on the 3 query latents, batched head) with UniRoPE
installed instead of scope-ff. ``to_omnicrafter_unirope`` turns one batch of
RayCrafter's ``dl3dv_points`` / panorama loaders into the set dict ``encode_set``
reads, with the geometry flux3_mix_unirope_v3 would build for it.
"""
import torch

import unirope_paths  # noqa: F401
from mixed_model import MixedTrainingModule, install_batch_head
from model_training.train import WanTrainingModule
from raycrafter.cameras.unirope import prepare_unirope
from scopeff_model import PATCH, ScopeFFTrainingModule, sort_refs_by_distance, to_pil
from unirope_wan import install_unirope, model_fn_unirope

# encode_set's flat geometry keys. MixedTrainingModule.forward reads .shape of every
# packed value, so they stay tensors until model_fn_unirope builds the geometry dict.
GEOMETRY = {'unirope_c2w': 'c2w', 'unirope_w2c': 'w2c', 'unirope_directions': 'directions',
            'unirope_patch_angle': 'patch_angle'}
DEPTH_HEAD = '.unirope_depth.'


class UniRoPETrainingModule(MixedTrainingModule):
    def __init__(self, *args, apply_vo, **kwargs):
        # Skip MixedTrainingModule.__init__, which installs the plain SCoPE module.
        WanTrainingModule.__init__(self, *args, **kwargs)
        assert self.pipe.dit.in_dim == 36
        install_batch_head(self.pipe.dit)
        install_unirope(self.pipe.dit, apply_vo=apply_vo)
        self.pipe.model_fn = model_fn_unirope
        self.last_batch_shapes = None

    # Fixed timestep and seeded noise; nothing in it is specific to scope-ff.
    val_loss = ScopeFFTrainingModule.val_loss

    @torch.no_grad()
    def encode_set(self, data):
        """``MixedTrainingModule.encode_set`` with the UniRoPE geometry as the camera input."""
        assert len(data['input_images']) == 2 and len(data['conditioning_images']) == 3
        assert len(data['target_images']) == 5 and data['prompt'] == ''
        assert 'raymap' not in data and 'scope_rays' not in data
        inputs = self.transfer_data_to_device(self._get_pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _ = inputs
        assert shared['mask_loss'] and shared['num_output_frames'] == 3
        packed = {k: ({**shared, **positive})[k] for k in ('input_latents', 'y', 'context')}
        for key in GEOMETRY:
            if key in data:
                packed[key] = data[key].to(self.pipe.device, dtype=torch.float32).unsqueeze(0)
        return packed


def optimizer_groups(module, depth_lr, depth_weight_decay):
    """AdamW groups: the backbone's trainable weights, then the depth heads with their
    own settings. As flux3, whose heads barely move at the backbone's 1e-5."""
    backbone, depth = [], []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            (depth if DEPTH_HEAD in name else backbone).append(parameter)
    return [{'params': backbone}, {'params': depth, 'lr': depth_lr, 'weight_decay': depth_weight_decay}]


def to_omnicrafter_unirope(batch, footprint=True, adaptive_freqs=True, sort_refs=True, black_queries=False):
    """One RayCrafter batch (batch size 1) → ``encode_set``'s set dict.

    Frames are [ref, ref, query x3], the order of the cameras in the geometry.
    ``prepare_unirope`` makes the poses relative to the first (nearest) ref with the
    second ref at unit distance, so the loaders' near-depth scale cancels out.
    ``black_queries`` turns the set into NVS.
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
    with torch.no_grad():
        geom = prepare_unirope(c2w, fxfycxcy, (height, width), (height // PATCH, width // PATCH),
                               footprint=footprint, camera_model=camera_model, adaptive_freqs=adaptive_freqs)
    refs = [to_pil(image) for image in ref]
    data = {'input_images': refs, 'conditioning_images': [to_pil(image) for image in src],
            'target_images': refs + [to_pil(image) for image in tgt], 'prompt': '',
            'camera_model': camera_model, 'key': batch['key'][0]}
    for key, name in GEOMETRY.items():
        if name in geom:
            data[key] = geom[name][0].contiguous()
    return data
