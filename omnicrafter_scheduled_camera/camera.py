"""Query 3 is the global origin. Manifest/model reference order NEVER changes.

Concat retains its existing mean-distance scale for warm-start equivalence.
SCoPE uses scene near depth. RayRoPE uses unit reference baseline; its
internal per-camera relative-attention passes do not change this global gauge.
Pinned RayCrafter source files remain unmodified.
"""
import math
import torch
from config import VARIANT
from concat_camera import camera_features
from vendor.raycrafter_camera.scope import plucker_tokens
from vendor.raycrafter_camera.rayrope import prepare_geometry

def ordered_cameras(record):
    h, w = int(record['height']), int(record['width'])
    if record['projection'] not in ('panorama', 'perspective') or min(h,w) <= 0 or h%16 or w%16:
        raise ValueError('Invalid projection or resolution')
    if len(record['references']) != 2 or len(record['targets']) != 3:
        raise ValueError('Expected two references and three targets')
    depth = float(record['near_depth'])
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError('Positive verified per-scene near depth required')
    views = record['references'] + record['targets']
    w2c = torch.tensor([v['w2c'] for v in views], dtype=torch.float64)
    if w2c.shape != (5,4,4) or not torch.isfinite(w2c).all():
        raise ValueError('Five finite OpenCV camera matrices required')
    c2w = torch.linalg.inv(w2c)
    relative = w2c[-1:] @ c2w
    relative[-1] = torch.eye(4, dtype=relative.dtype)
    K = torch.zeros(5,4)
    if record['projection'] == 'perspective':
        for i, view in enumerate(views):
            cam = view['camera']; k = cam['K']
            K[i] = torch.tensor([k[0][0]*w/cam['width'], k[1][1]*h/cam['height'],
                                 k[0][2]*w/cam['width'], k[1][2]*h/cam['height']])
        if not torch.isfinite(K).all() or (K[:,:2] <= 0).any():
            raise ValueError('Invalid perspective intrinsics')
    features, scale = camera_features(record, 'concat')
    return record, {'c2w': relative.float(), 'fxfycxcy': K, 'near_depth': depth,
        'image_hw': (h,w), 'latent_hw': (h//16,w//16),
        'camera_model': 'equirect' if record['projection']=='panorama' else 'pinhole',
        'raymap': features['raymap'], 'concat_scale': scale}

@torch.no_grad()
def prepare_camera_context(cameras, device):
    device = torch.device(device)
    c2w = cameras['c2w'].to(device=device, dtype=torch.float32)
    K = cameras['fxfycxcy'].to(device=device, dtype=torch.float32)
    if c2w.ndim == 3:
        c2w, K = c2w.unsqueeze(0), K.unsqueeze(0)
    if c2w.shape[1:] != (5,4,4):
        raise ValueError('Invalid camera shape')
    raymap = cameras['raymap']
    if raymap.ndim == 4:
        raymap = raymap.permute(1,0,2,3).unsqueeze(0)
    h,w = cameras['image_hw']
    if tuple(raymap.shape) != (c2w.shape[0],384,5,h//8,w//8):
        raise ValueError('Unexpected dense ray layout')
    result = {'raymap': raymap.to(device=device, dtype=torch.float32), 'num_frames': 5,
              'image_hw': (h,w), 'latent_hw': cameras['latent_hw']}
    with torch.autocast(device_type=device.type, enabled=False):
        if VARIANT == 'scope':
            scaled = c2w.clone()
            scaled[:,:,:3,3] /= cameras['near_depth']
            # Upstream anchors camera 0. Temporarily put q3 there for geometry
            # construction ONLY, then restore the original token order.
            indices, inverse = [4,0,1,2,3], [1,2,3,4,0]
            rays = plucker_tokens(scaled[:,indices], K[:,indices], (h,w),
                cameras['latent_hw'], camera_model=cameras['camera_model'], patch_size=False)
            result['plucker'] = rays.reshape(c2w.shape[0],5,-1,6)[:,inverse].reshape(c2w.shape[0],-1,6)
            result['normalized_c2w'] = scaled
        else:
            geom = prepare_geometry(c2w,K,(h,w),cameras['latent_hw'],camera_model=cameras['camera_model'])
            # Keep the vendor's reference-baseline SCALE, but change its
            # world gauge from ref1 to q3 consistently for every matrix.
            old_anchor = geom['c2w'][:,-1:].clone()
            transform = torch.linalg.inv(old_anchor)
            geom['c2w'] = transform @ geom['c2w']
            geom['w2c'] = geom['w2c'] @ old_anchor
            geom['projection'] = geom['projection'] @ old_anchor
            geom['projection_inv'] = transform @ geom['projection_inv']
            result['geom'] = geom
            result['normalized_c2w'] = geom['c2w']
    return result
