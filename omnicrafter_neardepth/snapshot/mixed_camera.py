"""Projection-aware Pluecker rays, keeping the existing last-query pose gauge."""
import sys
import math
import cv2
import numpy as np
import torch
from common import REPO
sys.path.insert(0, str(REPO))
from diffsynth.core.data.dataset import (
    _equirectangular_camera_to_raymap, _raymap_to_plucker,
)


def normalize_near_depth_last_query(w2c, near_depth):
    """Recenter AND rotate into query 3's frame; scale translations once.

    near_depth is a single positive scalar in the original pose units, fixed
    across all rows in a scene. There is deliberately no baseline normalization
    before or after this operation. Intrinsics and rotation matrices are unchanged
    by scale. The last of [ref1, ref2, query1, query2, query3] defines the frame.
    """
    scale = float(near_depth)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'Expected a finite positive scene near depth, got {near_depth!r}')
    if w2c.shape != (5,4,4) or not torch.isfinite(w2c).all():
        raise ValueError('Expected five finite 4x4 world-to-camera poses')
    c2w = torch.linalg.inv(w2c)
    rotation = c2w[-1,:3,:3].transpose(0,1)
    centered = c2w[:,:3,3] - c2w[-1:,:3,3]
    result = torch.eye(4,device=w2c.device,dtype=w2c.dtype).repeat(5,1,1)
    result[:,:3,:3] = rotation @ c2w[:,:3,:3]
    result[:,:3,3] = (rotation @ centered.T).T / scale
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite normalized poses')
    return result,scale


def scaled_intrinsics(camera, height, width):
    K = np.asarray(camera['K'], dtype=np.float64).copy()
    if K.shape != (3,3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError('Invalid perspective intrinsics')
    if not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0]) > 1e-8:
        raise ValueError('Expected the supplied zero-skew OpenCV camera model')
    K[0] *= width / camera['width']
    K[1] *= height / camera['height']
    return K


def distort_normalized(xy, distortion):
    k1,k2,p1,p2 = distortion
    x,y = xy[...,0],xy[...,1]
    r2 = x*x+y*y
    radial = 1+k1*r2+k2*r2*r2
    return np.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                     y*radial+p1*(r2+2*y*y)+2*p2*x*y],axis=-1)


def perspective_directions(camera, height, width, stride=1):
    """Unit camera rays at pixel/patch centers of the resized, distorted RGB.

    Priors were rendered with OpenCV k1,k2,p1,p2 distortion, so simply applying
    K^-1 to these images is not correct. Invert that same distortion model.
    With zero distortion this is exactly FrameCrafter's pinhole calculation.
    """
    if height % stride or width % stride:
        raise ValueError('Image dimensions must be divisible by ray stride')
    K = scaled_intrinsics(camera,height,width)
    y,x = np.meshgrid((np.arange(height//stride)+.5)*stride,
                      (np.arange(width//stride)+.5)*stride,indexing='ij')
    pixels = np.stack([x,y],axis=-1)
    distortion = np.asarray(camera.get('dist',[0,0,0,0]),dtype=np.float64)
    if distortion.shape != (4,) or not np.isfinite(distortion).all():
        raise ValueError('Expected finite OpenCV k1,k2,p1,p2 coefficients')
    observed = (pixels - K[:2,2]) / np.diag(K)[:2]
    if np.any(distortion):
        criteria = (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-10)
        if hasattr(cv2,'undistortPointsIter'):
            undistorted = cv2.undistortPointsIter(pixels.reshape(-1,1,2),K,distortion,None,None,criteria)
        else:  # OpenCV 5 merged the iterative variant into undistortPoints.
            undistorted = cv2.undistortPoints(pixels.reshape(-1,1,2),K,distortion,criteria=criteria)
        xy = undistorted.reshape(pixels.shape)
        error = np.max(np.abs(distort_normalized(xy,distortion)-observed))
        if not np.isfinite(xy).all() or error > 1e-5:
            raise ValueError(f'Distortion inversion failed, normalized residual={error}')
    else:
        xy = observed
    directions = np.concatenate([xy,np.ones_like(xy[...,:1])],axis=-1)
    directions /= np.linalg.norm(directions,axis=-1,keepdims=True)
    return torch.from_numpy(directions.astype(np.float32))


def camera_features(record, architecture):
    if architecture not in ('scope','concat'):
        raise ValueError('Specify scope or concat camera injection explicitly')
    projection = record['projection']
    h,w = record['height'],record['width']
    if (projection,h,w) not in (('panorama',256,512),('perspective',288,512)):
        raise ValueError(f'Unexpected projection/resolution: {projection},{h},{w}')
    views = record['references']+record['targets']
    if len(views) != 5:
        raise ValueError('Need exactly two references and three targets')
    w2c = torch.tensor([v['w2c'] for v in views],dtype=torch.float32)
    if 'near_depth' not in record:
        raise ValueError('Missing per-scene near depth; baseline fallback is forbidden')
    c2w,scale = normalize_near_depth_last_query(w2c,record['near_depth'])
    stride = 16 if architecture == 'scope' else 1
    if projection == 'panorama':
        rays = _equirectangular_camera_to_raymap(c2w,h//stride,w//stride)
    else:
        directions = torch.stack([perspective_directions(v['camera'],h,w,stride) for v in views])
        directions = torch.einsum('fij,fhwj->fhwi',c2w[:,:3,:3],directions)
        directions = torch.nn.functional.normalize(directions,dim=-1)
        origins = c2w[:,None,None,:3,3].expand_as(directions)
        rays = torch.cat([origins,directions],dim=-1)
    plucker = _raymap_to_plucker(rays)
    if not torch.isfinite(plucker).all():
        raise ValueError('Non-finite camera rays')
    if architecture == 'scope':
        return {'scope_rays':plucker.reshape(-1,6).contiguous()},float(scale)
    return {'raymap':torch.nn.functional.pixel_unshuffle(plucker.permute(0,3,1,2),8)},float(scale)
