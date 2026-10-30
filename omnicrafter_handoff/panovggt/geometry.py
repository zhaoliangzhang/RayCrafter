"""Reference-pose similarity alignment and raw ERP point rasterization.

Numerical functions extracted unchanged from render_jhu_heldout.py.
"""
from __future__ import annotations
import numpy as np

def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def align_cameras(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Find p_gt = scale * rotation @ p_pred + translation."""
    pred_r = pred_c2w[:, :3, :3]
    gt_r = gt_c2w[:, :3, :3]
    rotation = project_to_so3(sum(g @ p.T for g, p in zip(gt_r, pred_r)))

    pred_c = pred_c2w[:, :3, 3]
    gt_c = gt_c2w[:, :3, 3]
    pred_baseline = np.linalg.norm(pred_c[1] - pred_c[0])
    gt_baseline = np.linalg.norm(gt_c[1] - gt_c[0])
    if pred_baseline < 1e-8:
        raise ValueError("Predicted condition-camera baseline is zero")
    scale = float(gt_baseline / pred_baseline)
    translation = np.mean(gt_c - scale * (rotation @ pred_c.T).T, axis=0)

    aligned_c = scale * (rotation @ pred_c.T).T + translation
    center_errors = np.linalg.norm(aligned_c - gt_c, axis=1)
    rotation_errors = []
    for pred, gt in zip(pred_r, gt_r):
        relative = gt.T @ rotation @ pred
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    diagnostics = {
        "predicted_baseline": float(pred_baseline),
        "gt_baseline": float(gt_baseline),
        "scale": scale,
        "condition_center_errors": [float(x) for x in center_errors],
        "condition_rotation_errors_deg": rotation_errors,
    }
    return scale, rotation, translation, diagnostics


def render_erp(
    points_world: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    radius = np.linalg.norm(points_camera, axis=1)
    valid = np.isfinite(points_camera).all(axis=1) & np.isfinite(radius) & (radius > 1e-5)
    points_camera, radius, colors = points_camera[valid], radius[valid], colors[valid]

    x, y, z = points_camera.T
    longitude = np.arctan2(x, z)
    latitude_down = np.arcsin(np.clip(y / radius, -1.0, 1.0))
    u = np.mod((longitude / (2.0 * np.pi) + 0.5) * width, width)
    v = (latitude_down / np.pi + 0.5) * height
    ui = np.floor(u).astype(np.int64) % width
    vi = np.clip(np.floor(v).astype(np.int64), 0, height - 1)
    pixel = vi * width + ui

    # Sort by pixel first and radial depth second, then retain the nearest point.
    order = np.lexsort((radius, pixel))
    sorted_pixel = pixel[order]
    _, first = np.unique(sorted_pixel, return_index=True)
    selected = order[first]
    selected_pixel = pixel[selected]

    flat_color = np.zeros((height * width, 3), dtype=np.uint8)
    flat_depth = np.full(height * width, np.nan, dtype=np.float32)
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    flat_color[selected_pixel] = np.clip(colors[selected] * 255.0, 0, 255).astype(np.uint8)
    flat_depth[selected_pixel] = radius[selected].astype(np.float32)
    flat_mask[selected_pixel] = 255
    return (
        flat_color.reshape(height, width, 3),
        flat_depth.reshape(height, width),
        flat_mask.reshape(height, width),
    )


