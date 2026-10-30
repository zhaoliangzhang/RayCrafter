#!/usr/bin/env python3
"""Align a two-view PanoVGGT cloud to GT poses and render a held-out ERP view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


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


def wrap_nearest_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor visualization fill with horizontal ERP wrapping."""
    tiled_image = np.concatenate([image, image, image], axis=1)
    tiled_valid = np.concatenate([mask > 0, mask > 0, mask > 0], axis=1)
    _, indices = distance_transform_edt(~tiled_valid, return_indices=True)
    filled = tiled_image[indices[0], indices[1]]
    width = image.shape[1]
    return filled[:, width : 2 * width]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-indices", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=1036)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(args.predictions)
    pred_c2w = predictions["camera_poses"][0].astype(np.float64)
    points = predictions["world_points"][0].reshape(-1, 3).astype(np.float64)
    colors = predictions["images"][0].transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float64)

    metadata = json.loads(args.metadata.read_text())
    frames = metadata["frames"]
    condition_w2c = np.stack(
        [np.asarray(frames[i]["w2c"], dtype=np.float64) for i in args.condition_indices]
    )
    condition_c2w = np.linalg.inv(condition_w2c)
    target_w2c = np.asarray(frames[args.target_index]["w2c"], dtype=np.float64)

    scale, rotation, translation, diagnostics = align_cameras(pred_c2w, condition_c2w)
    aligned_points = scale * (points @ rotation.T) + translation
    finite = np.isfinite(aligned_points).all(axis=1) & np.isfinite(colors).all(axis=1)
    aligned_points, colors = aligned_points[finite], colors[finite]

    render, depth, mask = render_erp(
        aligned_points, colors, target_w2c, args.height, args.width
    )
    filled = wrap_nearest_fill(render, mask)

    gt_bgr = cv2.imread(frames[args.target_index]["image_path"], cv2.IMREAD_COLOR)
    if gt_bgr is None:
        raise FileNotFoundError(frames[args.target_index]["image_path"])
    gt_bgr = cv2.resize(gt_bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
    gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    comparison = np.concatenate([gt, render, filled], axis=0)

    cv2.imwrite(str(args.output_dir / "target_gt.png"), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_dir / "render_raw.png"), cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_dir / "render_nearest_filled.png"), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_dir / "render_valid_mask.png"), mask)
    cv2.imwrite(str(args.output_dir / "comparison_gt_raw_filled.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    np.save(args.output_dir / "render_radial_depth.npy", depth)

    # Open3D is only needed by this command-line visualization path.  Keep the
    # geometry-only helpers importable in lean inference environments.
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(aligned_points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(
        str(args.output_dir / "aligned_point_cloud.ply"), cloud, write_ascii=False
    )

    valid = mask > 0
    raw_mse = float(np.mean(((render[valid].astype(np.float32) - gt[valid]) / 255.0) ** 2))
    filled_mse = float(np.mean(((filled.astype(np.float32) - gt) / 255.0) ** 2))
    raw_psnr = float(-10.0 * np.log10(max(raw_mse, 1e-12)))
    filled_psnr = float(-10.0 * np.log10(max(filled_mse, 1e-12)))
    diagnostics.update(
        {
            "scene": metadata.get("scene_name"),
            "condition_indices_zero_based": args.condition_indices,
            "condition_images": [frames[i]["image_path"] for i in args.condition_indices],
            "target_index_zero_based": args.target_index,
            "target_image": frames[args.target_index]["image_path"],
            "similarity_rotation": rotation.tolist(),
            "similarity_translation": translation.tolist(),
            "aligned_point_count": int(len(aligned_points)),
            "render_resolution": [args.height, args.width],
            "raw_pixel_coverage": float(valid.mean()),
            "raw_valid_pixel_psnr_db": raw_psnr,
            "nearest_filled_full_image_psnr_db": filled_psnr,
        }
    )
    (args.output_dir / "alignment_and_render.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
