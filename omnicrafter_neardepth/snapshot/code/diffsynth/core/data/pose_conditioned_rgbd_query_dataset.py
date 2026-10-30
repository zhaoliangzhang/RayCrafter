"""Pose-conditioned panorama RGB-D cleanup/NVS dataset.

This is intentionally separate from the RGB-only render-query loader. It adds
one depth task slot for every RGB query slot and keeps depth as float data all
the way to the VAE. RGB and depth conditioning for a query share exactly the
same online pixel-dropout mask.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .dataset import (
    get_equirectangular_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)


RGB_MODALITY = 0
DEPTH_MODALITY = 1


def unpack_bit_mask(
    packed: np.ndarray, shape: tuple[int, ...], bitorder: str = "little"
) -> np.ndarray:
    """Unpack a compact geometry mask without importing preprocessing code."""
    count = int(np.prod(shape))
    return np.unpackbits(packed, bitorder=bitorder)[:count].reshape(shape).astype(bool)


def nearest_surface_downsample(
    depth: np.ndarray,
    valid: np.ndarray,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a radial z-buffer using the closest valid sample per cell.

    Geometry archives currently use an exact 2x target resolution. The generic
    integer-factor implementation keeps this behavior explicit and refuses an
    interpolation that could average foreground and background surfaces.
    """
    if depth.shape != valid.shape or depth.ndim != 3:
        raise ValueError(
            f"Expected matching [N,H,W] depth/mask arrays, got "
            f"{depth.shape} and {valid.shape}"
        )
    num_views, input_height, input_width = depth.shape
    if input_height % output_height or input_width % output_width:
        raise ValueError(
            "Nearest-surface depth resize requires integer factors; got "
            f"{input_width}x{input_height} -> {output_width}x{output_height}"
        )
    factor_y = input_height // output_height
    factor_x = input_width // output_width
    depth_cells = depth.reshape(
        num_views, output_height, factor_y, output_width, factor_x
    ).transpose(0, 1, 3, 2, 4)
    valid_cells = valid.reshape(
        num_views, output_height, factor_y, output_width, factor_x
    ).transpose(0, 1, 3, 2, 4)
    candidates = np.where(valid_cells, depth_cells, np.inf)
    output_depth = candidates.min(axis=(3, 4)).astype(np.float32)
    output_valid = valid_cells.any(axis=(3, 4))
    output_depth[~output_valid] = 0.0
    return output_depth, output_valid


def horizontal_wrap_nearest_fill(
    encoded_depth: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Fill invalid ERP pixels from the nearest valid pixel with seam wrap."""
    if encoded_depth.shape != valid.shape or encoded_depth.ndim != 2:
        raise ValueError("Expected matching [H,W] encoded depth and validity mask")
    if not np.any(valid):
        raise ValueError("Cannot fill a depth map with no valid pixels")
    if np.all(valid):
        return encoded_depth.astype(np.float32, copy=True)

    width = encoded_depth.shape[1]
    tiled_depth = np.concatenate([encoded_depth] * 3, axis=1).astype(np.float32)
    tiled_valid = np.concatenate([valid] * 3, axis=1)
    distance_input = (~tiled_valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    label_values = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    label_values[labels[tiled_valid]] = tiled_depth[tiled_valid]
    filled = label_values[labels]
    return filled[:, width : 2 * width].copy()


def encode_log_radial_depth(
    radial_depth: np.ndarray,
    valid: np.ndarray,
    pose_scale: float,
    minimum_normalized_range: float,
    maximum_normalized_range: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode metric radial range into the fixed [-1,1] log-range domain."""
    if pose_scale <= 0 or not math.isfinite(pose_scale):
        raise ValueError(f"Invalid pose scale: {pose_scale}")
    if not 0 < minimum_normalized_range < maximum_normalized_range:
        raise ValueError("Depth bounds must satisfy 0 < minimum < maximum")

    normalized = np.zeros_like(radial_depth, dtype=np.float32)
    normalized[valid] = radial_depth[valid].astype(np.float32) / pose_scale
    valid_values = normalized[valid]
    if valid_values.size == 0:
        raise ValueError("Depth map has no valid pixels")
    low_clipped = valid_values < minimum_normalized_range
    high_clipped = valid_values > maximum_normalized_range
    clipped = np.clip(
        valid_values, minimum_normalized_range, maximum_normalized_range
    )
    log_min = math.log(minimum_normalized_range)
    log_span = math.log(maximum_normalized_range) - log_min
    encoded = np.zeros_like(normalized, dtype=np.float32)
    encoded[valid] = 2.0 * (np.log(clipped) - log_min) / log_span - 1.0
    return encoded, {
        "low_clip_fraction": float(low_clipped.mean()),
        "high_clip_fraction": float(high_clipped.mean()),
    }


def decode_log_radial_depth(
    encoded_depth: np.ndarray,
    pose_scale: float,
    minimum_normalized_range: float,
    maximum_normalized_range: float,
) -> np.ndarray:
    """Invert ``encode_log_radial_depth`` in original scene units."""
    log_min = math.log(minimum_normalized_range)
    log_span = math.log(maximum_normalized_range) - log_min
    normalized = np.exp(log_min + 0.5 * (encoded_depth + 1.0) * log_span)
    return (pose_scale * normalized).astype(np.float32)


def apply_joint_pixel_dropout(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    dropped: torch.Tensor,
    fill_value: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one spatial dropout mask to an aligned RGB/depth pair."""
    if rgb.shape != depth.shape or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected matching [3,H,W] RGB/depth tensors")
    if dropped.shape != rgb.shape[-2:]:
        raise ValueError(
            f"Dropout mask {dropped.shape} does not match raster {rgb.shape[-2:]}"
        )
    degraded_rgb = rgb.clone()
    degraded_depth = depth.clone()
    degraded_rgb[:, dropped] = fill_value
    degraded_depth[:, dropped] = fill_value
    return degraded_rgb, degraded_depth


class WanPanoramaPoseConditionedRGBDQueryDataset(Dataset):
    """Return two RGB references and paired RGB/depth query tasks.

    Slot order is ``reference RGB``, ``query RGB``, then ``query depth``. The
    physical query pose is duplicated only after one common pose normalization
    scale has been computed from unique cameras.
    """

    def __init__(
        self,
        base_path,
        metadata_path,
        pose_manifest_path,
        geometry_root,
        repeat=1,
        height=256,
        width=512,
        max_samples=None,
        min_pixel_dropout=0.0,
        max_pixel_dropout=1.0,
        minimum_normalized_range=1.0 / 16.0,
        maximum_normalized_range=2048.0,
        no_pixel_unshuffle=False,
        deterministic_degradation=False,
    ):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.geometry_root = Path(geometry_root)
        self.repeat = repeat
        self.height = height
        self.width = width
        self.min_pixel_dropout = min_pixel_dropout
        self.max_pixel_dropout = max_pixel_dropout
        self.minimum_normalized_range = minimum_normalized_range
        self.maximum_normalized_range = maximum_normalized_range
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.deterministic_degradation = deterministic_degradation
        self._pose_cache: dict[str, np.ndarray] = {}

        if width != 2 * height:
            raise ValueError(f"Expected a 2:1 panorama, got {width}x{height}")
        if repeat < 1:
            raise ValueError(f"repeat must be positive, got {repeat}")
        if not 0.0 <= min_pixel_dropout <= max_pixel_dropout <= 1.0:
            raise ValueError("pixel dropout must satisfy 0 <= min <= max <= 1")

        manifest = json.loads(Path(pose_manifest_path).read_text())
        if manifest.get("pose_convention") != "w2c_opencv":
            raise ValueError("Pose manifest must contain OpenCV w2c poses")
        self.pose_manifest = manifest["scenes"]

        groups = defaultdict(list)
        with self.metadata_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "source_sample" not in row:
                    raise ValueError(
                        f"Missing source_sample in {self.metadata_path}:{line_number}"
                    )
                groups[row["source_sample"]].append(row)

        self.samples = []
        for sample_id in sorted(groups):
            rows = sorted(groups[sample_id], key=lambda row: row["target_number"])
            self._validate_group(sample_id, rows)
            scene_name = rows[0]["scene_name"]
            geometry_path = self.geometry_root / scene_name / f"{sample_id}.npz"
            if not geometry_path.is_file():
                raise FileNotFoundError(
                    f"Missing RGB-D geometry for {sample_id}: {geometry_path}"
                )
            refs = rows[0]["refs"]
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "scene_name": scene_name,
                    "refs": [self._resolve(path) for path in refs],
                    "ref_indices": [self._local_frame_index(path) for path in refs],
                    "noisy": [self._resolve(row["src"]) for row in rows],
                    "targets": [self._resolve(row["target"]) for row in rows],
                    "target_indices": [int(row["frame_index"]) for row in rows],
                    "geometry_path": geometry_path,
                }
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No complete samples found in {metadata_path}")
        print(
            f"Total pose-conditioned RGB-D query samples: {len(self.samples)} "
            f"({len(self)} with repeat={repeat})"
        )

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.base_path / path

    @staticmethod
    def _local_frame_index(path):
        return int(Path(path).stem)

    @staticmethod
    def _validate_group(sample_id, rows):
        if not rows:
            raise ValueError(f"Empty sample {sample_id}")
        target_numbers = [int(row["target_number"]) for row in rows]
        if target_numbers != list(range(len(rows))):
            raise ValueError(
                f"{sample_id} target numbers must be contiguous from zero; "
                f"got {target_numbers}"
            )
        if len({tuple(row["refs"]) for row in rows}) != 1:
            raise ValueError(f"{sample_id} must use the same references for all targets")
        if len({row["scene_name"] for row in rows}) != 1:
            raise ValueError(f"{sample_id} crosses scene boundaries")

    def _load_rgb_tensor(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = TF.resize(
                    image,
                    [self.height, self.width],
                    interpolation=InterpolationMode.BILINEAR,
                )
            tensor = TF.pil_to_tensor(image).float().div_(127.5).sub_(1.0)
        return tensor

    def _scene_poses(self, scene_name):
        if scene_name not in self._pose_cache:
            pose_path = self.base_path / self.pose_manifest[scene_name]["pose_path"]
            with np.load(pose_path) as data:
                poses = np.asarray(data["w2c_poses"], dtype=np.float32)
            expected = int(self.pose_manifest[scene_name]["num_frames"])
            if poses.shape != (expected, 4, 4):
                raise ValueError(f"Unexpected pose shape for {scene_name}: {poses.shape}")
            self._pose_cache[scene_name] = poses
        return self._pose_cache[scene_name]

    def _load_target_depth(self, sample, pose_scale):
        with np.load(sample["geometry_path"], allow_pickle=False) as geometry:
            if str(geometry["source_sample"]) != sample["sample_id"]:
                raise ValueError(f"Geometry/sample mismatch for {sample['sample_id']}")
            target_indices = geometry["target_frame_indices"].astype(int).tolist()
            if target_indices != sample["target_indices"]:
                raise ValueError(
                    f"Geometry target order mismatch for {sample['sample_id']}: "
                    f"{target_indices} != {sample['target_indices']}"
                )
            bitorder = str(geometry["mask_bitorder"])
            valid_shape = tuple(
                int(value) for value in geometry["target_valid_mask_shape"]
            )
            valid = unpack_bit_mask(
                geometry["target_valid_mask_packed"], valid_shape, bitorder
            )
            depth = geometry["target_radial_depth"].astype(np.float32)

        depth, valid = nearest_surface_downsample(
            depth, valid, self.height, self.width
        )
        encoded_views = []
        clipping = []
        for view_depth, view_valid in zip(depth, valid):
            encoded, stats = encode_log_radial_depth(
                view_depth,
                view_valid,
                pose_scale,
                self.minimum_normalized_range,
                self.maximum_normalized_range,
            )
            encoded = horizontal_wrap_nearest_fill(encoded, view_valid)
            encoded_views.append(torch.from_numpy(encoded).repeat(3, 1, 1))
            clipping.append(stats)
        return encoded_views, torch.from_numpy(valid), clipping

    def _rng(self, sample_id, repeated_index):
        if not self.deterministic_degradation:
            return random
        digest = hashlib.sha256(
            f"{sample_id}:{repeated_index}".encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))

    def _sample_dropout_mask(self, rng):
        min_percent = int(round(self.min_pixel_dropout * 100.0))
        max_percent = int(round(self.max_pixel_dropout * 100.0))
        dropout_ratio = rng.randint(min_percent, max_percent) / 100.0
        pixel_rng = np.random.default_rng(rng.getrandbits(64))
        dropped = pixel_rng.random((self.height, self.width)) < dropout_ratio
        return torch.from_numpy(dropped), dropout_ratio

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        rng = self._rng(sample["sample_id"], index // len(self.samples))

        references = [self._load_rgb_tensor(path) for path in sample["refs"]]
        noisy_renders = [self._load_rgb_tensor(path) for path in sample["noisy"]]
        clean_targets = [self._load_rgb_tensor(path) for path in sample["targets"]]

        physical_indices = list(
            dict.fromkeys(sample["ref_indices"] + sample["target_indices"])
        )
        scene_poses = self._scene_poses(sample["scene_name"])
        if max(physical_indices) >= len(scene_poses):
            raise IndexError(f"Frame index outside pose array for {sample['sample_id']}")
        physical_w2cs = torch.from_numpy(scene_poses[physical_indices]).float()
        _, physical_c2w_norm, scale = normalize_w2c_make_cam_last_origin(
            physical_w2cs
        )
        pose_scale = float(scale)
        normalized_pose_by_index = {
            frame_index: physical_c2w_norm[position]
            for position, frame_index in enumerate(physical_indices)
        }

        rgb_slot_indices = sample["ref_indices"] + sample["target_indices"]
        all_slot_indices = rgb_slot_indices + sample["target_indices"]
        slot_c2ws = torch.stack(
            [normalized_pose_by_index[frame_index] for frame_index in all_slot_indices]
        )
        raymap = get_equirectangular_plucker_rays(
            slot_c2ws,
            height=self.height,
            width=self.width,
            no_pixel_unshuffle=self.no_pixel_unshuffle,
        )

        encoded_depth, depth_valid, clipping = self._load_target_depth(
            sample, pose_scale
        )
        if len(encoded_depth) != len(noisy_renders):
            raise ValueError(
                f"RGB/depth query count mismatch for {sample['sample_id']}"
            )

        degraded_rgb = []
        degraded_depth = []
        dropout_masks = []
        dropout_ratios = []
        for rgb, depth in zip(noisy_renders, encoded_depth):
            dropped, dropout_ratio = self._sample_dropout_mask(rng)
            # -1 is black in the VAE's [-1,1] domain. The same boolean array is
            # deliberately applied to both modalities for this physical view.
            degraded_rgb_view, degraded_depth_view = apply_joint_pixel_dropout(
                rgb, depth, dropped, fill_value=-1.0
            )
            degraded_rgb.append(degraded_rgb_view)
            degraded_depth.append(degraded_depth_view)
            dropout_masks.append(dropped)
            dropout_ratios.append(dropout_ratio)

        num_references = len(references)
        num_queries = len(clean_targets)
        modality_ids = torch.tensor(
            [RGB_MODALITY] * (num_references + num_queries)
            + [DEPTH_MODALITY] * num_queries,
            dtype=torch.long,
        )
        output_validity = torch.cat(
            [
                torch.ones(num_queries, self.height, self.width, dtype=torch.float32),
                depth_valid.float(),
            ],
            dim=0,
        )

        return {
            "input_rasters": references,
            "conditioning_rasters": degraded_rgb + degraded_depth,
            "target_rasters": references + clean_targets + encoded_depth,
            "raymap": raymap,
            "modality_ids": modality_ids,
            "output_validity_masks": output_validity,
            "prompt": "",
            "sample_id": sample["sample_id"],
            "scene_name": sample["scene_name"],
            "slot_roles": ["reference_rgb"] * num_references
            + ["query_rgb"] * num_queries
            + ["query_depth"] * num_queries,
            "slot_frame_indices": all_slot_indices,
            "query_pixel_dropout_masks": torch.stack(dropout_masks),
            "query_pixel_dropout_ratios": dropout_ratios,
            "query_depth_conditioning_validity_masks": depth_valid.float(),
            "target_task_types": ["rgb"] * num_queries + ["depth"] * num_queries,
            "pose_scale": pose_scale,
            "depth_clipping": clipping,
            "depth_encoding": {
                "representation": "normalized_log_radial_range",
                "minimum_normalized_range": self.minimum_normalized_range,
                "maximum_normalized_range": self.maximum_normalized_range,
            },
        }
