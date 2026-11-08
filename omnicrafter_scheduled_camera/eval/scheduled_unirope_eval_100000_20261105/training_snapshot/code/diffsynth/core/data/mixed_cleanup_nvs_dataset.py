"""Pose-aware joint panorama cleanup and novel-view synthesis dataset."""

from __future__ import annotations

import json
import hashlib
import random
from collections import defaultdict
from pathlib import Path

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


class WanPanoramaMixedCleanupNVSDataset(Dataset):
    """Return clean/noisy known views followed by three clean query views.

    Two clean references are always known. A random subset of the three noisy
    point-cloud renders is added as known input views. All three clean target
    views are supervised query slots. A target is a cleanup task when its noisy
    counterpart was selected and an NVS task otherwise.
    """

    def __init__(
        self,
        base_path,
        metadata_path,
        pose_manifest_path,
        repeat=1,
        height=256,
        width=512,
        max_samples=None,
        min_noisy_inputs=1,
        max_noisy_inputs=2,
        fixed_noisy_input_ids=None,
        no_pixel_unshuffle=False,
        deterministic_noisy_selection=False,
    ):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.repeat = repeat
        self.height = height
        self.width = width
        self.min_noisy_inputs = min_noisy_inputs
        self.max_noisy_inputs = max_noisy_inputs
        self.fixed_noisy_input_ids = (
            None
            if fixed_noisy_input_ids is None
            else tuple(int(index) for index in fixed_noisy_input_ids)
        )
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.deterministic_noisy_selection = deterministic_noisy_selection
        self._pose_cache = {}

        if width != 2 * height:
            raise ValueError(f"Expected a 2:1 panorama, got {width}x{height}")
        if not (0 <= min_noisy_inputs <= max_noisy_inputs <= 3):
            raise ValueError("Require 0 <= min_noisy_inputs <= max_noisy_inputs <= 3")
        if self.fixed_noisy_input_ids is not None:
            if (
                tuple(sorted(set(self.fixed_noisy_input_ids)))
                != self.fixed_noisy_input_ids
                or any(index not in range(3) for index in self.fixed_noisy_input_ids)
            ):
                raise ValueError(
                    "fixed_noisy_input_ids must be sorted, unique indices in [0, 2]"
                )

        manifest = json.loads(Path(pose_manifest_path).read_text())
        if manifest.get("pose_convention") != "w2c_opencv":
            raise ValueError("Pose manifest must contain OpenCV w2c poses")
        self.pose_manifest = manifest["scenes"]

        groups = defaultdict(list)
        with self.metadata_path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    groups[row["source_sample"]].append(row)

        self.samples = []
        for sample_id in sorted(groups):
            rows = sorted(groups[sample_id], key=lambda row: row["target_number"])
            self._validate_group(sample_id, rows)
            scene_name = rows[0]["scene_name"]
            if scene_name not in self.pose_manifest:
                raise KeyError(f"No pose manifest entry for {scene_name}")
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "scene_name": scene_name,
                    "refs": [self._resolve(path) for path in rows[0]["refs"]],
                    "ref_indices": [self._local_frame_index(path) for path in rows[0]["refs"]],
                    "noisy": [self._resolve(row["src"]) for row in rows],
                    "targets": [self._resolve(row["target"]) for row in rows],
                    "target_indices": [int(row["frame_index"]) for row in rows],
                }
            )
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No mixed cleanup/NVS samples in {metadata_path}")
        print(
            f"Total mixed cleanup/NVS samples: {len(self.samples)} "
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
        if len(rows) != 3 or [row["target_number"] for row in rows] != [0, 1, 2]:
            raise ValueError(f"{sample_id} must contain target numbers [0, 1, 2]")
        if len({tuple(row["refs"]) for row in rows}) != 1 or len(rows[0]["refs"]) != 2:
            raise ValueError(f"{sample_id} must contain two shared clean references")
        if len({row["scene_name"] for row in rows}) != 1:
            raise ValueError(f"{sample_id} crosses scene boundaries")

    def _load_image(self, path):
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
            return image.copy()

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

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        if self.fixed_noisy_input_ids is not None:
            noisy_input_ids = list(self.fixed_noisy_input_ids)
        elif self.deterministic_noisy_selection:
            digest = hashlib.sha256(sample["sample_id"].encode()).digest()
            sample_rng = random.Random(int.from_bytes(digest[:8], "little"))
            num_noisy = sample_rng.randint(self.min_noisy_inputs, self.max_noisy_inputs)
            noisy_input_ids = sorted(sample_rng.sample(range(3), num_noisy))
        else:
            sample_rng = random
            num_noisy = sample_rng.randint(self.min_noisy_inputs, self.max_noisy_inputs)
            noisy_input_ids = sorted(sample_rng.sample(range(3), num_noisy))
        num_noisy = len(noisy_input_ids)

        clean_refs = [self._load_image(path) for path in sample["refs"]]
        noisy_inputs = [self._load_image(sample["noisy"][idx]) for idx in noisy_input_ids]
        clean_targets = [self._load_image(path) for path in sample["targets"]]
        input_images = clean_refs + noisy_inputs
        target_images = input_images + clean_targets

        scene_poses = self._scene_poses(sample["scene_name"])
        input_indices = sample["ref_indices"] + [
            sample["target_indices"][idx] for idx in noisy_input_ids
        ]
        slot_indices = input_indices + sample["target_indices"]
        if max(slot_indices) >= len(scene_poses):
            raise IndexError(f"Frame index outside pose array for {sample['sample_id']}")
        w2cs = torch.from_numpy(scene_poses[slot_indices]).float()
        _, c2w_norm, scale = normalize_w2c_make_cam_last_origin(w2cs)
        raymap = get_equirectangular_plucker_rays(
            c2w_norm,
            height=self.height,
            width=self.width,
            no_pixel_unshuffle=self.no_pixel_unshuffle,
        )

        selected = set(noisy_input_ids)
        return {
            "input_images": input_images,
            "target_images": target_images,
            "raymap": raymap,
            "prompt": "",
            "sample_id": sample["sample_id"],
            "scene_name": sample["scene_name"],
            "input_roles": ["clean_reference"] * 2 + ["noisy_render"] * num_noisy,
            "target_task_types": [
                "cleanup" if idx in selected else "nvs" for idx in range(3)
            ],
            "noisy_input_target_ids": noisy_input_ids,
            "slot_frame_indices": slot_indices,
            "pose_scale": float(scale),
        }
