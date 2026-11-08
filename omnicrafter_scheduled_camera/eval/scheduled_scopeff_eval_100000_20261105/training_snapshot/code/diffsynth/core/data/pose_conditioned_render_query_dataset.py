"""Pose-conditioned panorama cleanup/NVS dataset with query images.

Each target pose is accompanied by an aligned point-cloud render. Online pixel
dropout removes a randomly sampled fraction of render pixels while the target
pose remains available through the ERP Plucker raymap.

Two optional task-mixture controls are supported. ``nvs_probability`` keeps
the legacy sample-level behavior and replaces every query render in a sample
with black. ``nvs_ratio`` operates inside the 2D augmentation independently
for every query: with that probability its pixel-dropout rate is forced to
100%; otherwise the stored noisy render receives the configured cleanup
dropout. The latter permits mixed cleanup/NVS targets in a multi-query sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import fcntl
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


class _LazyMetadataSamples(Sequence):
    """Map-style view over JSONL groups without retaining every row in RAM."""

    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.dataset._read_streamed_sample(index)


class WanPanoramaPoseConditionedRenderQueryDataset(Dataset):
    """Return clean references and image-conditioned query slots.

    The temporal layout is ``M`` clean reference slots followed by ``N`` query
    slots. Query slots receive both an ERP pose raymap and an aligned degraded
    render; flow-matching supervision uses the corresponding clean image.
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
        min_pixel_dropout=0.0,
        max_pixel_dropout=1.0,
        no_pixel_unshuffle=False,
        deterministic_degradation=False,
        stream_metadata=False,
        max_query_slots=None,
        nvs_probability=0.0,
        nvs_ratio=0.0,
    ):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.repeat = repeat
        self.height = height
        self.width = width
        self.min_pixel_dropout = min_pixel_dropout
        self.max_pixel_dropout = max_pixel_dropout
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.deterministic_degradation = deterministic_degradation
        self.stream_metadata = stream_metadata
        self.max_query_slots = max_query_slots
        self.nvs_probability = nvs_probability
        self.nvs_ratio = nvs_ratio
        self._pose_cache = {}
        self._metadata_handle = None
        self._metadata_handle_pid = None

        if width != 2 * height:
            raise ValueError(f"Expected a 2:1 panorama, got {width}x{height}")
        if repeat < 1:
            raise ValueError(f"repeat must be positive, got {repeat}")
        if not 0.0 <= min_pixel_dropout <= max_pixel_dropout <= 1.0:
            raise ValueError("pixel dropout must satisfy 0 <= min <= max <= 1")
        if max_query_slots is not None and max_query_slots < 1:
            raise ValueError("max_query_slots must be positive when provided")
        if not 0.0 <= nvs_probability <= 1.0:
            raise ValueError("nvs_probability must be in [0, 1]")
        if not 0.0 <= nvs_ratio <= 1.0:
            raise ValueError("nvs_ratio must be in [0, 1]")
        if nvs_probability > 0.0 and nvs_ratio > 0.0:
            raise ValueError(
                "nvs_probability (sample-level) and nvs_ratio (query-level) "
                "cannot both be enabled"
            )

        manifest = json.loads(Path(pose_manifest_path).read_text())
        if manifest.get("pose_convention") != "w2c_opencv":
            raise ValueError("Pose manifest must contain OpenCV w2c poses")
        self.pose_manifest = manifest["scenes"]

        if stream_metadata:
            self._stream_offsets = self._load_or_build_stream_index()
            sample_count = len(self._stream_offsets) - 1
            if max_samples is not None:
                sample_count = min(sample_count, max_samples)
            self.samples = _LazyMetadataSamples(self, sample_count)
        else:
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
                rows = self._limit_query_rows(rows)
                self.samples.append(self._sample_from_rows(sample_id, rows))

            if max_samples is not None:
                self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No complete samples found in {metadata_path}")
        print(
            f"Total pose-conditioned render-query samples: {len(self.samples)} "
            f"({len(self)} with repeat={repeat}, stream_metadata={stream_metadata}, "
            f"nvs_probability={nvs_probability}, nvs_ratio={nvs_ratio})"
        )

    def _limit_query_rows(self, rows):
        if self.max_query_slots is None:
            return rows
        return rows[: self.max_query_slots]

    def _sample_from_rows(self, sample_id, rows):
        scene_name = rows[0]["scene_name"]
        if scene_name not in self.pose_manifest:
            raise KeyError(f"No pose manifest entry for {scene_name}")
        refs = rows[0]["refs"]
        return {
            "sample_id": sample_id,
            "scene_name": scene_name,
            "refs": [self._resolve(path) for path in refs],
            "ref_indices": [self._local_frame_index(path) for path in refs],
            "noisy": [self._resolve(row["src"]) for row in rows],
            "targets": [self._resolve(row["target"]) for row in rows],
            "target_indices": [int(row["frame_index"]) for row in rows],
        }

    def _stream_index_paths(self):
        source = self.metadata_path.resolve()
        prefix = Path(str(source) + ".sample_offsets")
        return (
            source,
            Path(str(prefix) + ".npy"),
            Path(str(prefix) + ".json"),
            Path(str(prefix) + ".lock"),
        )

    @staticmethod
    def _stream_index_is_current(source, index_path, info_path):
        if not index_path.is_file() or not info_path.is_file():
            return False
        try:
            info = json.loads(info_path.read_text())
            stat = source.stat()
            return (
                info.get("format_version") == 1
                and info.get("source_size") == stat.st_size
                and info.get("source_mtime_ns") == stat.st_mtime_ns
            )
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _scan_stream_offsets(source):
        offsets = []
        current_sample = None
        expected_target = 0
        line_number = 0
        with source.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    file_size = handle.tell()
                    break
                line_number += 1
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row.get("source_sample")
                if sample_id is None:
                    raise ValueError(f"Missing source_sample in {source}:{line_number}")
                target_number = int(row["target_number"])
                if sample_id != current_sample:
                    if target_number != 0:
                        raise ValueError(
                            f"First row for {sample_id} in {source}:{line_number} "
                            f"has target_number={target_number}, expected 0"
                        )
                    offsets.append(offset)
                    current_sample = sample_id
                    expected_target = 1
                else:
                    if target_number != expected_target:
                        raise ValueError(
                            f"Non-contiguous target_number in {source}:{line_number}: "
                            f"got {target_number}, expected {expected_target}"
                        )
                    expected_target += 1
        if not offsets:
            raise ValueError(f"No samples found in {source}")
        offsets.append(file_size)
        return np.asarray(offsets, dtype=np.uint64)

    def _load_or_build_stream_index(self):
        source, index_path, info_path, lock_path = self._stream_index_paths()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not self._stream_index_is_current(source, index_path, info_path):
                    print(f"Building streamed metadata index for {source}")
                    offsets = self._scan_stream_offsets(source)
                    stat = source.stat()
                    index_tmp = Path(f"{index_path}.{os.getpid()}.tmp")
                    info_tmp = Path(f"{info_path}.{os.getpid()}.tmp")
                    try:
                        with index_tmp.open("wb") as handle:
                            np.save(handle, offsets, allow_pickle=False)
                        info_tmp.write_text(
                            json.dumps(
                                {
                                    "format_version": 1,
                                    "source_size": stat.st_size,
                                    "source_mtime_ns": stat.st_mtime_ns,
                                    "num_samples": len(offsets) - 1,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        os.replace(index_tmp, index_path)
                        os.replace(info_tmp, info_path)
                    finally:
                        for temporary in (index_tmp, info_tmp):
                            if temporary.exists():
                                temporary.unlink()
                offsets = np.load(index_path, mmap_mode="r", allow_pickle=False)
        except OSError as error:
            print(
                f"Warning: could not use metadata sidecar index ({error}); "
                "building an in-memory offset table"
            )
            offsets = self._scan_stream_offsets(source)
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError(f"Invalid streamed metadata index for {source}")
        return offsets

    def _metadata_stream(self):
        pid = os.getpid()
        if self._metadata_handle is None or self._metadata_handle_pid != pid:
            if self._metadata_handle is not None:
                self._metadata_handle.close()
            self._metadata_handle = self.metadata_path.resolve().open("rb")
            self._metadata_handle_pid = pid
        return self._metadata_handle

    def _read_streamed_sample(self, index):
        start = int(self._stream_offsets[index])
        end = int(self._stream_offsets[index + 1])
        handle = self._metadata_stream()
        handle.seek(start)
        rows = [
            json.loads(line)
            for line in handle.read(end - start).splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Empty streamed sample at offset {start}")
        sample_id = rows[0]["source_sample"]
        if any(row.get("source_sample") != sample_id for row in rows):
            raise ValueError(f"Streamed metadata group crosses samples at offset {start}")
        rows = sorted(rows, key=lambda row: row["target_number"])
        self._validate_group(sample_id, rows)
        rows = self._limit_query_rows(rows)
        return self._sample_from_rows(sample_id, rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_metadata_handle"] = None
        state["_metadata_handle_pid"] = None
        return state

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
        if not rows[0]["refs"]:
            raise ValueError(f"{sample_id} must contain at least one reference")
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

    def _rng(self, sample_id, repeated_index):
        if not self.deterministic_degradation:
            return random
        digest = hashlib.sha256(
            f"{sample_id}:{repeated_index}".encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))

    def _degrade_query(self, image, rng):
        # Sample percentage points so the 100% endpoint has non-zero probability.
        min_percent = int(round(self.min_pixel_dropout * 100.0))
        max_percent = int(round(self.max_pixel_dropout * 100.0))
        dropout_ratio = rng.randint(min_percent, max_percent) / 100.0
        pixels = np.asarray(image, dtype=np.uint8).copy()
        pixel_rng = np.random.default_rng(rng.getrandbits(64))
        remove = pixel_rng.random((image.height, image.width)) < dropout_ratio
        pixels[remove] = 0
        return Image.fromarray(pixels, mode="RGB"), dropout_ratio

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        rng = self._rng(sample["sample_id"], index // len(self.samples))

        references = [self._load_image(path) for path in sample["refs"]]
        clean_targets = [self._load_image(path) for path in sample["targets"]]

        # Keep the default cleanup-only path bit-for-bit compatible: do not
        # consume an extra RNG draw when nvs_probability is zero.
        is_nvs_sample = (
            self.nvs_probability > 0.0
            and rng.random() < self.nvs_probability
        )
        if is_nvs_sample:
            degraded_queries = [
                Image.new("RGB", (self.width, self.height), (0, 0, 0))
                for _ in sample["noisy"]
            ]
            dropout_ratios = [1.0] * len(degraded_queries)
            query_condition_types = ["all_black_nvs"] * len(degraded_queries)
            query_task_types = ["nvs"] * len(degraded_queries)
        else:
            # Always load the stored noisy renders. Query-level NVS is part of
            # online 2D augmentation and never rewrites the training samples.
            noisy_renders = [self._load_image(path) for path in sample["noisy"]]
            degraded_queries = []
            dropout_ratios = []
            query_condition_types = []
            query_task_types = []
            for image in noisy_renders:
                # Avoid an extra RNG draw when nvs_ratio is disabled so prior
                # cleanup and sample-level NVS runs remain reproducible.
                is_nvs_query = (
                    self.nvs_ratio > 0.0 and rng.random() < self.nvs_ratio
                )
                if is_nvs_query:
                    degraded_queries.append(
                        Image.new("RGB", (self.width, self.height), (0, 0, 0))
                    )
                    dropout_ratios.append(1.0)
                    query_condition_types.append("all_black_nvs")
                    query_task_types.append("nvs")
                else:
                    degraded, dropout_ratio = self._degrade_query(image, rng)
                    degraded_queries.append(degraded)
                    dropout_ratios.append(dropout_ratio)
                    query_condition_types.append("augmented_noisy_render")
                    query_task_types.append("cleanup")

        if all(task == "nvs" for task in query_task_types):
            sample_task_type = "nvs"
        elif all(task == "cleanup" for task in query_task_types):
            sample_task_type = "cleanup"
        else:
            sample_task_type = "mixed_cleanup_nvs"

        slot_indices = sample["ref_indices"] + sample["target_indices"]
        scene_poses = self._scene_poses(sample["scene_name"])
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

        num_references = len(references)
        num_queries = len(degraded_queries)
        return {
            "input_images": references,
            "conditioning_images": degraded_queries,
            "target_images": references + clean_targets,
            "raymap": raymap,
            "prompt": "",
            "sample_id": sample["sample_id"],
            "scene_name": sample["scene_name"],
            "slot_roles": ["reference"] * num_references + ["query"] * num_queries,
            # Diagnostic only. WanTrainingModule does not pass this field to the
            # DiT, so the network is not told the sampled dropout ratio.
            "query_pixel_dropout_ratios": dropout_ratios,
            "target_task_types": query_task_types,
            "sample_task_type": sample_task_type,
            "query_condition_types": query_condition_types,
            "slot_frame_indices": slot_indices,
            "pose_scale": float(scale),
        }
