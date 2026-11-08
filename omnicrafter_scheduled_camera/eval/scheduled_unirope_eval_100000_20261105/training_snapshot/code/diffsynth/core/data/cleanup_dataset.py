"""Paired panorama dataset for multi-view render cleanup.

Each JSONL row describes one noisy/clean target pair and two clean references.
Rows are grouped by ``source_sample`` into the fixed task:

    2 clean references + 3 noisy renders -> 3 clean targets

The three targets remain separate temporal slots in Wan. This preserves the
pretrained RGB VAE while allowing all output slots to interact through DiT
self-attention.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import fcntl
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class _LazyCleanupSamples(Sequence):
    """Map-style JSONL group view that keeps only byte offsets in memory."""

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


class WanPanoramaCleanupDataset(Dataset):
    """Load fixed 2-reference/3-render cleanup samples from JSONL metadata."""

    def __init__(
        self,
        base_path,
        metadata_path,
        repeat=1,
        height=256,
        width=512,
        max_samples=None,
        stream_metadata=False,
        min_pixel_dropout=0.0,
        max_pixel_dropout=0.0,
    ):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.repeat = repeat
        self.height = height
        self.width = width
        self.stream_metadata = stream_metadata
        self.min_pixel_dropout = min_pixel_dropout
        self.max_pixel_dropout = max_pixel_dropout
        self._metadata_handle = None
        self._metadata_handle_pid = None

        if self.width != 2 * self.height:
            raise ValueError(
                f"Panorama cleanup requires a 2:1 resolution, got "
                f"{self.width}x{self.height}"
            )
        if self.repeat < 1:
            raise ValueError(f"repeat must be positive, got {self.repeat}")
        if not 0.0 <= min_pixel_dropout <= max_pixel_dropout <= 1.0:
            raise ValueError("pixel dropout must satisfy 0 <= min <= max <= 1")

        if stream_metadata:
            self._stream_offsets = self._load_or_build_stream_index()
            sample_count = len(self._stream_offsets) - 1
            if max_samples is not None and max_samples > 0:
                sample_count = min(sample_count, max_samples)
            self.samples = _LazyCleanupSamples(self, sample_count)
        else:
            groups = defaultdict(list)
            with self.metadata_path.open() as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        groups[row["source_sample"]].append(row)
                    except KeyError as error:
                        raise ValueError(
                            f"Missing {error} in {self.metadata_path}:{line_number}"
                        ) from error

            self.samples = []
            for sample_id in sorted(groups):
                rows = sorted(groups[sample_id], key=lambda row: row["target_number"])
                self._validate_group(sample_id, rows)
                self.samples.append(self._sample_from_rows(sample_id, rows))

            if max_samples is not None and max_samples > 0:
                self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No complete cleanup samples found in {self.metadata_path}")

        print(
            f"Total cleanup samples: {len(self.samples)} "
            f"({len(self)} with repeat={self.repeat}, "
            f"stream_metadata={self.stream_metadata})"
        )

    def _sample_from_rows(self, sample_id, rows):
        refs = rows[0]["refs"]
        return {
            "sample_id": sample_id,
            "scene_name": rows[0]["scene_name"],
            "refs": [self._resolve(path) for path in refs],
            "noisy": [self._resolve(row["src"]) for row in rows],
            "targets": [self._resolve(row["target"]) for row in rows],
            # Coverage is diagnostic-only and is absent from the large-scale
            # manifest.  Keep the legacy field with a neutral numeric value so
            # default DataLoader collation remains valid.
            "coverage": [float(row.get("coverage", 0.0)) for row in rows],
            "frame_indices": [int(row["frame_index"]) for row in rows],
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
                    if current_sample is not None and expected_target != 3:
                        raise ValueError(
                            f"Cleanup sample {current_sample} has {expected_target} "
                            "targets; expected 3"
                        )
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
        if expected_target != 3:
            raise ValueError(
                f"Cleanup sample {current_sample} has {expected_target} targets; expected 3"
            )
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
            raise ValueError(f"Empty streamed cleanup sample at offset {start}")
        sample_id = rows[0]["source_sample"]
        if any(row.get("source_sample") != sample_id for row in rows):
            raise ValueError(f"Streamed metadata group crosses samples at offset {start}")
        rows = sorted(rows, key=lambda row: row["target_number"])
        self._validate_group(sample_id, rows)
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
    def _validate_group(sample_id, rows):
        if len(rows) != 3:
            raise ValueError(f"Cleanup sample {sample_id} has {len(rows)} targets; expected 3")
        target_numbers = [row["target_number"] for row in rows]
        if target_numbers != [0, 1, 2]:
            raise ValueError(
                f"Cleanup sample {sample_id} has target numbers {target_numbers}; "
                "expected [0, 1, 2]"
            )
        refs = {tuple(row["refs"]) for row in rows}
        if len(refs) != 1 or len(rows[0]["refs"]) != 2:
            raise ValueError(f"Cleanup sample {sample_id} must have two shared references")

    def _load(self, path):
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

    def _degrade_query(self, image):
        min_percent = int(round(self.min_pixel_dropout * 100.0))
        max_percent = int(round(self.max_pixel_dropout * 100.0))
        dropout_ratio = random.randint(min_percent, max_percent) / 100.0
        if dropout_ratio == 0.0:
            return image, dropout_ratio
        pixels = np.asarray(image, dtype=np.uint8).copy()
        pixel_rng = np.random.default_rng(random.getrandbits(64))
        remove = pixel_rng.random((image.height, image.width)) < dropout_ratio
        pixels[remove] = 0
        return Image.fromarray(pixels, mode="RGB"), dropout_ratio

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        references = [self._load(path) for path in sample["refs"]]
        noisy = [self._load(path) for path in sample["noisy"]]
        degraded_queries = []
        dropout_ratios = []
        for image in noisy:
            degraded, dropout_ratio = self._degrade_query(image)
            degraded_queries.append(degraded)
            dropout_ratios.append(dropout_ratio)
        clean_targets = [self._load(path) for path in sample["targets"]]

        return {
            "input_images": references,
            "conditioning_images": degraded_queries,
            # The clean references occupy known latent slots. FlowMatchSFTLoss
            # supervises only the final three target slots.
            "target_images": references + clean_targets,
            "prompt": "",
            "sample_id": sample["sample_id"],
            "coverage": sample["coverage"],
            "frame_indices": sample["frame_indices"],
            "query_pixel_dropout_ratios": dropout_ratios,
        }
