"""Equirectangular panorama sets (FrameCrafter / OmniCrafter 5-input data) in the
RayCrafter batch format.

Each item is 2 clean reference panoramas and 3 query slots, with the clean
target for every query: the same layout and keys as ``dl3dv_points``
(``src``, ``tgt``, ``ref``, cameras, ``noise_alpha``), plus
``camera_model = "equirect"``. Queries get the panorama render-query
degradation (pixel dropout drawn from the integer percents in
``[min_pixel_dropout, max_pixel_dropout]``, black queries via ``nvs_ratio`` or
``nvs_probability``), as in OmniCrafter's
``pose_conditioned_render_query_dataset.py``.

Manifests are JSONL with one row per query. Rows sharing
``original_source_sample`` (or ``source_sample`` in the top-level manifest)
share their references and form one set; only sets with exactly
``num_views`` queries are kept.

Cameras: ``poses/framecrafter/<dataset>/<scene>.npz`` holds OpenCV w2c per
frame. The equirect convention (checked against the data and the OmniCrafter
code) is that of ``cameras.unirope.equirect_directions``: longitude -π..π
left to right, latitude π/2..-π/2 top to bottom, image center = +z. There are
no intrinsics, so ``camera_fxfycxcy`` is zero. With a near-depth map
(``scripts/estimate_near_depth_panorama.py``) translations are divided by
their scene's near depth, SCoPE's normalization and the unit ``dl3dv_points``
uses, and sets of scenes without one are dropped. Without it they are divided
by the mean distance of the five cameras to the last query camera
(OmniCrafter's gauge). Either way they are first re-centered on the last query
camera, which changes nothing downstream (everything is made relative to a
camera) but keeps float32 precise. RayRoPE / UniRoPE renormalize on their own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from webdataset.utils import pytorch_worker_info

from .dl3dv_points import _rng, load_near_depth, pixel_dropout

CAMERA_MODEL = "equirect"
# In raw pose units: below this mean distance a set is a pure rotation.
MIN_BASELINE = 1e-3
_IMAGE = re.compile(r"^(?P<dir>.+)/(?P<frame>\d+)\.png$")


def _split_frame(path: str) -> Tuple[str, int]:
    """``"HM3D/images/<scene>/000074.png"`` → (``"HM3D/images/<scene>"``, 74)."""
    match = _IMAGE.match(path)
    if match is None:
        raise ValueError(f"unexpected image path {path!r}")
    return match["dir"], int(match["frame"])


class PanoramaSets(Dataset):
    """Grouped panorama sets: 2 references + ``num_views`` queries each."""

    def __init__(
        self,
        root: Union[str, Path],
        manifests: Sequence[Union[str, Path]],
        image_size: Tuple[int, int] = (256, 512),
        use_pose: bool = True,
        num_views: int = 3,
        min_pixel_dropout: float = 0.0,
        max_pixel_dropout: float = 1.0,
        nvs_probability: float = 0.0,
        nvs_ratio: float = 0.0,
        datasets: Optional[Sequence[str]] = None,
        near_depth: Optional[Dict[str, float]] = None,
    ):
        self.root = Path(root)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        if self.width != 2 * self.height:
            raise ValueError(f"equirect images are 2:1, got {self.height}x{self.width}")
        self.use_pose = use_pose
        self.num_views = int(num_views)
        self.min_pixel_dropout = float(min_pixel_dropout)
        self.max_pixel_dropout = float(max_pixel_dropout)
        self.nvs_probability = float(nvs_probability)
        self.nvs_ratio = float(nvs_ratio)
        if not 0.0 <= self.min_pixel_dropout <= self.max_pixel_dropout <= 1.0:
            raise ValueError("pixel dropout must satisfy 0 <= min <= max <= 1")
        if self.nvs_probability > 0.0 and self.nvs_ratio > 0.0:
            raise ValueError("nvs_probability and nvs_ratio cannot both be set")
        poses = json.loads((self.root / "poses" / "manifest.json").read_text())
        if poses.get("pose_convention") != "w2c_opencv":
            raise ValueError("pose manifest must hold OpenCV w2c poses")
        self._pose_paths = {scene: entry["pose_path"] for scene, entry in poses["scenes"].items()}
        self._pose_cache: Dict[str, np.ndarray] = {}
        self.near_depth = near_depth
        self._index(manifests, set(datasets) if datasets else None)

    def _index(self, manifests: Sequence[Union[str, Path]], datasets: Optional[set]) -> None:
        """Group rows into sets and keep them as fixed-width numpy arrays, so
        DataLoader workers share the index instead of copying Python objects."""
        groups: Dict[str, Dict[int, dict]] = {}
        for manifest in manifests:
            path = Path(manifest)
            path = path if path.is_absolute() else self.root / path
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if datasets is not None and row["dataset"] not in datasets:
                        continue
                    key = row.get("original_source_sample", row["source_sample"])
                    number = int(row.get("original_target_number", row["target_number"]))
                    groups.setdefault(key, {})[number] = row
        keys, scenes, image_dirs, src_dirs, ref_frames, target_frames = [], [], [], [], [], []
        incomplete = no_depth = 0
        for key in sorted(groups):
            rows = groups[key]
            if sorted(rows) != list(range(self.num_views)):
                incomplete += 1
                continue
            rows = [rows[i] for i in range(self.num_views)]
            first = rows[0]
            if len(first["refs"]) != 2 or any(r["refs"] != first["refs"] for r in rows):
                raise ValueError(f"{key}: the queries of a set must share two references")
            image_dir, refs = zip(*(_split_frame(p) for p in first["refs"]))
            target_dir, targets = zip(*(_split_frame(r["target"]) for r in rows))
            src_dir, srcs = zip(*(_split_frame(r["src"]) for r in rows))
            if len({*image_dir, *target_dir}) != 1 or len(set(src_dir)) != 1 or srcs != targets:
                raise ValueError(f"{key}: references, targets and renders must share their folders")
            if first["scene_name"] not in self._pose_paths:
                raise KeyError(f"{key}: no poses for {first['scene_name']}")
            if self.near_depth is not None and first["scene_name"] not in self.near_depth:
                no_depth += 1
                continue
            keys.append(key)
            scenes.append(first["scene_name"])
            image_dirs.append(image_dir[0])
            src_dirs.append(src_dir[0])
            ref_frames.append(refs)
            target_frames.append(targets)
        if not keys:
            raise ValueError(f"no complete {self.num_views}-query sets in {list(manifests)}")
        self.keys = np.array(keys, dtype=np.bytes_)
        self.scenes = np.array(scenes, dtype=np.bytes_)
        self.image_dirs = np.array(image_dirs, dtype=np.bytes_)
        self.src_dirs = np.array(src_dirs, dtype=np.bytes_)
        self.ref_frames = np.array(ref_frames, dtype=np.int32)
        self.target_frames = np.array(target_frames, dtype=np.int32)
        self.incomplete = incomplete
        self.dropped = no_depth

    def __len__(self) -> int:
        return len(self.keys)

    def _load(self, relative: str) -> Image.Image:
        with Image.open(self.root / relative) as image:
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = TF.resize(
                    image, [self.height, self.width],
                    interpolation=InterpolationMode.BILINEAR, antialias=True,
                )
            return image.copy()

    def _black(self) -> Image.Image:
        return Image.new("RGB", (self.width, self.height), (0, 0, 0))

    def _scene_w2c(self, scene: str) -> np.ndarray:
        if scene not in self._pose_cache:
            with np.load(self.root / self._pose_paths[scene]) as data:
                self._pose_cache[scene] = np.asarray(data["w2c_poses"], dtype=np.float64)
        return self._pose_cache[scene]

    def cameras(self, index: int) -> torch.Tensor:
        """c2w ``(5, 4, 4)`` ordered [ref, ref, query...]; see :meth:`frame_cameras`."""
        frames = np.concatenate((self.ref_frames[index], self.target_frames[index]))
        return self.frame_cameras(self.scenes[index].decode(), frames)

    def frame_cameras(self, scene: str, frames: Sequence[int]) -> torch.Tensor:
        """c2w ``(N, 4, 4)`` of ``frames`` of ``scene``, re-centered on the last
        one, translations divided by the scene's near depth or, without a
        near-depth map, by the mean camera distance to the last one.

        In the second case, pure-rotation sets (all centers within
        ``MIN_BASELINE``; ~3% of HM3D) get zero translation: their offsets are
        float noise, which the division would blow up to unit length in random
        directions."""
        frames = np.asarray(frames)
        w2c = self._scene_w2c(scene)
        if frames.max() >= len(w2c):
            raise IndexError(f"{scene}: frame {frames.max()} outside the pose array")
        c2w = np.linalg.inv(w2c[frames])
        centers = c2w[:, :3, 3] - c2w[-1, :3, 3]
        if self.near_depth is not None:
            c2w[:, :3, 3] = centers / self.near_depth[scene]
        else:
            scale = np.linalg.norm(centers, axis=-1).mean()
            c2w[:, :3, 3] = centers / scale if scale > MIN_BASELINE else 0.0
        return torch.from_numpy(c2w).float()

    def __getitem__(self, index: int) -> Dict:
        image_dir = self.image_dirs[index].decode()
        src_dir = self.src_dirs[index].decode()
        refs = [self._load(f"{image_dir}/{f:06d}.png") for f in self.ref_frames[index]]
        targets = [self._load(f"{image_dir}/{f:06d}.png") for f in self.target_frames[index]]

        rng = _rng()
        # Same control flow as dl3dv_points / OmniCrafter: no extra draw when a
        # switch is off, and stored renders are only read when they are used.
        if self.nvs_probability > 0.0 and rng.random() < self.nvs_probability:
            queries = [self._black() for _ in targets]
        else:
            queries = []
            for frame in self.target_frames[index]:
                if self.nvs_ratio > 0.0 and rng.random() < self.nvs_ratio:
                    queries.append(self._black())
                else:
                    render = self._load(f"{src_dir}/{frame:06d}.png")
                    queries.append(pixel_dropout(render, self.min_pixel_dropout, self.max_pixel_dropout, rng))

        def model(image: Image.Image) -> torch.Tensor:
            return TF.to_tensor(image) * 2 - 1

        out = {
            "src": torch.stack([model(image) for image in queries]),
            "tgt": torch.stack([model(image) for image in targets]),
            "ref": torch.stack([model(image) for image in refs]),
            "key": self.keys[index].decode(),
            "noise_alpha": 0.0,
            "camera_model": CAMERA_MODEL,
        }
        if self.use_pose:
            out["camera_c2w"] = self.cameras(index)
            out["camera_fxfycxcy"] = torch.zeros(2 + self.num_views, 4)
        return out


def make_panorama_dataloader(
    root: Union[str, Path],
    manifests: Union[str, List[str]],
    num_views: int,
    ref_size: int,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 1,
    use_pose: bool = True,
    shuffle: bool = True,
    min_pixel_dropout: float = 0.0,
    max_pixel_dropout: float = 1.0,
    nvs_probability: float = 0.0,
    nvs_ratio: float = 0.0,
    datasets: Optional[Sequence[str]] = None,
    near_depth_json: Optional[str] = None,
    limit: Optional[int] = None,
) -> DataLoader:
    """``limit`` keeps that many sets, evenly spaced over the (sorted) index: a
    validation subset that covers every scene instead of the first few."""
    if ref_size != 2:
        raise ValueError(f"panorama sets have 2 references, got ref_size={ref_size}")
    manifests = [manifests] if isinstance(manifests, str) else list(manifests)
    ds = PanoramaSets(
        root, manifests, image_size=tuple(image_size), use_pose=use_pose, num_views=num_views,
        min_pixel_dropout=min_pixel_dropout, max_pixel_dropout=max_pixel_dropout,
        nvs_probability=nvs_probability, nvs_ratio=nvs_ratio, datasets=datasets,
        near_depth=load_near_depth(near_depth_json) if near_depth_json else None,
    )
    if pytorch_worker_info()[0] == 0:
        print(
            f"data: {len(ds)} panorama sets ({ds.incomplete} incomplete dropped) from "
            f"{', '.join(manifests)} (dropout {min_pixel_dropout:g}..{max_pixel_dropout:g}, "
            f"nvs_probability={nvs_probability:g}, nvs_ratio={nvs_ratio:g})"
            + (f", near depth from {near_depth_json} ({ds.dropped} dropped)" if near_depth_json else ""),
            flush=True,
        )
    from .dataset import _collate

    if limit is not None and limit < len(ds):
        ds = Subset(ds, np.linspace(0, len(ds) - 1, limit).round().astype(int).tolist())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=_collate,
    )
