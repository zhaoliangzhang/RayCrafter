"""DL3DV point-splat samples in the RayCrafter batch format.

Each directory under ``<split>/priors/samples/`` is one item: 2 clean reference
images, 3 point-splat priors, 3 clean targets, and OpenCV w2c + pinhole K in
``metadata.json``. The batch is the same dict the WebDataset loader returns
(``src``, ``tgt``, ``ref``, cameras, ``noise_alpha``), so training is unchanged.

Query priors use the panorama render-query dropout schedule. A percent is drawn
uniformly from the integers in ``[min_pixel_dropout, max_pixel_dropout]`` (so
100% is reachable), and that fraction of pixels is set to black.
``nvs_probability`` replaces every query in the sample with black;
``nvs_ratio`` does that independently per query, and otherwise the query gets
the cleanup dropout. The two cannot both be set. Cameras stay pinhole, so
SCoPE / RayRoPE build the same Plücker rays as the tar path.

With a near-depth map (``scripts/estimate_near_depth_dl3dv_points.py``), each
camera translation is divided by its scene's near depth, SCoPE's
normalization; scenes without an estimate are dropped.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from webdataset.utils import pytorch_worker_info

_WORKER_RNG: Dict[int, random.Random] = {}


def _rng():
    """Generator used for one sample.

    Draws match the panorama loader (``random.randint`` is inclusive, so a
    100% dropout is reachable). Each DataLoader worker keeps its own stream;
    a shared one would repeat the same masks on every worker.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return random
    rng = _WORKER_RNG.get(info.id)
    if rng is None:
        rng = random.Random(torch.initial_seed())
        _WORKER_RNG[info.id] = rng
    return rng


def _split_root(shards: Union[str, List[str]]) -> Path:
    if isinstance(shards, (list, tuple)):
        if len(shards) != 1:
            raise ValueError(f"dl3dv_points takes one split directory, got {len(shards)}")
        shards = shards[0]
    root = Path(str(shards))
    samples = root / "priors" / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"no priors/samples under {root}")
    return root


def _scale_k(cam: dict, src_wh: Tuple[int, int], dst_hw: Tuple[int, int]) -> torch.Tensor:
    """K at the resized image, in pixels. Resize stretches, so x and y scale apart."""
    kw, kh = float(cam["width"]), float(cam["height"])
    sw, sh = src_wh
    k = cam["K"]
    fx, fy, cx, cy = k[0][0] * sw / kw, k[1][1] * sh / kh, k[0][2] * sw / kw, k[1][2] * sh / kh
    height, width = dst_hw
    sx, sy = width / sw, height / sh
    return torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32)


def _c2w(w2c) -> torch.Tensor:
    """OpenCV world-to-camera → camera-to-world. No axis flip (already OpenCV)."""
    return torch.linalg.inv(torch.tensor(w2c, dtype=torch.float64)).to(torch.float32)


def pixel_dropout(image: Image.Image, min_dropout: float, max_dropout: float, rng) -> Image.Image:
    """Black out a fraction of pixels drawn uniformly from the integer percents
    in ``[min_dropout, max_dropout]`` (inclusive, so 100% is reachable)."""
    min_percent = int(round(min_dropout * 100.0))
    max_percent = int(round(max_dropout * 100.0))
    dropout_ratio = rng.randint(min_percent, max_percent) / 100.0
    pixels = np.asarray(image, dtype=np.uint8).copy()
    pixel_rng = np.random.default_rng(rng.getrandbits(64))
    remove = pixel_rng.random((image.height, image.width)) < dropout_ratio
    pixels[remove] = 0
    return Image.fromarray(pixels, mode="RGB")


def load_near_depth(path: Union[str, Path]) -> Dict[str, float]:
    """``{scene_name: {"near_depth": d, ...}}`` → ``{scene_name: d}``, skipping failed scenes."""
    records = json.loads(Path(path).read_text())
    return {scene: float(r["near_depth"]) for scene, r in records.items() if r.get("near_depth")}


def scale_translation(c2w: torch.Tensor, near_depth: float) -> torch.Tensor:
    """Divide the translation of ``(..., 4, 4)`` c2w by the scene's near depth."""
    out = c2w.clone()
    out[..., :3, 3] /= near_depth
    return out


def _scene_name(sample_dir: Path) -> str:
    return json.loads((sample_dir / "metadata.json").read_text())["scene_name"]


class Dl3dvPoints(Dataset):
    """One sample directory → one RayCrafter item (2 refs, 3 views)."""

    def __init__(
        self,
        root: Path,
        image_size: Tuple[int, int],
        use_pose: bool,
        min_pixel_dropout: float = 0.0,
        max_pixel_dropout: float = 1.0,
        nvs_probability: float = 0.0,
        nvs_ratio: float = 0.0,
        near_depth: Optional[Dict[str, float]] = None,
    ):
        self.root = root
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.use_pose = use_pose
        self.min_pixel_dropout = float(min_pixel_dropout)
        self.max_pixel_dropout = float(max_pixel_dropout)
        self.nvs_probability = float(nvs_probability)
        self.nvs_ratio = float(nvs_ratio)
        if not 0.0 <= self.min_pixel_dropout <= self.max_pixel_dropout <= 1.0:
            raise ValueError("pixel dropout must satisfy 0 <= min <= max <= 1")
        if self.nvs_probability > 0.0 and self.nvs_ratio > 0.0:
            raise ValueError("nvs_probability and nvs_ratio cannot both be set")
        samples = root / "priors" / "samples"
        self.samples = sorted(p for p in samples.iterdir() if (p / "metadata.json").is_file())
        if not self.samples:
            raise FileNotFoundError(f"no sample directories in {samples}")
        self.near_depth: Optional[List[float]] = None
        self.dropped = 0
        if near_depth is not None:
            with ThreadPoolExecutor(16) as pool:
                scenes = list(pool.map(_scene_name, self.samples))
            kept = [(path, near_depth[scene]) for path, scene in zip(self.samples, scenes) if scene in near_depth]
            if not kept:
                raise ValueError(f"no scene under {samples} has a near depth")
            self.samples = [path for path, _ in kept]
            self.near_depth = [depth for _, depth in kept]
            self.dropped = len(scenes) - len(kept)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> Tuple[Image.Image, Tuple[int, int]]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            src_wh = image.size
            if src_wh != (self.width, self.height):
                image = TF.resize(
                    image, [self.height, self.width],
                    interpolation=InterpolationMode.BILINEAR, antialias=True,
                )
            return image.copy(), src_wh

    def _degrade(self, image: Image.Image, rng: random.Random) -> Image.Image:
        return pixel_dropout(image, self.min_pixel_dropout, self.max_pixel_dropout, rng)

    def _black(self) -> Image.Image:
        return Image.new("RGB", (self.width, self.height), (0, 0, 0))

    def __getitem__(self, index: int) -> Dict:
        sample_dir = self.samples[index]
        meta = json.loads((sample_dir / "metadata.json").read_text())
        refs = meta["references"]
        targets = sorted(meta["targets"], key=lambda row: row["target_number"])
        if len(refs) != 2 or len(targets) != 3:
            raise ValueError(f"{sample_dir.name} has {len(refs)} refs and {len(targets)} targets")

        ref_imgs, ref_wh = [], []
        for i in range(2):
            image, wh = self._load(sample_dir / f"ref_{i}.png")
            ref_imgs.append(image)
            ref_wh.append(wh)
        gt_imgs, gt_wh, priors = [], [], []
        for i in range(3):
            gt, wh = self._load(sample_dir / f"gt_{i}.png")
            prior, _ = self._load(sample_dir / f"prior_{i}.png")
            gt_imgs.append(gt)
            gt_wh.append(wh)
            priors.append(prior)

        rng = _rng()
        # Same control flow as the panorama loader: no extra draw when the
        # sample-level switch is off, so a cleanup-only run stays reproducible.
        if self.nvs_probability > 0.0 and rng.random() < self.nvs_probability:
            queries = [self._black() for _ in priors]
        else:
            queries = []
            for image in priors:
                if self.nvs_ratio > 0.0 and rng.random() < self.nvs_ratio:
                    queries.append(self._black())
                else:
                    queries.append(self._degrade(image, rng))

        def model(image: Image.Image) -> torch.Tensor:
            return TF.to_tensor(image) * 2 - 1

        out = {
            "src": torch.stack([model(image) for image in queries]),
            "tgt": torch.stack([model(image) for image in gt_imgs]),
            "ref": torch.stack([model(image) for image in ref_imgs]),
            "key": sample_dir.name,
            # Degradation is the pixel dropout above, not latent noise.
            "noise_alpha": 0.0,
        }
        if self.use_pose:
            dst = (self.height, self.width)
            cams = [_c2w(row["w2c"]) for row in refs] + [_c2w(row["w2c"]) for row in targets]
            intr = [_scale_k(row["camera"], wh, dst) for row, wh in zip(refs, ref_wh)]
            intr += [_scale_k(row["camera"], wh, dst) for row, wh in zip(targets, gt_wh)]
            out["camera_c2w"] = torch.stack(cams)
            out["camera_fxfycxcy"] = torch.stack(intr)
            if self.near_depth is not None:
                out["camera_c2w"] = scale_translation(out["camera_c2w"], self.near_depth[index])
        return out


def make_points_dataloader(
    shards: Union[str, List[str]],
    num_views: int,
    ref_size: int,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 1,
    use_pose: bool = False,
    shuffle: bool = True,
    min_pixel_dropout: float = 0.0,
    max_pixel_dropout: float = 1.0,
    nvs_probability: float = 0.0,
    nvs_ratio: float = 0.0,
    near_depth_json: Optional[str] = None,
) -> DataLoader:
    if num_views != 3 or ref_size != 2:
        raise ValueError(
            "dl3dv_points samples are 2 refs + 3 targets, "
            f"got ref_size={ref_size} num_views={num_views}"
        )
    root = _split_root(shards)
    ds = Dl3dvPoints(
        root, image_size=tuple(image_size), use_pose=use_pose,
        min_pixel_dropout=min_pixel_dropout, max_pixel_dropout=max_pixel_dropout,
        nvs_probability=nvs_probability, nvs_ratio=nvs_ratio,
        near_depth=load_near_depth(near_depth_json) if near_depth_json else None,
    )
    if pytorch_worker_info()[0] == 0:
        near = f", near depth from {near_depth_json} ({ds.dropped} dropped)" if near_depth_json else ""
        print(
            f"data: {len(ds)} dl3dv_points samples from {root} "
            f"(dropout {min_pixel_dropout:g}..{max_pixel_dropout:g}, "
            f"nvs_probability={nvs_probability:g}, nvs_ratio={nvs_ratio:g}{near})",
            flush=True,
        )
    from .dataset import _collate

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=_collate,
    )
