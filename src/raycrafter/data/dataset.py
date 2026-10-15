"""WebDataset loader. Shard format is documented in the README.

Each batch is a dict:
    src, tgt         (B, V, 3, H, W) in [-1, 1]   degraded input / clean target views
    ref              (B, R, 3, H, W) in [-1, 1]   clean reference views
    camera_c2w       (B, R+V, 4, 4)               only when use_pose; order [ref..., views...]
    camera_fxfycxcy  (B, R+V, 4)                  pixels at image_size
    noise_alpha      (B,)                         curriculum noise level (0 = real degraded render)
    key              list[str]
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import braceexpand
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from .curriculum import (
    CurriculumShardList,
    TarfileToSamplesWithMeta,
    inject_query_noise_tensor,
    parse_float_list,
)
from .multiview import MultiViewConfig, MultiViewStage


def expand_shards(shards: Union[str, List[str]]) -> List[str]:
    specs = [shards] if isinstance(shards, str) else list(shards)
    return [url for spec in specs if spec for url in braceexpand.braceexpand(spec)]


def _load_image(data: bytes, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    """PNG bytes → (3, H, W) in [0, 1], resized to ``size`` if given."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if size is not None:
        img = TF.resize(img, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.to_tensor(img)


def _colmap_fxfycxcy(cam: dict, h: int, w: int) -> torch.Tensor:
    p, sx, sy = cam["params"], w / cam["width"], h / cam["height"]
    return torch.tensor([p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy], dtype=torch.float32)


def _parse_camera(raw: Any, h: int, w: int, ref: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """pose.json → (c2w (4, 4), fxfycxcy (4,) in pixels at (h, w)).

    Two layouts are accepted (``ref`` picks the reference camera's fields):
      flat:   {"camera_c2w", "camera_fxfycxcy", "ref_camera_c2w", "ref_camera_fxfycxcy"}
              with intrinsics normalized by image width / height;
      colmap: {"view": {"c2w", "camera"}, "ref": {"c2w", "camera"}} where camera
              is {"width", "height", "params": [fx, fy, cx, cy, ...]} in pixels.
    """
    cam = json.loads(raw.decode("utf-8")) if isinstance(raw, (bytes, bytearray)) else raw
    prefix = "ref_camera" if ref else "camera"
    if f"{prefix}_c2w" in cam:
        c2w = torch.tensor(cam[f"{prefix}_c2w"], dtype=torch.float32)
        fxfycxcy = torch.tensor(cam[f"{prefix}_fxfycxcy"], dtype=torch.float32)
        return c2w, fxfycxcy * torch.tensor([w, h, w, h], dtype=torch.float32)
    entry = cam["ref" if ref else "view"]
    return torch.tensor(entry["c2w"], dtype=torch.float32), _colmap_fxfycxcy(entry["camera"], h, w)


def make_dataloader(
    shards: Union[str, List[str]],
    num_views: int,
    ref_size: int,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 1,
    use_pose: bool = False,
    shuffle: bool = True,
    window: int = 15,
    shuffle_buffer: int = 20,
    enable_curriculum: bool = False,
    curriculum_sigma: float = 0.16,
    curriculum_min_mass: float = 0.10,
    curriculum_start_center: float = 0.0,
    curriculum_end_center: float = 0.52,
    curriculum_start_probs=None,
    curriculum_end_probs=None,
    noise_levels=None,
    noise_mode: str = "lerp",
    noise_space: str = "latent",
    curriculum_progress=None,
    repeat: bool = True,
) -> DataLoader:
    """Training uses the curriculum shard sampler (infinite) whenever curriculum or
    noise levels are on; otherwise a plain shard list (repeated if ``repeat``)."""
    h, w = int(image_size[0]), int(image_size[1])
    urls = expand_shards(shards)
    if not urls:
        raise ValueError("make_dataloader: empty shard list")
    levels = parse_float_list(noise_levels)
    use_curriculum = bool(enable_curriculum) or bool(levels)

    if use_curriculum:
        pipeline: List[Any] = [CurriculumShardList(
            urls,
            progress=curriculum_progress,
            enable_curriculum=bool(enable_curriculum),
            curriculum_sigma=curriculum_sigma,
            curriculum_min_mass=curriculum_min_mass,
            curriculum_start_center=curriculum_start_center,
            curriculum_end_center=curriculum_end_center,
            curriculum_start_probs=curriculum_start_probs,
            curriculum_end_probs=curriculum_end_probs,
            noise_levels=levels or None,
            seed=0,
        )]
        tar_stage = TarfileToSamplesWithMeta()
    else:
        pipeline = [wds.SimpleShardList(urls)]
        tar_stage = wds.tarfile_to_samples(handler=wds.warn_and_continue)

    if shuffle:
        pipeline.append(wds.shuffle(shuffle_buffer, handler=wds.warn_and_continue))
    pipeline.append(wds.split_by_node)
    if shuffle:
        pipeline.append(wds.shuffle(shuffle_buffer, handler=wds.warn_and_continue))
    pipeline.extend([wds.split_by_worker, tar_stage])
    stage = MultiViewStage(MultiViewConfig(num_views=num_views, num_refs=ref_size, window=window))
    pipeline.append(wds.map(stage, handler=wds.warn_and_continue))

    def process(sample: Dict) -> Optional[Dict]:
        try:
            alpha = float(sample.get("__noise_alpha__", sample.get("noise_alpha", 0.0)))
            src, tgt = [], []
            for i in range(1, num_views + 1):
                s = _load_image(sample[f"img{i}.png"], (h, w))
                g = _load_image(sample[f"gt{i}.png"], (h, w))
                if alpha > 0.0 and noise_space == "pixel":
                    s = inject_query_noise_tensor(g, alpha, mode=noise_mode)
                src.append(s * 2 - 1)
                tgt.append(g * 2 - 1)
            refs = [_load_image(sample[f"ref{r}.png"], (h, w)) * 2 - 1 for r in range(1, ref_size + 1)]
            out = {
                "src": torch.stack(src),
                "tgt": torch.stack(tgt),
                "ref": torch.stack(refs),
                "key": sample.get("__key__", ""),
                "noise_alpha": alpha,
            }
            if use_pose:
                cams = [_parse_camera(sample[f"ref_camera{r}.json"], h, w, ref=True)
                        for r in range(1, ref_size + 1)]
                cams += [_parse_camera(sample[f"camera{i}.json"], h, w, ref=False)
                         for i in range(1, num_views + 1)]
                out["camera_c2w"] = torch.stack([c[0] for c in cams])
                out["camera_fxfycxcy"] = torch.stack([c[1] for c in cams])
            return out
        except Exception as exc:
            # Loud, so a format problem does not silently filter every sample.
            print(f"[raycrafter.data] skip sample: {exc!r}", flush=True)
            return None

    pipeline.append(wds.map(process, handler=wds.warn_and_continue))
    pipeline.append(wds.select(lambda x: x is not None))
    ds = wds.DataPipeline(*pipeline)
    if repeat and not use_curriculum:
        ds = ds.repeat()

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=_collate)


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("src", "tgt", "ref")}
    if "camera_c2w" in batch[0]:
        out["camera_c2w"] = torch.stack([b["camera_c2w"] for b in batch])
        out["camera_fxfycxcy"] = torch.stack([b["camera_fxfycxcy"] for b in batch])
    out["noise_alpha"] = torch.tensor([b["noise_alpha"] for b in batch])
    out["key"] = [b["key"] for b in batch]
    return out
