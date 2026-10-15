"""DreamSim-bucket curriculum for WebDataset shard sampling.

Mirrors FrameCrafter's ``cleanup_dataset`` schedule: group ``train_shards`` by
the ``dsim_*`` directory in the path, then sample the next tar with a moving
Gaussian whose peak slides from easy (``dsim_lt_0.25``) to hard
(``dsim_ge_0.4``) as training progress goes 0 -> 1.

Optional ``noise_levels`` add synthetic query-noise buckets (``noise_0.25``, …):
when one is selected, a random tar is opened and degraded inputs are replaced
at decode time by ``lerp(gt, uniform_noise, alpha)`` (FrameCrafter-compatible).
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

try:
    from webdataset.pytorch import IterableDataset
except ModuleNotFoundError:  # pragma: no cover
    from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

_DSIM_DIR = re.compile(r"(dsim_[^/]+)")
DEFAULT_DSIM_ORDER = (
    "dsim_lt_0.25",
    "dsim_0.25_0.3",
    "dsim_0.3_0.35",
    "dsim_0.35_0.4",
    "dsim_ge_0.4",
)


def bucket_from_tar_path(path: str) -> str:
    """Extract ``dsim_*`` from a shard URL / ``pipe:cat ...`` command."""
    match = _DSIM_DIR.search(str(path).replace("\\", "/"))
    if match:
        return match.group(1)
    # Fall back to parent dir name (e.g. .../train/pairs-000.tar -> train)
    cleaned = str(path).replace("pipe:cat", "").strip().split()[-1]
    return os.path.basename(os.path.dirname(os.path.abspath(cleaned))) or "other"


def order_dsim_buckets(names: Iterable[str]) -> List[str]:
    rank = {name: i for i, name in enumerate(DEFAULT_DSIM_ORDER)}
    names = list(names)
    known = [name for name in DEFAULT_DSIM_ORDER if name in names]
    extra = sorted(name for name in names if name not in rank)
    return known + extra


def group_shards_by_bucket(urls: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for url in urls:
        grouped[bucket_from_tar_path(url)].append(url)
    return {name: grouped[name] for name in order_dsim_buckets(grouped.keys())}


def gaussian_curriculum_probs(
    n: int,
    progress: float,
    start_center: float = 0.0,
    end_center: float = 0.52,
    sigma: float = 0.16,
    min_mass: float = 0.10,
) -> np.ndarray:
    """Softmax over bucket index; peak moves easy -> hard, with a uniform floor."""
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.ones((1,), dtype=np.float64)
    progress = float(np.clip(progress, 0.0, 1.0))
    centers = np.linspace(0.0, 1.0, n)
    mean = (1.0 - progress) * start_center + progress * end_center
    logits = -0.5 * ((centers - mean) / max(float(sigma), 1e-6)) ** 2
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    min_mass = float(np.clip(min_mass, 0.0, 1.0))
    return (1.0 - min_mass) * weights + min_mass / n


def interpolate_probs(start, end, progress: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    weights = (1.0 - progress) * np.asarray(start, dtype=np.float64) + progress * np.asarray(
        end, dtype=np.float64
    )
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def noise_bucket_name(alpha: float) -> str:
    return f"noise_{float(alpha):g}"


def inject_query_noise_tensor(
    image: torch.Tensor,
    alpha: float,
    mode: str = "lerp",
) -> torch.Tensor:
    """Corrupt a clean ``[0, 1]`` tensor into a synthetic degraded query."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return image
    if mode == "additive":
        noise = torch.randn_like(image)
        return torch.clamp(image + alpha * noise, 0.0, 1.0)
    if mode != "lerp":
        raise ValueError(f"noise_mode must be lerp or additive, got {mode!r}")
    noise = torch.rand_like(image)
    return (1.0 - alpha) * image + alpha * noise


class TarfileToSamplesWithMeta:
    """Like ``wds.tarfile_to_samples`` but copies shard-level fields onto samples."""

    def __init__(self, handler=None, rename_files=None):
        import webdataset as wds

        if handler is None:
            handler = wds.warn_and_continue
        self._inner = wds.tarfile_to_samples(handler=handler, rename_files=rename_files)

    def __call__(self, src):
        for shard in src:
            noise_alpha = 0.0
            if isinstance(shard, dict):
                noise_alpha = float(shard.get("noise_alpha", 0.0))
                shard = {"url": shard["url"]}
            for sample in self._inner(iter([shard])):
                if isinstance(sample, dict):
                    sample["__noise_alpha__"] = noise_alpha
                yield sample


def parse_float_list(text: Union[str, Sequence[float], None]) -> List[float]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    raw = str(text).strip()
    if not raw or raw.lower() in ("none", "off", "false"):
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def make_progress_holder(initial: float = 0.0):
    """Shared float for DataLoader workers (fork-safe)."""
    try:
        from torch.multiprocessing import Value

        return Value("d", float(initial))
    except Exception:  # noqa: BLE001
        return {"value": float(initial)}


def _read_progress(holder) -> float:
    if holder is None:
        return 0.0
    try:
        return float(np.clip(holder.value, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        try:
            return float(np.clip(holder["value"], 0.0, 1.0))
        except Exception:  # noqa: BLE001
            return 0.0


def _write_progress(holder, value: float):
    if holder is None:
        return
    v = float(np.clip(value, 0.0, 1.0))
    try:
        holder.value = v
    except Exception:  # noqa: BLE001
        try:
            holder["value"] = v
        except Exception:  # noqa: BLE001
            pass


class CurriculumShardList(IterableDataset):
    """Infinite shard URL source that samples tars by dsim-bucket curriculum.

    Yields ``{"url": ..., "noise_alpha": ...}`` dicts like ``ResampledShards``,
    so it plugs into the existing WDS pipeline ahead of ``split_by_node``.
    """

    def __init__(
        self,
        urls: Sequence[str],
        progress=None,
        enable_curriculum: bool = True,
        curriculum_sigma: float = 0.16,
        curriculum_min_mass: float = 0.10,
        curriculum_start_center: float = 0.0,
        curriculum_end_center: float = 0.52,
        curriculum_start_probs: Union[str, Sequence[float], None] = None,
        curriculum_end_probs: Union[str, Sequence[float], None] = None,
        noise_levels: Union[str, Sequence[float], None] = None,
        seed: int = 0,
    ):
        if not urls:
            raise ValueError("CurriculumShardList requires at least one shard URL")
        super().__init__()
        dsim_bucket_shards = group_shards_by_bucket(urls)
        dsim_names = list(dsim_bucket_shards.keys())
        self.flat_urls = [u for bucket in dsim_bucket_shards.values() for u in bucket]
        self.noise_levels = [float(a) for a in parse_float_list(noise_levels)]
        for alpha in self.noise_levels:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"noise level must be in [0, 1], got {alpha}")
        self.noise_alphas = {noise_bucket_name(alpha): alpha for alpha in self.noise_levels}

        self.bucket_shards = dict(dsim_bucket_shards)
        for name in self.noise_alphas:
            self.bucket_shards[name] = list(self.flat_urls)
        self.bucket_names = dsim_names + list(self.noise_alphas.keys())

        self.progress = progress if progress is not None else make_progress_holder(0.0)
        self.curriculum_sigma = float(curriculum_sigma)
        self.curriculum_min_mass = float(curriculum_min_mass)
        self.curriculum_start_center = float(curriculum_start_center)
        self.curriculum_end_center = float(curriculum_end_center)
        self.seed = int(seed)
        n = len(self.bucket_names)
        self.enable_curriculum = bool(enable_curriculum) and n > 1

        start = parse_float_list(curriculum_start_probs)
        end = parse_float_list(curriculum_end_probs)
        if start or end:
            if len(start) != n or len(end) != n:
                raise ValueError(
                    f"curriculum_start/end_probs must each have {n} values "
                    f"(dsim buckets + noise levels={self.bucket_names}), got "
                    f"{len(start)} and {len(end)}"
                )
            self._start_probs = np.asarray(start, dtype=np.float64)
            self._start_probs = self._start_probs / self._start_probs.sum()
            self._end_probs = np.asarray(end, dtype=np.float64)
            self._end_probs = self._end_probs / self._end_probs.sum()
        else:
            self._start_probs = None
            self._end_probs = None

        for i, name in enumerate(self.bucket_names):
            if name in self.noise_alphas:
                log.info(
                    f"  curriculum bucket [{i}] {name:20s}  synthetic  "
                    f"alpha={self.noise_alphas[name]:g}  source={len(self.flat_urls)} tars"
                )
            else:
                log.info(f"  curriculum bucket [{i}] {name:20s}  {len(self.bucket_shards[name])} shards")
        if self.enable_curriculum:
            p0 = self.bucket_probs(0.0)
            p1 = self.bucket_probs(1.0)
            log.info(
                "  curriculum start: "
                + ", ".join(f"{n}={p:.3f}" for n, p in zip(self.bucket_names, p0))
            )
            log.info(
                "  curriculum end:   "
                + ", ".join(f"{n}={p:.3f}" for n, p in zip(self.bucket_names, p1))
            )
        else:
            log.info("  curriculum off (single bucket or disabled)")

    def set_progress(self, progress: float):
        _write_progress(self.progress, progress)

    def bucket_probs(self, progress: Optional[float] = None) -> np.ndarray:
        n = len(self.bucket_names)
        if n == 0:
            return np.zeros((0,), dtype=np.float64)
        if progress is None:
            progress = _read_progress(self.progress)
        if self._start_probs is not None:
            return interpolate_probs(self._start_probs, self._end_probs, progress)
        return gaussian_curriculum_probs(
            n,
            progress,
            start_center=self.curriculum_start_center,
            end_center=self.curriculum_end_center,
            sigma=self.curriculum_sigma,
            min_mass=self.curriculum_min_mass,
        )

    def _yield_shard(self, rng: random.Random, bucket_name: str) -> Dict[str, Union[str, float]]:
        noise_alpha = float(self.noise_alphas.get(bucket_name, 0.0))
        return {
            "url": rng.choice(self.bucket_shards[bucket_name]),
            "noise_alpha": noise_alpha,
        }

    def __iter__(self):
        # Independent RNG per worker process (re-seeded on fork via pid).
        rng = random.Random(self.seed + os.getpid())
        while True:
            if not self.enable_curriculum:
                if self.noise_levels:
                    bucket_name = rng.choice(self.bucket_names)
                    yield self._yield_shard(rng, bucket_name)
                else:
                    yield {"url": rng.choice(self.flat_urls), "noise_alpha": 0.0}
                continue
            probs = self.bucket_probs()
            bi = int(np.random.choice(len(self.bucket_names), p=probs))
            name = self.bucket_names[bi]
            yield self._yield_shard(rng, name)
