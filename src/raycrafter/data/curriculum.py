"""DreamSim-bucket curriculum, drawn per sample.

Mirrors FrameCrafter's ``cleanup_dataset`` schedule: group ``train_shards`` by
the ``dsim_*`` directory in the path, and move the bucket probabilities from
easy (``dsim_lt_0.25``) to hard (``dsim_ge_0.4``) as training progress goes
0 -> 1. Optional ``noise_levels`` add synthetic buckets (``noise_0.9``, …) whose
inputs are replaced by noise of that strength.

The bucket of *every sample* is drawn from the probabilities at the current
step (:class:`CurriculumMixer`). Drawing per tar does not work for large tars:
one draw would cover thousands of steps, and a shard-level shuffle buffer fills
with draws made at progress 0.
"""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence, Union

import numpy as np
import torch
import webdataset as wds
from webdataset.utils import pytorch_worker_info

from .multiview import MultiViewConfig, Sequence as FrameSequence, build_sample, iter_sequences

try:
    from webdataset.pytorch import IterableDataset
except ModuleNotFoundError:  # pragma: no cover
    from torch.utils.data import IterableDataset

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


class BucketSchedule:
    """Bucket names and their sampling probabilities as a function of progress.

    Buckets: the real ``dsim_*`` buckets in curriculum order, then one
    ``noise_<a>`` bucket per noise level.
    """

    def __init__(
        self,
        urls: Sequence[str],
        enable_curriculum: bool = True,
        curriculum_sigma: float = 0.16,
        curriculum_min_mass: float = 0.10,
        curriculum_start_center: float = 0.0,
        curriculum_end_center: float = 0.52,
        curriculum_start_probs: Union[str, Sequence[float], None] = None,
        curriculum_end_probs: Union[str, Sequence[float], None] = None,
        noise_levels: Union[str, Sequence[float], None] = None,
    ):
        if not urls:
            raise ValueError("BucketSchedule requires at least one shard URL")
        self.real_shards = group_shards_by_bucket(urls)
        self.noise_alphas: Dict[str, float] = {}
        for alpha in parse_float_list(noise_levels):
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"noise level must be in [0, 1], got {alpha}")
            self.noise_alphas[noise_bucket_name(alpha)] = float(alpha)
        self.bucket_names = list(self.real_shards) + list(self.noise_alphas)

        n = len(self.bucket_names)
        self.enable_curriculum = bool(enable_curriculum) and n > 1
        self.curriculum_sigma = float(curriculum_sigma)
        self.curriculum_min_mass = float(curriculum_min_mass)
        self.curriculum_start_center = float(curriculum_start_center)
        self.curriculum_end_center = float(curriculum_end_center)

        start = parse_float_list(curriculum_start_probs)
        end = parse_float_list(curriculum_end_probs)
        self._start_probs = self._end_probs = None
        if start or end:
            if len(start) != n or len(end) != n:
                raise ValueError(
                    f"curriculum_start/end_probs must each have {n} values "
                    f"(dsim buckets + noise levels={self.bucket_names}), got "
                    f"{len(start)} and {len(end)}"
                )
            self._start_probs = np.asarray(start, dtype=np.float64) / np.sum(start)
            self._end_probs = np.asarray(end, dtype=np.float64) / np.sum(end)

    def probs(self, progress: float) -> np.ndarray:
        n = len(self.bucket_names)
        if not self.enable_curriculum:
            if self.noise_alphas:
                return np.full(n, 1.0 / n)
            # No curriculum and no noise: like a plain shard list (∝ tar count).
            counts = np.asarray([len(u) for u in self.real_shards.values()], dtype=np.float64)
            return counts / counts.sum()
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

    def describe(self) -> str:
        p0, p1 = self.probs(0.0), self.probs(1.0)
        mode = "per-sample curriculum" if self.enable_curriculum else "fixed bucket mix"
        lines = [f"data: {sum(len(u) for u in self.real_shards.values())} tars, {mode}"]
        for i, name in enumerate(self.bucket_names):
            what = (f"synthetic, alpha={self.noise_alphas[name]:g}" if name in self.noise_alphas
                    else f"{len(self.real_shards[name])} tars")
            lines.append(f"  [{i}] {name:16s} {what:22s} start {p0[i]:.3f}  end {p1[i]:.3f}")
        return "\n".join(lines)


def _tar_samples(url: str) -> Iterator[Dict]:
    return wds.tarfile_to_samples(handler=wds.warn_and_continue)(iter([{"url": url}]))


class _BucketStream:
    """Endless multi-view samples from one bucket's tars.

    Reads tars front to back (in random order, cycled) and keeps
    ``open_sequences`` whole sequences in memory. Each sample's target is a
    random not-yet-used frame among them, so every frame of a sequence is a
    target once, in random order, interleaved with other sequences.
    """

    def __init__(self, urls: List[str], rng: random.Random, cfg: MultiViewConfig, open_sequences: int):
        self.urls = list(urls)
        self.rng = rng
        self.cfg = cfg
        self.open_sequences = max(1, int(open_sequences))
        self._queue: List[str] = []
        self._sequences: Iterator[FrameSequence] = iter(())
        self._open: List[tuple] = []  # (sequence, frame positions not yet emitted)

    def _next_sequence(self) -> FrameSequence:
        tars_opened = 0
        while True:
            seq = next(self._sequences, None)
            if seq is not None:
                return seq
            if tars_opened > len(self.urls):
                raise RuntimeError(f"no usable samples in {len(self.urls)} tars, e.g. {self.urls[0]}")
            if not self._queue:
                self._queue = self.rng.sample(self.urls, len(self.urls))
            self._sequences = iter_sequences(_tar_samples(self._queue.pop()), self.cfg)
            tars_opened += 1

    def next_sample(self) -> Dict:
        while len(self._open) < self.open_sequences:
            seq = self._next_sequence()
            self._open.append((seq, list(range(len(seq.frames)))))
        # A uniformly random remaining frame across the open sequences.
        j = self.rng.choices(range(len(self._open)), weights=[len(r) for _, r in self._open])[0]
        seq, remaining = self._open[j]
        i = remaining.pop(self.rng.randrange(len(remaining)))
        if not remaining:
            self._open.pop(j)
        return build_sample(seq, i, self.rng, self.cfg)


class CurriculumMixer(IterableDataset):
    """Endless training samples whose bucket is drawn per sample from the live schedule.

    Each DataLoader worker keeps one :class:`_BucketStream` per real bucket,
    over its share of that bucket's tars (tars are split across all GPUs ×
    workers when there are enough of them). For each sample a bucket is drawn
    from ``schedule.probs(progress)``; a synthetic ``noise_<a>`` bucket takes the
    sample from a real stream picked in proportion to tar counts and sets
    ``__noise_alpha__ = a``. ``progress`` is the shared value train.py updates
    every step, so the mix follows the schedule up to the DataLoader prefetch.
    """

    def __init__(
        self,
        schedule: BucketSchedule,
        cfg: MultiViewConfig,
        progress=None,
        open_sequences: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.schedule = schedule
        self.cfg = cfg
        self.progress = progress
        self.open_sequences = open_sequences
        self.seed = seed

    def __iter__(self) -> Iterator[Dict]:
        rank, world_size, worker, num_workers = pytorch_worker_info()
        stream_id, num_streams = rank * num_workers + worker, world_size * num_workers
        # pid: a restarted (resumed) run does not replay the same stream.
        rng = random.Random(f"{self.seed}|{stream_id}|{os.getpid()}")
        streams = {
            name: _BucketStream(urls[stream_id::num_streams] if len(urls) >= num_streams else urls,
                                rng, self.cfg, self.open_sequences)
            for name, urls in self.schedule.real_shards.items()
        }
        real_names = list(streams)
        tar_counts = [len(urls) for urls in self.schedule.real_shards.values()]
        names = self.schedule.bucket_names
        while True:
            name = rng.choices(names, weights=self.schedule.probs(_read_progress(self.progress)))[0]
            alpha = self.schedule.noise_alphas.get(name, 0.0)
            if name in self.schedule.noise_alphas:
                name = rng.choices(real_names, weights=tar_counts)[0]
            sample = streams[name].next_sample()
            sample["__noise_alpha__"] = alpha
            yield sample
