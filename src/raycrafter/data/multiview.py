"""Group single-view WebDataset samples into one multi-view sample.

Each incoming sample's ``__key__`` must be ``"{f1}__{f2}__{f3}__{f4}__{frame}"``
(5+ fields split by ``__``). Everything but the last field is the sequence id;
the last field's trailing ``_<int>`` is the frame index.

For every incoming sample exactly one grouped sample is emitted (padding with
itself when no neighbors are cached yet), so the yield count never depends on
buffering — important for DDP stability.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MultiViewConfig:
    num_views: int = 1      # degraded input views per sample (current frame + neighbors)
    num_refs: int = 1       # clean reference views per sample
    window: int = 15        # neighbors are within ±window frames of the current one
    seed: int = 0
    input_key: str = "png"
    target_key: str = "gt.png"
    ref_key: str = "ref.png"
    camera_key: str = "pose.json"


class MultiViewStage:
    """Stateful mapper: emits ``img{i}.png, gt{i}.png, camera{i}.json`` (i=1..V) and
    ``ref{r}.png, ref_camera{r}.json`` (r=1..R) for each incoming frame."""

    def __init__(self, cfg: MultiViewConfig):
        self.cfg = cfg
        # seq_id -> {frame_idx: (inp, tgt, ref, camera)}; only the 2 most recent sequences.
        self._cache: Dict[str, Dict[int, Tuple]] = {}
        # seq_id -> {ref_hash: (None, None, ref, camera)}. Distinct refs seen in the
        # sequence, not limited by the pairing window and with a larger LRU: the
        # curriculum interleaves many sequences, which evicts the frame cache often.
        self._ref_pool: Dict[str, Dict[bytes, Tuple]] = {}
        self._ref_pool_max_seqs = 64
        self._ref_pool_max_refs = 32

    @staticmethod
    def _parse_key(sample_key: str) -> Tuple[str, int]:
        parts = sample_key.split("__")
        if len(parts) < 5:
            raise ValueError(f"Expected __key__ with 5+ fields split by '__', got: {sample_key}")
        return "__".join(parts[:-1]), int(parts[-1].split("_")[-1])

    def _rng_for(self, sample_key: str) -> random.Random:
        h = hashlib.blake2b((str(self.cfg.seed) + "|" + sample_key).encode("utf-8"), digest_size=8).digest()
        return random.Random(int.from_bytes(h, "little"))

    def __call__(self, sample: Dict) -> Dict:
        cfg = self.cfg
        key = sample.get("__key__")
        if key is None:
            raise ValueError("WebDataset sample missing '__key__'")
        seq_id, idx = self._parse_key(key)
        if cfg.input_key not in sample or cfg.target_key not in sample:
            raise KeyError(f"Required keys not found in sample keys: {list(sample.keys())}")

        ref_bytes = sample[cfg.ref_key]
        camera_bytes = sample.get(cfg.camera_key)

        # 1. Frame cache (LRU over sequences).
        cache = self._cache.pop(seq_id, {})
        self._cache[seq_id] = cache
        if len(self._cache) > 2:
            del self._cache[next(iter(self._cache))]
        cache[idx] = (sample[cfg.input_key], sample[cfg.target_key], ref_bytes, camera_bytes)

        # 2. Ref pool.
        ref_hash = hashlib.blake2b(ref_bytes, digest_size=16).digest()
        ref_pool = self._ref_pool.pop(seq_id, {})
        self._ref_pool[seq_id] = ref_pool
        if len(self._ref_pool) > self._ref_pool_max_seqs:
            del self._ref_pool[next(iter(self._ref_pool))]
        if ref_hash not in ref_pool and len(ref_pool) < self._ref_pool_max_refs:
            ref_pool[ref_hash] = (None, None, ref_bytes, camera_bytes)

        # 3. Prune frames outside the window.
        if cfg.window > 0 and len(cache) > 8:
            lo, hi = idx - cfg.window, idx + cfg.window
            for k in list(cache.keys()):
                if k < lo or k > hi:
                    cache.pop(k, None)

        # 4. Neighbors: without replacement if enough, else reuse, else self.
        rng = self._rng_for(key)
        candidates = [j for j in cache if j != idx and abs(j - idx) <= cfg.window]
        neighbors = []
        if cfg.num_views > 1:
            k_needed = cfg.num_views - 1
            if len(candidates) >= k_needed:
                neighbors = rng.sample(candidates, k=k_needed)
            elif candidates:
                neighbors = list(candidates) + rng.choices(candidates, k=k_needed - len(candidates))
            else:
                neighbors = [idx] * k_needed
        selected = [idx] + neighbors

        # 5. Refs: the frame's own ref plus distinct refs from the sequence
        # (repeats only when fewer exist), in random slot order — pinning the own
        # (usually nearest) ref to slot 1 let the model copy it and ignore cameras.
        ref_providers = []
        if cfg.num_refs > 0:
            others = [e for h, e in ref_pool.items() if h != ref_hash]
            ref_providers = [cache[idx]] + rng.sample(others, k=min(cfg.num_refs - 1, len(others)))
            distinct = list(ref_providers)
            while len(ref_providers) < cfg.num_refs:
                ref_providers.append(rng.choice(distinct))
            rng.shuffle(ref_providers)

        out: Dict = {"__key__": key, "__url__": sample.get("__url__")}
        if "noise_alpha" in sample:
            out["__noise_alpha__"] = sample["noise_alpha"]
        elif "__noise_alpha__" in sample:
            out["__noise_alpha__"] = sample["__noise_alpha__"]

        for r, (_, _, ref, cam) in enumerate(ref_providers, start=1):
            out[f"ref{r}.png"] = ref
            if cam is not None:
                out[f"ref_camera{r}.json"] = cam
        for i, s_idx in enumerate(selected, start=1):
            inp, tgt, _, cam = cache[s_idx]
            out[f"img{i}.png"] = inp
            out[f"gt{i}.png"] = tgt
            if cam is not None:
                out[f"camera{i}.json"] = cam
        return out
