"""Group single-view WebDataset samples into multi-view samples, one whole sequence at a time.

Each incoming sample's ``__key__`` must be ``"{f1}__{f2}__{f3}__{f4}__{frame}"``
(5+ fields split by ``__``). Everything but the last field is the sequence id;
the last field's trailing ``_<int>`` is the frame index. A sequence's samples
must be contiguous in the tar.

A whole sequence is read before any of its samples is built, so every target
gets refs (and neighbor views) from the entire sequence, not only from the
frames streamed before it.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class MultiViewConfig:
    num_views: int = 1      # target views per sample (the chosen frame + neighbors)
    num_refs: int = 1       # clean reference views per sample
    window: int = 15        # neighbors are within ±window frame indices
    seed: int = 0
    input_key: str = "png"
    target_key: str = "gt.png"
    ref_key: str = "ref.png"
    camera_key: str = "pose.json"


def parse_key(sample_key: str) -> Tuple[str, int]:
    """``__key__`` → (sequence id, frame index)."""
    parts = sample_key.split("__")
    if len(parts) < 5:
        raise ValueError(f"Expected __key__ with 5+ fields split by '__', got: {sample_key}")
    return "__".join(parts[:-1]), int(parts[-1].split("_")[-1])


def key_rng(seed: int, sample_key: str) -> random.Random:
    """Deterministic RNG per sample key (validation / sampling)."""
    h = hashlib.blake2b((str(seed) + "|" + sample_key).encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "little"))


@dataclass
class Frame:
    key: str
    index: int
    inp: bytes
    tgt: bytes
    camera: Optional[bytes]
    ref_hash: bytes


@dataclass
class Sequence:
    """All frames of one sequence, plus its distinct refs: ``ref_hash -> (ref bytes, camera bytes)``.

    Each ref is stored once; the camera is the pose.json of a frame that uses it
    (its ``ref`` entry is that ref's camera).
    """
    seq_id: str
    url: Optional[str]
    frames: List[Frame] = field(default_factory=list)
    refs: Dict[bytes, Tuple[bytes, Optional[bytes]]] = field(default_factory=dict)


def iter_sequences(samples: Iterable[Dict], cfg: MultiViewConfig) -> Iterator[Sequence]:
    """Group a stream of raw tar samples into whole sequences (runs of equal sequence id)."""
    seq: Optional[Sequence] = None
    for sample in samples:
        key = sample.get("__key__")
        try:
            if key is None:
                raise ValueError("WebDataset sample missing '__key__'")
            missing = [k for k in (cfg.input_key, cfg.target_key, cfg.ref_key) if k not in sample]
            if missing:
                raise KeyError(f"missing {missing}; sample keys: {list(sample.keys())}")
            seq_id, idx = parse_key(key)
        except (KeyError, ValueError) as exc:
            # Loud, so a format problem does not silently filter every sample.
            print(f"[raycrafter.data] skip sample {key!r}: {exc}", flush=True)
            continue
        if seq is None or seq_id != seq.seq_id:
            if seq is not None:
                yield seq
            seq = Sequence(seq_id, sample.get("__url__"))
        ref = sample[cfg.ref_key]
        camera = sample.get(cfg.camera_key)
        ref_hash = hashlib.blake2b(ref, digest_size=16).digest()
        seq.refs.setdefault(ref_hash, (ref, camera))
        seq.frames.append(Frame(key, idx, sample[cfg.input_key], sample[cfg.target_key], camera, ref_hash))
    if seq is not None:
        yield seq


def build_sample(seq: Sequence, i: int, rng: random.Random, cfg: MultiViewConfig) -> Dict:
    """Multi-view sample whose first target view is ``seq.frames[i]``.

    Emits ``img{v}.png, gt{v}.png, camera{v}.json`` (v=1..V: frame ``i``, then
    neighbors within ±window of it) and ``ref{r}.png, ref_camera{r}.json``
    (r=1..R): the frame's own ref plus distinct refs of the whole sequence,
    repeated only when the sequence has fewer than R.
    """
    frame = seq.frames[i]

    selected = [frame]
    if cfg.num_views > 1:
        k = cfg.num_views - 1
        candidates = [f for j, f in enumerate(seq.frames)
                      if j != i and abs(f.index - frame.index) <= cfg.window]
        if len(candidates) >= k:
            selected += rng.sample(candidates, k=k)
        elif candidates:
            selected += candidates + rng.choices(candidates, k=k - len(candidates))
        else:
            selected += [frame] * k

    # Random slot order: pinning the own (usually nearest) ref to slot 1 let the
    # model copy it and ignore cameras.
    refs: List[bytes] = []
    if cfg.num_refs > 0:
        others = [h for h in seq.refs if h != frame.ref_hash]
        refs = [frame.ref_hash] + rng.sample(others, k=min(cfg.num_refs - 1, len(others)))
        distinct = list(refs)
        while len(refs) < cfg.num_refs:
            refs.append(rng.choice(distinct))
        rng.shuffle(refs)

    out: Dict = {"__key__": frame.key, "__url__": seq.url}
    for r, ref_hash in enumerate(refs, start=1):
        ref, cam = seq.refs[ref_hash]
        out[f"ref{r}.png"] = ref
        if cam is not None:
            out[f"ref_camera{r}.json"] = cam
    for v, f in enumerate(selected, start=1):
        out[f"img{v}.png"] = f.inp
        out[f"gt{v}.png"] = f.tgt
        if f.camera is not None:
            out[f"camera{v}.json"] = f.camera
    return out


class OrderedMultiView:
    """Pipeline stage for validation / sampling: every frame of every sequence once,
    in tar order, with refs chosen deterministically per sample key."""

    def __init__(self, cfg: MultiViewConfig):
        self.cfg = cfg

    def __call__(self, samples: Iterable[Dict]) -> Iterator[Dict]:
        for seq in iter_sequences(samples, self.cfg):
            for i, frame in enumerate(seq.frames):
                yield build_sample(seq, i, key_rng(self.cfg.seed, frame.key), self.cfg)
