#!/usr/bin/env python3
"""Evaluate one trained run on HM3D, Replica, JHU, and/or DL3DV.

The run directory is the training output folder: it must contain
``config.yaml`` and ``transformer_step*.safetensors``. ``--step -1`` (the
default) uses the latest checkpoint.

``--tasks`` is ``refine``, ``nvs``, or ``refine,nvs``. Refinement feeds the
stored degraded query (panorama render, or the DL3DV point splat). NVS feeds
a black query image, which is what training does for ``nvs_ratio``.

Panorama sets are 2 references + ``num_views`` targets (3 in the mix configs),
at ``panorama_image_size``. DL3DV uses ``image_size``. Poses are normalized as
in training: re-centered on the last camera, translations divided by the
scene's near depth. Panorama scenes missing from the near-depth map (the HM3D
and Replica val splits are not in ``near_depth/framecrafter_5input.json``,
which only covers the training scenes) fall back to the loader's mean-baseline
gauge and are counted in the summary.

Default test sets:

  HM3D     /cis/net/io72a/data/xliu253/hm3d_val_data
           nvs:    val_2ref3_nvs.jsonl
           refine: val_2ref3_cleanup_drop050.jsonl
  Replica  /cis/net/io72a/data/xliu253/replica_val_data
           same manifest names as HM3D
  JHU      /cis/net/io80/data/xliu253/framecrafter/5input_model/evaluation/jhu_holdout10_500_2ref3
           val_2ref3_nvs.jsonl (500 sets, 50 per held-out scene).
           Refinement loads ``original_noisy_render`` from the 5-input tree.
  DL3DV    /cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test

``--dl3dv_sample_rate R`` keeps every R-th sample of each scene, starting at
the first. A scene with 20 samples and R=5 contributes 4 samples.

    python eval_ckpt.py --run_dir outputs/flux3_mix_rayrope_dlr \\
        --tasks refine,nvs --datasets HM3D,JHU,DL3DV --dl3dv_sample_rate 5

Writes ``<out_dir>/<dataset>/{refine,nvs}/`` with panels, samples.jsonl, and
summary.json. The default out dir is
``outputs/eval_ckpt/<run>/<checkpoint>/n<steps>``.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent / "src"))

from raycrafter.config import Config  # noqa: E402
from raycrafter.data.dl3dv_points import _c2w, _scale_k, load_near_depth, scale_translation  # noqa: E402
from raycrafter.model import RayCrafterModel  # noqa: E402

# Equirect convention used by training. Pure-rotation sets (mean baseline below
# this, in raw pose units) get zero translation.
CAMERA_MODEL = "equirect"
MIN_BASELINE = 1e-3
_IMAGE = re.compile(r"^(?P<dir>.+)/(?P<frame>\d+)\.png$")
DREAMSIM_CACHE = os.environ.get("DREAMSIM_CACHE", str(Path(__file__).parent / "weights" / "dreamsim"))
IMAGE_METRICS = ("psnr", "ssim", "lpips", "dreamsim")

HM3D_ROOT = "/cis/net/io72a/data/xliu253/hm3d_val_data"
REPLICA_ROOT = "/cis/net/io72a/data/xliu253/replica_val_data"
JHU_ROOT = (
    "/cis/net/io80/data/xliu253/framecrafter/5input_model/evaluation/jhu_holdout10_500_2ref3"
)
JHU_RENDER_ROOT = "/cis/net/io80/data/xliu253/framecrafter/5input_model"
DL3DV_ROOT = "/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/test"
PANO_NEAR = "near_depth/framecrafter_5input.json"
DL3DV_NEAR = "near_depth/dl3dv_points.json"

# nvs manifest is the black-query test list. refine manifest, when present, is
# the stored-render list (one fixed dropout). JHU has no separate refine file:
# its nvs rows point at the original render.
_PANO = {
    "HM3D": {
        "root": HM3D_ROOT,
        "nvs_manifest": "val_2ref3_nvs.jsonl",
        "refine_manifest": "val_2ref3_cleanup_drop050.jsonl",
        "render_roots": (),
    },
    "Replica": {
        "root": REPLICA_ROOT,
        "nvs_manifest": "val_2ref3_nvs.jsonl",
        "refine_manifest": "val_2ref3_cleanup_drop050.jsonl",
        "render_roots": (),
    },
    "JHU": {
        "root": JHU_ROOT,
        "nvs_manifest": "val_2ref3_nvs.jsonl",
        "refine_manifest": None,
        "render_roots": (JHU_RENDER_ROOT,),
    },
}


def _split_frame(path: str) -> tuple[str, int]:
    """``"HM3D/images/<scene>/000074.png"`` → (directory, 74)."""
    match = _IMAGE.match(path)
    if match is None:
        raise ValueError(f"unexpected image path {path!r}")
    return match["dir"], int(match["frame"])


def _load(path: Path, size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        src_wh = image.size
        if src_wh != (size[1], size[0]):
            image = TF.resize(
                image, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
        return TF.to_tensor(image) * 2 - 1, src_wh


def _uint8(image: torch.Tensor) -> np.ndarray:
    chw = image.detach().float().clamp(-1, 1).cpu()
    return ((chw + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def _match(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    height, width = pred.shape[-2:]
    top, left = (gt.shape[-2] - height) // 2, (gt.shape[-1] - width) // 2
    return gt[..., top:top + height, left:left + width]


def _start(src: torch.Tensor, start: str) -> torch.Tensor:
    """Stored query, or black (-1) for NVS."""
    return torch.full_like(src, -1.0) if start == "black" else src


def _panel(src, pred, gt, refs) -> np.ndarray:
    row = np.concatenate([_uint8(src), _uint8(pred), _uint8(gt)], axis=1)
    strip = np.concatenate([_uint8(ref) for ref in refs], axis=1)
    strip = np.array(Image.fromarray(strip).resize((row.shape[1], row.shape[1] * strip.shape[0] // strip.shape[1])))
    return np.concatenate([row, strip], axis=0)


def _saved_steps(run_dir: Path) -> list[int]:
    steps = [int(path.stem.removeprefix("transformer_step")) for path in run_dir.glob("transformer_step*.safetensors")]
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    return steps


def _pose(row: dict, src_wh: tuple[int, int], size: tuple[int, int], near_depth: float | None):
    c2w = _c2w(row["w2c"])
    if near_depth is not None:
        c2w = scale_translation(c2w, near_depth)
    return c2w, _scale_k(row["camera"], src_wh, size)


def _scan(root: Path):
    """DL3DV metadata only. Images stay on disk until a scene is selected."""
    samples_dir = root / "priors" / "samples"
    scenes: dict[str, list] = {}
    for sample_dir in sorted(path for path in samples_dir.iterdir() if (path / "metadata.json").is_file()):
        meta = json.loads((sample_dir / "metadata.json").read_text())
        refs, targets = [], []
        for index, row in enumerate(meta["references"]):
            refs.append({"frame": int(row["frame_index"]), "path": sample_dir / f"ref_{index}.png", "row": row})
        for index, row in enumerate(sorted(meta["targets"], key=lambda item: item["target_number"])):
            targets.append({
                "frame": int(row["frame_index"]),
                "path": sample_dir / f"gt_{index}.png",
                "prior": sample_dir / f"prior_{index}.png",
                "row": row,
            })
        scenes.setdefault(meta["scene_name"], []).append({"id": sample_dir.name, "refs": refs, "targets": targets})
    return scenes


def _materialize(samples: list, size: tuple[int, int], near_depth: float | None = None):
    """Load one scene's DL3DV samples. ``near_depth`` rescales the translations."""
    loaded = []
    for sample in samples:
        refs, targets = [], []
        for item in sample["refs"]:
            image, src_wh = _load(item["path"], size)
            c2w, k = _pose(item["row"], src_wh, size, near_depth)
            refs.append({"frame": item["frame"], "image": image, "c2w": c2w, "k": k})
        for item in sample["targets"]:
            gt, src_wh = _load(item["path"], size)
            prior, _ = _load(item["prior"], size)
            c2w, k = _pose(item["row"], src_wh, size, near_depth)
            targets.append({"frame": item["frame"], "image": gt, "prior": prior, "c2w": c2w, "k": k})
        loaded.append({"id": sample["id"], "refs": refs, "targets": targets})
    return loaded


def _allow_dreamsim_lora() -> None:
    """DreamSim's adapter config has a field this peft build does not accept."""
    from peft import LoraConfig

    init = LoraConfig.__init__
    if getattr(init, "_raycrafter_wrapped", False):
        return

    def wrapped(self, *args, **kwargs):
        kwargs.pop("layer_replication", None)
        return init(self, *args, **kwargs)

    wrapped._raycrafter_wrapped = True
    LoraConfig.__init__ = wrapped


class _Metrics:
    def __init__(self, device: torch.device):
        import lpips
        from dreamsim import dreamsim
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        _allow_dreamsim_lora()
        self.dreamsim, _ = dreamsim(pretrained=True, device=device, cache_dir=DREAMSIM_CACHE)
        self.dreamsim.eval()

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        pred, gt = pred.float(), gt.float()
        unit_pred = (pred.clamp(-1, 1) + 1) * 0.5
        unit_gt = (gt.clamp(-1, 1) + 1) * 0.5
        self.psnr.reset()
        self.ssim.reset()
        small_pred = torch.nn.functional.interpolate(
            unit_pred[None], (224, 224), mode="bicubic", antialias=True, align_corners=False,
        ).clamp(0, 1)
        small_gt = torch.nn.functional.interpolate(
            unit_gt[None], (224, 224), mode="bicubic", antialias=True, align_corners=False,
        ).clamp(0, 1)
        return {
            "psnr": float(self.psnr(unit_pred[None], unit_gt[None])),
            "ssim": float(self.ssim(unit_pred[None], unit_gt[None])),
            "lpips": float(self.lpips(pred[None], gt[None]).flatten()[0]),
            "dreamsim": float(self.dreamsim(small_pred, small_gt).flatten()[0]),
        }


def _mean(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in IMAGE_METRICS:
        out[key] = float(np.mean([row[key] for row in rows])) if rows else None
    return out


def _fid(groups: dict[str, dict], device: torch.device) -> tuple[float, dict[str, float]]:
    """Inception FID of pred against gt. One number per scene, plus every scene together."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def score(real_hwc: list, fake_hwc: list) -> float:
        metric.reset()
        for real_flag, images in ((True, real_hwc), (False, fake_hwc)):
            batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
            for start in range(0, len(batch), 32):
                metric.update(batch[start:start + 32].to(device), real=real_flag)
        return float(metric.compute())

    per_scene = {scene: score(pair["gt"], pair["pred"]) for scene, pair in groups.items()}
    overall = score(
        [img for pair in groups.values() for img in pair["gt"]],
        [img for pair in groups.values() for img in pair["pred"]],
    )
    del metric
    return overall, per_scene


def _split_arg(value) -> list[str]:
    """Fire turns an unquoted ``a,b`` into a tuple. Accept both."""
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [str(part).strip() for part in raw if str(part).strip()]


def _tasks(value) -> tuple[str, ...]:
    names = _split_arg(value)
    if len(names) == 1 and names[0].lower() == "both":
        names = ["refine", "nvs"]
    names = [name.lower() for name in names]
    unknown = [name for name in names if name not in ("refine", "nvs")]
    if unknown or not names:
        raise ValueError(f"tasks must be refine, nvs, or both, got {value!r}")
    # Stable order, duplicates dropped.
    return tuple(name for name in ("refine", "nvs") if name in names)


def _datasets(value) -> list[str]:
    wanted = []
    for name in _split_arg(value):
        key = {"hm3d": "HM3D", "replica": "Replica", "jhu": "JHU", "dl3dv": "DL3DV"}.get(name.lower())
        if key is None:
            raise ValueError(f"unknown dataset {name!r}; choose from HM3D, Replica, JHU, DL3DV")
        if key not in wanted:
            wanted.append(key)
    if not wanted:
        raise ValueError("datasets is empty")
    return wanted


def _checkpoint(run_dir: Path, step: int) -> Path:
    if not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"{run_dir} has no config.yaml")
    saved = _saved_steps(run_dir)
    step = max(saved) if int(step) < 0 else int(step)
    path = run_dir / f"transformer_step{step}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"{run_dir.name} has no checkpoint at step {step} (saved: {saved})"
        )
    return path


def _predict(model, src, ref, c2w, k, tgt, steps: int, task: str, seed: int, camera_model: str):
    """``task`` is ``refine`` (stored query) or ``nvs`` (black query image)."""
    torch.manual_seed(seed)
    device = src.device
    alpha = torch.zeros(src.shape[0], device=device)
    if task == "nvs":
        src = torch.full_like(src, -1.0)
    return model.sample(
        src, ref, num_steps=steps, camera_c2w=c2w, camera_fxfycxcy=k,
        noise_alpha=alpha, tgt_imgs=tgt, camera_model=camera_model,
    )


def _index_sets(root: Path, manifest: str, num_views: int) -> list[list[dict]]:
    path = Path(manifest)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    groups: dict[str, dict[int, dict]] = {}
    try:
        lines = path.read_text().splitlines()
    except PermissionError as exc:
        raise PermissionError(
            f"cannot read {path} (mode 600, owner is not this user). "
            "Ask xliu253 to grant read access on the val tree."
        ) from exc
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        groups.setdefault(row["source_sample"], {})[int(row["target_number"])] = row
    sets, incomplete = [], 0
    for key in sorted(groups):
        rows = groups[key]
        if sorted(rows) != list(range(num_views)):
            incomplete += 1
            continue
        ordered = [rows[i] for i in range(num_views)]
        if any(row["refs"] != ordered[0]["refs"] for row in ordered):
            raise ValueError(f"{key}: queries of a set must share two references")
        sets.append(ordered)
    if not sets:
        raise ValueError(f"no complete {num_views}-query sets in {path}")
    if incomplete:
        print(f"{path.name}: dropped {incomplete} sets that are not {num_views} queries", flush=True)
    return sets


class _PanoramaPoses:
    """OpenCV w2c from ``poses/manifest.json``, normalized like ``PanoramaSets``."""

    def __init__(self, root: Path, near: dict[str, float] | None):
        manifest = root / "poses" / "manifest.json"
        try:
            poses = json.loads(manifest.read_text())
        except PermissionError as exc:
            raise PermissionError(
                f"cannot read {manifest}. Ask xliu253 to grant read access on the val tree."
            ) from exc
        if poses.get("pose_convention") != "w2c_opencv":
            raise ValueError(f"{manifest} must hold OpenCV w2c poses")
        self.root = root
        self.near = near
        self.paths = {scene: entry["pose_path"] for scene, entry in poses["scenes"].items()}
        self.cache: dict[str, np.ndarray] = {}
        self.fallback_scenes: set[str] = set()

    def c2w(self, scene: str, frames: list[int]) -> torch.Tensor:
        if scene not in self.cache:
            if scene not in self.paths:
                raise KeyError(f"no poses for {scene}")
            with np.load(self.root / self.paths[scene]) as data:
                self.cache[scene] = np.asarray(data["w2c_poses"], dtype=np.float64)
        w2c = self.cache[scene]
        index = np.asarray(frames)
        if index.max() >= len(w2c):
            raise IndexError(f"{scene}: frame {int(index.max())} outside the pose array")
        c2w = np.linalg.inv(w2c[index])
        centers = c2w[:, :3, 3] - c2w[-1, :3, 3]
        if self.near is not None and scene in self.near:
            c2w[:, :3, 3] = centers / self.near[scene]
        else:
            self.fallback_scenes.add(scene)
            scale = np.linalg.norm(centers, axis=-1).mean()
            c2w[:, :3, 3] = centers / scale if scale > MIN_BASELINE else 0.0
        return torch.from_numpy(c2w).float()


def _resolve(root: Path, relative: str, extra_roots: tuple[str, ...]) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    candidates = [root / relative, *(Path(base) / relative for base in extra_roots)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _render_path(row: dict, root: Path, extra_roots: tuple[str, ...]) -> Path:
    """Stored degraded query. NVS manifests point ``src`` at a black image and
    keep the render in ``original_noisy_render``."""
    src = row["src"]
    if "black" not in Path(src).name:
        return _resolve(root, src, extra_roots)
    render = row.get("original_noisy_render")
    if not render:
        raise FileNotFoundError(
            f"{row.get('source_sample')}: src is black and original_noisy_render is missing"
        )
    return _resolve(root, render, extra_roots)


def _rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    try:
        image, _ = _load(path, size)
    except PermissionError as exc:
        raise PermissionError(
            f"cannot read {path}. Ask xliu253 to grant read access on the val tree."
        ) from exc
    return image


def _score(pending: list[dict], gallery: dict, device, out: Path) -> dict:
    metrics = _Metrics(device)
    rows = []
    for item in pending:
        pred, gt = item["pred"].to(device), item["gt"].to(device)
        rows.append({
            "scene": item["scene"], "sample": item["sample"],
            "frame": item["frame"], "view": item["view"], **metrics(pred, gt),
        })
        gallery[item["scene"]]["pred"].append(_uint8(pred))
        gallery[item["scene"]]["gt"].append(_uint8(gt))
    del metrics
    torch.cuda.empty_cache()
    by_scene: dict[str, list] = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    fid_overall, fid_scene = _fid(gallery, device)
    summary = {
        "overall": {**_mean(rows), "fid": fid_overall},
        "per_scene": {scene: {**_mean(items), "fid": fid_scene[scene]} for scene, items in by_scene.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return summary


def _report(dataset: str, task: str, summary: dict) -> None:
    overall = summary["overall"]
    print(
        f"[{dataset} {task}] n={overall['n']}  psnr={overall['psnr']:.3f}  "
        f"ssim={overall['ssim']:.4f}  lpips={overall['lpips']:.4f}  "
        f"dreamsim={overall['dreamsim']:.4f}  fid={overall['fid']:.2f}",
        flush=True,
    )


def _eval_panorama(model, dataset: str, spec: dict, tasks: tuple[str, ...], size: tuple[int, int],
                   num_views: int, steps: int, seed: int, near: dict[str, float] | None,
                   dest: Path, device) -> dict:
    root = Path(spec["root"])
    extra = tuple(spec["render_roots"])
    poses = _PanoramaPoses(root, near)
    manifest_for = {"nvs": spec["nvs_manifest"], "refine": spec["refine_manifest"] or spec["nvs_manifest"]}
    result = {"root": str(root), "image_size": list(size)}
    for task in tasks:
        sets = _index_sets(root, manifest_for[task], num_views)
        scenes = sorted({rows[0]["scene_name"] for rows in sets})
        print(f"{dataset} {task}: {len(sets)} sets, {len(scenes)} scenes from {manifest_for[task]}", flush=True)
        pending = []
        gallery = defaultdict(lambda: {"pred": [], "gt": []})
        out = dest / dataset / task
        for rows in sets:
            scene = rows[0]["scene_name"]
            short = scene.replace("::", "__")
            ref_paths = rows[0]["refs"]
            ref = torch.stack([_rgb(_resolve(root, path, ()), size) for path in ref_paths]).to(device)
            tgt = torch.stack([
                _rgb(_resolve(root, row["target"], ()), size) for row in rows
            ]).to(device)
            if task == "refine":
                src = torch.stack([
                    _rgb(_render_path(row, root, extra), size) for row in rows
                ]).to(device)
            else:
                src = torch.full_like(tgt, -1.0)
            frames = [_split_frame(path)[1] for path in ref_paths]
            frames += [_split_frame(row["target"])[1] for row in rows]
            c2w = poses.c2w(scene, frames)[None].to(device)
            k = torch.zeros(1, len(frames), 4, device=device)
            pred = _predict(model, src[None], ref[None], c2w, k, tgt[None], steps, task, seed, CAMERA_MODEL)[0]
            mode_dir = out / short
            mode_dir.mkdir(parents=True, exist_ok=True)
            sample = rows[0]["source_sample"]
            for view, row in enumerate(rows):
                gt = _match(tgt[view], pred[view])
                shown = _start(src[view], "black" if task == "nvs" else "prior")
                Image.fromarray(_panel(shown, pred[view], gt, ref)).save(mode_dir / f"{row['frame_index']:06d}_v{view}.png")
                pending.append({
                    "scene": scene, "sample": sample, "frame": int(row["frame_index"]), "view": view,
                    "pred": pred[view].half().cpu(), "gt": gt.half().cpu(),
                })
        summary = _score(pending, gallery, device, out)
        summary["manifest"] = manifest_for[task]
        summary["scenes"] = scenes
        summary["near_depth_fallback_scenes"] = sorted(
            scene for scene in scenes if scene in poses.fallback_scenes
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        _report(dataset, task, summary)
        result[task] = summary
        del pending, gallery
    if poses.fallback_scenes:
        print(
            f"{dataset}: {len(poses.fallback_scenes)} scenes have no near depth; "
            "translations use the mean camera baseline",
            flush=True,
        )
    return result


def _subsample(samples: list, rate: int) -> list:
    """Every ``rate``-th sample, starting at the first, in sample-id order."""
    ordered = sorted(samples, key=lambda sample: sample["id"])
    return ordered[::rate]


def _eval_dl3dv(model, root: Path, rate: int, steps: int, seed: int, size: tuple[int, int],
                tasks: tuple[str, ...], near: dict[str, float] | None, dest: Path, device) -> dict:
    if rate < 1:
        raise ValueError(f"dl3dv_sample_rate must be >= 1, got {rate}")
    print(f"indexing {root}", flush=True)
    scenes = _scan(root)
    picked = {}
    for scene in sorted(scenes):
        chosen = _subsample(scenes[scene], rate)
        if chosen:
            picked[scene] = chosen
    n_samples = sum(len(samples) for samples in picked.values())
    print(f"DL3DV: {len(picked)} scenes, {n_samples} samples (sample_rate={rate})", flush=True)
    result = {"root": str(root), "sample_rate": rate, "image_size": list(size), "scenes": sorted(picked)}
    for task in tasks:
        pending = []
        gallery = defaultdict(lambda: {"pred": [], "gt": []})
        out = dest / "DL3DV" / task
        for scene, samples in picked.items():
            if near is not None and scene not in near:
                raise KeyError(f"{scene} has no near depth")
            loaded = _materialize(samples, size, near and near[scene])
            for sample in loaded:
                ref = torch.stack([item["image"] for item in sample["refs"]]).to(device)
                tgt = torch.stack([item["image"] for item in sample["targets"]]).to(device)
                src = torch.stack([item["prior"] for item in sample["targets"]]).to(device)
                c2w = torch.stack(
                    [item["c2w"] for item in sample["refs"]] + [item["c2w"] for item in sample["targets"]]
                )[None].to(device)
                k = torch.stack(
                    [item["k"] for item in sample["refs"]] + [item["k"] for item in sample["targets"]]
                )[None].to(device)
                pred = _predict(model, src[None], ref[None], c2w, k, tgt[None], steps, task, seed, "pinhole")[0]
                mode_dir = out / scene
                mode_dir.mkdir(parents=True, exist_ok=True)
                for view, target in enumerate(sample["targets"]):
                    gt = _match(tgt[view], pred[view])
                    shown = _start(src[view], "black" if task == "nvs" else "prior")
                    Image.fromarray(_panel(shown, pred[view], gt, ref)).save(
                        mode_dir / f"{sample['id']}_v{view}.png"
                    )
                    pending.append({
                        "scene": scene, "sample": sample["id"], "frame": target["frame"], "view": view,
                        "pred": pred[view].half().cpu(), "gt": gt.half().cpu(),
                    })
            print(f"DL3DV {task} {scene[:12]} done", flush=True)
        summary = _score(pending, gallery, device, out)
        summary["sample_rate"] = rate
        summary["scenes"] = sorted(picked)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        _report("DL3DV", task, summary)
        result[task] = summary
        del pending, gallery
    return result


def main(
    run_dir: str,
    step: int = -1,
    tasks: str = "refine,nvs",
    datasets: str = "HM3D,Replica,JHU,DL3DV",
    out_dir: str | None = None,
    num_steps: int = 4,
    seed: int = 0,
    dl3dv_sample_rate: int = 1,
    hm3d_root: str = HM3D_ROOT,
    replica_root: str = REPLICA_ROOT,
    jhu_root: str = JHU_ROOT,
    jhu_render_root: str = JHU_RENDER_ROOT,
    dl3dv_root: str = DL3DV_ROOT,
    panorama_near_depth_json: str = PANO_NEAR,
    dl3dv_near_depth_json: str = DL3DV_NEAR,
) -> None:
    run = Path(run_dir)
    ckpt = _checkpoint(run, step)
    chosen_tasks = _tasks(tasks)
    chosen = _datasets(datasets)
    cfg = Config.from_yaml(run / "config.yaml")
    cfg.resume_from = str(ckpt)
    cfg.lpips_loss_weight = 0.0
    cfg.l1_pixel_loss_weight = 0.0
    pano_size = (int(cfg.panorama_image_size[0]), int(cfg.panorama_image_size[1]))
    dl3dv_size = (int(cfg.image_size[0]), int(cfg.image_size[1]))
    dest = Path(out_dir) if out_dir else Path("outputs/eval_ckpt") / run.name / ckpt.stem / f"n{num_steps}"
    dest.mkdir(parents=True, exist_ok=True)
    specs = {name: dict(item) for name, item in _PANO.items()}
    specs["HM3D"]["root"] = hm3d_root
    specs["Replica"]["root"] = replica_root
    specs["JHU"]["root"] = jhu_root
    specs["JHU"]["render_roots"] = (jhu_render_root,)
    pano_near = load_near_depth(panorama_near_depth_json) if panorama_near_depth_json else None
    dl3dv_near = load_near_depth(dl3dv_near_depth_json) if dl3dv_near_depth_json else None
    print(
        f"run={run} ckpt={ckpt.name} tasks={','.join(chosen_tasks)} "
        f"datasets={','.join(chosen)} out={dest}",
        flush=True,
    )
    device = torch.device("cuda")
    model = RayCrafterModel(cfg, device).eval()
    summary = {
        "run_dir": str(run), "ckpt": str(ckpt), "num_steps": int(num_steps),
        "tasks": list(chosen_tasks), "datasets": chosen, "dl3dv_sample_rate": int(dl3dv_sample_rate),
    }
    for name in chosen:
        if name == "DL3DV":
            summary[name] = _eval_dl3dv(
                model, Path(dl3dv_root), int(dl3dv_sample_rate), int(num_steps), int(seed),
                dl3dv_size, chosen_tasks, dl3dv_near, dest, device,
            )
        else:
            summary[name] = _eval_panorama(
                model, name, specs[name], chosen_tasks, pano_size, int(cfg.num_views),
                int(num_steps), int(seed), pano_near, dest, device,
            )
    # Per-task summaries are already on disk; the top-level file repeats overall numbers only.
    brief = {
        key: value for key, value in summary.items()
        if key not in chosen
    }
    for name in chosen:
        brief[name] = {
            task: summary[name][task]["overall"]
            for task in chosen_tasks
        }
    (dest / "summary.json").write_text(json.dumps(brief, indent=2))
    print(f"wrote {dest / 'summary.json'}", flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
