#!/usr/bin/env python3
"""NVS, refinement, and one NVS video for the 10 held-out JHU panorama scenes.

The panorama counterpart of ``eval_dl3dv.py`` (same metrics, panels, FID and
videos; ``camera_model = "equirect"``). Sets come from the held-out manifest
(``manifests/jhu_holdout10_seed20260822/jhu_test_2ref1_10_scenes.jsonl``):
``--sets_per_scene`` of them per scene, evenly spaced over that scene's sets.
Refinement uses the stored render (no extra dropout, as in training's
validation). NVS starts from a black query image (``black``) or a Gaussian
query latent (``noise``). The video keeps two refs fixed and walks the scene's
frames in order (at most ``--video_frames``, evenly spaced), once per NVS start.

Poses are normalized as in training: re-centered on the last camera, and
translations divided by the scene's near depth: ``--near_depth_json`` if given,
else the run's ``panorama_near_depth_json``, else ``<root>/near_depth.json``.
``--root`` points at the dataset on another machine (it holds the manifest,
``poses/`` and the images).

    python eval_panorama.py --runs flux3_mix_scope,flux3_mix_rayrope,flux3_mix_unirope

``--step 0`` scores every run at the least-trained run's furthest step;
``--step -1`` scores each run at its own latest checkpoint.

Writes outputs/eval_panorama/<run>/<ckpt>/n<steps>/{refine,nvs_<start>,video_<start>}/.
Metrics are on the full equirect image (PSNR, SSIM, LPIPS; DreamSim on the
224×224 resize, as for DL3DV), so they weight the poles like the equator.
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval_dl3dv import (
    NVS_STARTS, _fid, _match, _mean, _Metrics, _panel, _saved_steps, _shared_step, _start, _uint8,
    _write_mp4,
)
from raycrafter.config import Config
from raycrafter.data.dl3dv_points import load_near_depth
from raycrafter.data.panorama import CAMERA_MODEL, PanoramaSets
from raycrafter.model import RayCrafterModel

DEFAULT_RUNS = ("flux3_mix_scope", "flux3_mix_rayrope", "flux3_mix_unirope")
ROOT = "/home/jhu/zzhan288/projects_rchella4/zzhan288/data/framecrafter_5input"
MANIFEST = "manifests/jhu_holdout10_seed20260822/jhu_test_2ref1_10_scenes.jsonl"


def _pick_sets(ds: PanoramaSets, per_scene: int) -> dict[str, list[int]]:
    """``per_scene`` set indices per scene, evenly spaced over its (sorted) sets."""
    by_scene: dict[str, list[int]] = defaultdict(list)
    for i, scene in enumerate(ds.scenes):
        by_scene[scene.decode()].append(i)
    picked = {}
    for scene in sorted(by_scene):
        idx = by_scene[scene]
        take = np.linspace(0, len(idx) - 1, min(per_scene, len(idx))).round().astype(int)
        picked[scene] = [idx[k] for k in dict.fromkeys(take.tolist())]
    return picked


def _forward(model, src, ref, c2w, tgt, steps: int, start: str, seed: int) -> torch.Tensor:
    """``start``: prior (refine), noise (Gaussian query latent), or black (black query image)."""
    torch.manual_seed(seed)
    alpha = torch.full((src.shape[0],), float(start == "noise"), device=src.device)
    k = torch.zeros(*c2w.shape[:2], 4, device=src.device)
    return model.sample(
        _start(src, start), ref, num_steps=steps, camera_c2w=c2w, camera_fxfycxcy=k,
        noise_alpha=alpha, tgt_imgs=tgt, camera_model=CAMERA_MODEL,
    )


def _video(model, ds: PanoramaSets, scene: str, dest: Path, steps: int, fps: float, seed: int,
           device, start: str, max_frames: int) -> None:
    dataset, name = scene.split("::", 1)
    image_dir = f"{dataset}/images/{name}"
    frames = sorted(int(p.stem) for p in (Path(ds.root) / image_dir).glob("*.png"))
    if len(frames) < 4:
        raise RuntimeError(f"need at least 4 frames for a video, got {len(frames)}")
    refs = [frames[len(frames) // 5], frames[4 * len(frames) // 5]]
    queries = [f for f in frames if f not in refs]
    if len(queries) > max_frames:
        queries = [queries[i] for i in np.linspace(0, len(queries) - 1, max_frames).round().astype(int)]
    def load(frame: int) -> torch.Tensor:
        return TF.to_tensor(ds._load(f"{image_dir}/{frame:06d}.png")) * 2 - 1

    ref = torch.stack([load(f) for f in refs]).to(device)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        pred_dir, cmp_dir = Path(tmp) / "pred", Path(tmp) / "cmp"
        pred_dir.mkdir()
        cmp_dir.mkdir()
        for i, frame in enumerate(queries):
            gt = load(frame).to(device)
            # Cameras of this triplet, normalized like a training set (last = query).
            c2w = ds.frame_cameras(scene, refs + [frame])[None].to(device)
            pred = _forward(model, torch.zeros_like(gt)[None, None], ref[None], c2w, gt[None, None],
                            steps, start, seed)[0, 0]
            p8, g8 = _uint8(pred), _uint8(_match(gt, pred))
            Image.fromarray(p8).save(pred_dir / f"frame_{i:04d}.png")
            Image.fromarray(np.concatenate([g8, p8], axis=0)).save(cmp_dir / f"frame_{i:04d}.png")
        _write_mp4(dest / "pred.mp4", pred_dir, fps)
        _write_mp4(dest / "compare.mp4", cmp_dir, fps)
    Image.fromarray(np.concatenate([_uint8(r) for r in ref], axis=0)).save(dest / "refs.png")
    (dest / "frames.json").write_text(json.dumps(
        {"refs": refs, "num_steps": steps, "start": start, "frames": queries}, indent=2))
    print(f"video_{start} {dest.name}: {len(queries)} frames", flush=True)


def _eval_run(run_dir: Path, ckpt: Path | None, picked: dict[str, list[int]], manifest: str,
              steps: int, fps: float, seed: int, out_root: Path, nvs_starts: tuple[str, ...],
              near_depth_json: str | None, video_frames: int, root: str = ROOT):
    cfg = Config.from_yaml(run_dir / "config.yaml")
    cfg.resume_from = str(ckpt) if ckpt else None
    cfg.lpips_loss_weight = 0.0
    cfg.l1_pixel_loss_weight = 0.0
    near_depth_json = near_depth_json or cfg.panorama_near_depth_json or f"{root}/near_depth.json"
    size = tuple(int(v) for v in cfg.panorama_image_size)
    ds = PanoramaSets(root, [manifest], image_size=size, min_pixel_dropout=0.0, max_pixel_dropout=0.0,
                      near_depth=load_near_depth(near_depth_json) if near_depth_json else None)
    device = torch.device("cuda")
    model = RayCrafterModel(cfg, device).eval()
    out = out_root / run_dir.name / (ckpt.stem if ckpt else "init") / f"n{steps}"
    modes = [("refine", "prior")] + [(f"nvs_{s}", s) for s in nvs_starts]
    pending = {mode: [] for mode, _ in modes}

    for scene, indices in picked.items():
        short = scene.split("::", 1)[1]
        for index in indices:
            item = ds[index]
            src, tgt, ref = (item[k].to(device) for k in ("src", "tgt", "ref"))
            c2w = item["camera_c2w"].to(device)
            for mode, start in modes:
                pred = _forward(model, src[None], ref[None], c2w[None], tgt[None], steps, start, seed)[0]
                mode_dir = out / mode / short
                mode_dir.mkdir(parents=True, exist_ok=True)
                for v in range(pred.shape[0]):
                    gt = _match(tgt[v], pred[v])
                    shown = torch.zeros_like(src[v]) if start == "noise" else _start(src[v], start)
                    Image.fromarray(_panel(shown, pred[v], gt, ref)).save(mode_dir / f"{index:06d}_v{v}.png")
                    pending[mode].append({
                        "scene": scene, "sample": item["key"], "frame": int(ds.target_frames[index][v]),
                        "view": v, "pred": pred[v].half().cpu(), "gt": gt.half().cpu(),
                    })
        for start in nvs_starts:
            _video(model, ds, scene, out / f"video_{start}" / short, steps, fps, seed, device, start,
                   video_frames)
        print(f"{run_dir.name} {short} stills done", flush=True)

    del model
    torch.cuda.empty_cache()
    metrics = _Metrics(device)
    rows = {mode: [] for mode in pending}
    gallery = {mode: defaultdict(lambda: {"pred": [], "gt": []}) for mode in pending}
    for mode, items in pending.items():
        for item in items:
            pred, gt = item["pred"].to(device), item["gt"].to(device)
            rows[mode].append({k: item[k] for k in ("scene", "sample", "frame", "view")} | metrics(pred, gt))
            gallery[mode][item["scene"]]["pred"].append(_uint8(pred))
            gallery[mode][item["scene"]]["gt"].append(_uint8(gt))
    del metrics
    torch.cuda.empty_cache()

    summary = {"run": run_dir.name, "ckpt": str(ckpt), "num_steps": steps, "scenes": list(picked),
               "near_depth_json": near_depth_json}
    for mode, items in rows.items():
        by_scene: dict[str, list] = defaultdict(list)
        for row in items:
            by_scene[row["scene"]].append(row)
        fid_overall, fid_scene = _fid(gallery[mode], device)
        summary[mode] = {
            "overall": {**_mean(items), "fid": fid_overall},
            "per_scene": {s: {**_mean(v), "fid": fid_scene[s]} for s, v in by_scene.items()},
        }
        (out / mode).mkdir(parents=True, exist_ok=True)
        (out / mode / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        overall = summary[mode]["overall"]
        print(
            f"[{run_dir.name} n{steps}] {mode:9s} n={overall['n']}  "
            f"psnr={overall['psnr']:.3f}  ssim={overall['ssim']:.4f}  "
            f"lpips={overall['lpips']:.4f}  dreamsim={overall['dreamsim']:.4f}  fid={overall['fid']:.2f}",
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main(
    outputs: str = "outputs",
    runs: str = ",".join(DEFAULT_RUNS),
    step: int = 0,
    manifest: str = MANIFEST,
    sets_per_scene: int = 20,
    num_steps: int = 4,
    fps: float = 8.0,
    seed: int = 0,
    out_dir: str = "outputs/eval_panorama",
    nvs_starts: str = "black",
    near_depth_json: str | None = None,
    video_frames: int = 60,
    root: str = ROOT,
) -> None:
    # Fire turns "a,b,c" into a tuple, so accept both.
    raw = runs.split(",") if isinstance(runs, str) else list(runs)
    run_names = [str(name).strip() for name in raw if str(name).strip()]
    raw = nvs_starts.split(",") if isinstance(nvs_starts, str) else list(nvs_starts)
    starts = tuple(str(s).strip() for s in raw if str(s).strip())
    if not set(starts) <= set(NVS_STARTS):
        raise ValueError(f"nvs_starts must be from {NVS_STARTS}, got {starts}")
    # The set index (grouping, order) does not depend on image size or near depth.
    picked = _pick_sets(PanoramaSets(root, [manifest]), sets_per_scene)
    print(f"{len(picked)} scenes, {sum(map(len, picked.values()))} sets", flush=True)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sets.json").write_text(json.dumps(picked, indent=2))
    run_dirs = [Path(outputs) / name for name in run_names]
    step = int(step)
    if step == 0:
        step = _shared_step(run_dirs)
    weights = {}
    for run_dir in run_dirs:
        # -1: each run at its own latest checkpoint.
        run_step = max(_saved_steps(run_dir)) if step < 0 else step
        weights[run_dir] = run_dir / f"transformer_step{run_step}.safetensors"
        if not weights[run_dir].is_file():
            raise FileNotFoundError(f"{run_dir.name} has no checkpoint at step {run_step}")
    for run_dir, weight in weights.items():
        print(f"=== {run_dir.name}  {weight.name}  nvs_starts={','.join(starts)} ===", flush=True)
        _eval_run(run_dir, weight, picked, manifest, num_steps, fps, seed, out_root, starts,
                  near_depth_json, video_frames, root)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
