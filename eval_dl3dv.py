#!/usr/bin/env python3
"""NVS, refinement, and one NVS video for 10 DL3DV eval scenes.

Scenes are 10 names spread evenly over the test split (140 scenes, 20 samples
each). Refinement uses the stored point splat. NVS starts from one of two
sources (``--nvs_starts``):
  noise: the query latent is Gaussian noise (noise_alpha=1, the tar recipe);
  black: the query image is all black (VAE of black, noise_alpha=0). This is
         what dl3dv_points training feeds for nvs_ratio / nvs_probability.
The video keeps two refs fixed and walks every other stored frame of that
scene in frame order, once per NVS start.

    python eval_dl3dv.py \
        --runs klein9b_dl3dv_scope,klein9b_dl3dv_rayrope,klein9b_dl3dv_scope_rayrope \
        --nvs_starts black

``--step 0`` scores every run at the least-trained run's furthest step;
``--step -1`` scores each run at its own latest checkpoint.

Writes outputs/eval_dl3dv/<run>/<ckpt>/n<steps>/{refine,nvs_<start>,video_<start>}/.
Stills get PSNR, SSIM, LPIPS, and DreamSim per view in samples.jsonl.
summary.json has those four plus FID, per scene and over all scenes.
Videos are written and not scored.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent / "src"))

from raycrafter.config import Config
from raycrafter.data.dl3dv_points import _c2w, _scale_k, load_near_depth, scale_translation
from raycrafter.model import RayCrafterModel

DEFAULT_RUNS = (
    "klein9b_dl3dv_scope",
    "klein9b_dl3dv_rayrope",
    "klein9b_dl3dv_scope_rayrope",
)
# Already downloaded for the previous eval jobs. DreamSim looks here before fetching.
DREAMSIM_CACHE = os.environ.get(
    "DREAMSIM_CACHE", "/home/jhu/zzhan288/scratch_rchella4/zzhan288/geometric_syncfix/models"
)
IMAGE_METRICS = ("psnr", "ssim", "lpips", "dreamsim")
NVS_STARTS = ("noise", "black")


def _load(path: Path, size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        src_wh = image.size
        if src_wh != (size[1], size[0]):
            image = TF.resize(
                image, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
        return TF.to_tensor(image) * 2 - 1, src_wh


def _pose(row: dict, src_wh: tuple[int, int], size: tuple[int, int], near_depth: float | None):
    c2w = _c2w(row["w2c"])
    if near_depth is not None:
        c2w = scale_translation(c2w, near_depth)
    return c2w, _scale_k(row["camera"], src_wh, size)


def _scan(root: Path):
    """Metadata only. Images stay on disk until a scene is selected."""
    samples_dir = root / "priors" / "samples"
    scenes: dict[str, list] = {}
    for sample_dir in sorted(p for p in samples_dir.iterdir() if (p / "metadata.json").is_file()):
        meta = json.loads((sample_dir / "metadata.json").read_text())
        scene = meta["scene_name"]
        refs, targets = [], []
        for i, row in enumerate(meta["references"]):
            refs.append({
                "frame": int(row["frame_index"]),
                "path": sample_dir / f"ref_{i}.png",
                "row": row,
            })
        for i, row in enumerate(sorted(meta["targets"], key=lambda r: r["target_number"])):
            targets.append({
                "frame": int(row["frame_index"]),
                "path": sample_dir / f"gt_{i}.png",
                "prior": sample_dir / f"prior_{i}.png",
                "row": row,
            })
        scenes.setdefault(scene, []).append({"id": sample_dir.name, "refs": refs, "targets": targets})
    return scenes


def _materialize(samples: list, size: tuple[int, int], near_depth: float | None = None):
    """Load one scene: its samples, and one clean image per frame index.
    ``near_depth`` (the run's ``near_depth_json``) rescales the translations."""
    frames: dict[int, dict] = {}
    loaded = []
    for sample in samples:
        refs, targets = [], []
        for item in sample["refs"]:
            image, wh = _load(item["path"], size)
            c2w, k = _pose(item["row"], wh, size, near_depth)
            rec = {"frame": item["frame"], "image": image, "c2w": c2w, "k": k}
            refs.append(rec)
            frames.setdefault(item["frame"], rec)
        for item in sample["targets"]:
            gt, wh = _load(item["path"], size)
            prior, _ = _load(item["prior"], size)
            c2w, k = _pose(item["row"], wh, size, near_depth)
            rec = {"frame": item["frame"], "image": gt, "c2w": c2w, "k": k}
            targets.append({**rec, "prior": prior})
            frames.setdefault(item["frame"], rec)
        loaded.append({"id": sample["id"], "refs": refs, "targets": targets})
    return loaded, frames


def _pick_scenes(names: list[str], n: int) -> list[str]:
    if n >= len(names):
        return list(names)
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in dict.fromkeys(idx.tolist())]


def _saved_steps(run_dir: Path) -> list[int]:
    steps = [int(p.stem.removeprefix("transformer_step")) for p in run_dir.glob("transformer_step*.safetensors")]
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    return steps


def _shared_step(run_dirs: list[Path]) -> int:
    """The least-trained run sets the step, and every run is scored at that step."""
    furthest = {d.name: max(_saved_steps(d)) for d in run_dirs}
    step = min(furthest.values())
    print("furthest saved step: " + ", ".join(f"{k}={v}" for k, v in furthest.items()), flush=True)
    print(f"evaluating every run at step {step}", flush=True)
    return step


def _uint8(image: torch.Tensor) -> np.ndarray:
    chw = image.detach().float().clamp(-1, 1).cpu()
    return ((chw + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def _match(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    h, w = pred.shape[-2:]
    top, left = (gt.shape[-2] - h) // 2, (gt.shape[-1] - w) // 2
    return gt[..., top:top + h, left:left + w]


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
        self.device = device

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        pred, gt = pred.float(), gt.float()
        unit_p = (pred.clamp(-1, 1) + 1) * 0.5
        unit_g = (gt.clamp(-1, 1) + 1) * 0.5
        self.psnr.reset()
        self.ssim.reset()
        small_p = torch.nn.functional.interpolate(
            unit_p[None], (224, 224), mode="bicubic", antialias=True, align_corners=False,
        ).clamp(0, 1)
        small_g = torch.nn.functional.interpolate(
            unit_g[None], (224, 224), mode="bicubic", antialias=True, align_corners=False,
        ).clamp(0, 1)
        return {
            "psnr": float(self.psnr(unit_p[None], unit_g[None])),
            "ssim": float(self.ssim(unit_p[None], unit_g[None])),
            "lpips": float(self.lpips(pred[None], gt[None]).flatten()[0]),
            "dreamsim": float(self.dreamsim(small_p, small_g).flatten()[0]),
        }


def _mean(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in IMAGE_METRICS:
        out[key] = float(np.mean([r[key] for r in rows])) if rows else None
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


def _write_mp4(path: Path, frame_dir: Path, fps: float) -> None:
    import imageio_ffmpeg

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-crf", "18",
            str(path),
        ],
        check=True,
    )


def _start(src: torch.Tensor, start: str) -> torch.Tensor:
    """Query image fed to the model: the stored prior, or black (-1) for NVS."""
    return torch.full_like(src, -1.0) if start == "black" else src


def _forward(model, src, ref, c2w, k, tgt, steps: int, start: str, seed: int) -> torch.Tensor:
    """``start``: prior (refine), noise (Gaussian query latent), or black (black query image)."""
    device = src.device
    torch.manual_seed(seed)
    alpha = torch.full((src.shape[0],), float(start == "noise"), device=device)
    src = _start(src, start)
    return model.sample(
        src, ref, num_steps=steps, camera_c2w=c2w, camera_fxfycxcy=k,
        noise_alpha=alpha, tgt_imgs=tgt,
    )


def _panel(src, pred, gt, refs) -> np.ndarray:
    row = np.concatenate([_uint8(src), _uint8(pred), _uint8(gt)], axis=1)
    strip = np.concatenate([_uint8(r) for r in refs], axis=1)
    strip = np.array(Image.fromarray(strip).resize((row.shape[1], row.shape[1] * strip.shape[0] // strip.shape[1])))
    return np.concatenate([row, strip], axis=0)


def _eval_run(run_dir: Path, ckpt: Path, scenes, size, picked, steps: int, fps: float, seed: int,
              out_root: Path, nvs_starts: tuple[str, ...], near_depth_json: str | None = None):
    cfg = Config.from_yaml(run_dir / "config.yaml")
    cfg.resume_from = str(ckpt)
    cfg.lpips_loss_weight = 0.0
    cfg.l1_pixel_loss_weight = 0.0
    device = torch.device("cuda")
    model = RayCrafterModel(cfg, device).eval()
    near_depth_json = near_depth_json or cfg.near_depth_json
    near_depth = load_near_depth(near_depth_json) if near_depth_json else None
    out = out_root / run_dir.name / ckpt.stem / f"n{steps}"
    modes = [("refine", "prior")] + [(f"nvs_{s}", s) for s in nvs_starts]
    pending = {mode: [] for mode, _ in modes}

    for scene in picked:
        short = scene[:12]
        if near_depth is not None and scene not in near_depth:
            raise KeyError(f"{scene} has no near depth in {near_depth_json}")
        samples, frame_map = _materialize(scenes[scene], size, near_depth and near_depth[scene])
        for sample in samples:
            ref = torch.stack([r["image"] for r in sample["refs"]]).to(device)
            tgt = torch.stack([t["image"] for t in sample["targets"]]).to(device)
            src = torch.stack([t["prior"] for t in sample["targets"]]).to(device)
            c2w = torch.stack(
                [r["c2w"] for r in sample["refs"]] + [t["c2w"] for t in sample["targets"]]
            ).to(device)
            k = torch.stack(
                [r["k"] for r in sample["refs"]] + [t["k"] for t in sample["targets"]]
            ).to(device)
            for mode, start in modes:
                pred = _forward(
                    model, src[None], ref[None], c2w[None], k[None], tgt[None],
                    steps, start, seed,
                )[0]
                mode_dir = out / mode / short
                mode_dir.mkdir(parents=True, exist_ok=True)
                for v, target in enumerate(sample["targets"]):
                    gt = _match(tgt[v], pred[v])
                    # Gaussian noise never touches pixels: show gray for it.
                    shown = torch.zeros_like(src[v]) if start == "noise" else _start(src[v], start)
                    Image.fromarray(_panel(shown, pred[v], gt, ref)).save(
                        mode_dir / f"{sample['id']}_v{v}.png"
                    )
                    pending[mode].append({
                        "scene": scene, "sample": sample["id"], "frame": target["frame"], "view": v,
                        "pred": pred[v].half().cpu(), "gt": gt.half().cpu(),
                    })
        for start in nvs_starts:
            _video(model, frame_map, out / f"video_{start}" / short, steps, fps, seed, device, start)
        print(f"{run_dir.name} {short} stills done", flush=True)

    del model
    torch.cuda.empty_cache()
    metrics = _Metrics(device)
    rows = {mode: [] for mode in pending}
    gallery = {mode: defaultdict(lambda: {"pred": [], "gt": []}) for mode in pending}
    for mode, items in pending.items():
        for item in items:
            pred, gt = item["pred"].to(device), item["gt"].to(device)
            score = metrics(pred, gt)
            rows[mode].append({
                "scene": item["scene"], "sample": item["sample"],
                "frame": item["frame"], "view": item["view"], **score,
            })
            gallery[mode][item["scene"]]["pred"].append(_uint8(pred))
            gallery[mode][item["scene"]]["gt"].append(_uint8(gt))
    del metrics
    torch.cuda.empty_cache()

    summary = {"run": run_dir.name, "ckpt": str(ckpt), "num_steps": steps, "scenes": picked}
    for mode, items in rows.items():
        by_scene: dict[str, list] = {}
        for row in items:
            by_scene.setdefault(row["scene"], []).append(row)
        fid_overall, fid_scene = _fid(gallery[mode], device)
        summary[mode] = {
            "overall": {**_mean(items), "fid": fid_overall},
            "per_scene": {
                s: {**_mean(v), "fid": fid_scene[s]} for s, v in by_scene.items()
            },
        }
        (out / mode).mkdir(parents=True, exist_ok=True)
        (out / mode / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        overall = summary[mode]["overall"]
        print(
            f"[{run_dir.name} n{steps}] {mode:7s} n={overall['n']}  "
            f"psnr={overall['psnr']:.3f}  ssim={overall['ssim']:.4f}  "
            f"lpips={overall['lpips']:.4f}  dreamsim={overall['dreamsim']:.4f}  fid={overall['fid']:.2f}",
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def _video(model, frame_map, dest: Path, steps: int, fps: float, seed: int, device, start: str) -> None:
    ordered = [frame_map[i] for i in sorted(frame_map)]
    if len(ordered) < 4:
        raise RuntimeError(f"need at least 4 frames for a video, got {len(ordered)}")
    ref_i = {ordered[len(ordered) // 5]["frame"], ordered[4 * len(ordered) // 5]["frame"]}
    refs = [frame for frame in ordered if frame["frame"] in ref_i]
    queries = [frame for frame in ordered if frame["frame"] not in ref_i]
    ref = torch.stack([r["image"] for r in refs]).to(device)
    ref_c2w = torch.stack([r["c2w"] for r in refs])
    ref_k = torch.stack([r["k"] for r in refs])
    dest.mkdir(parents=True, exist_ok=True)
    frame_ids = []
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        pred_dir, cmp_dir = Path(tmp) / "pred", Path(tmp) / "cmp"
        pred_dir.mkdir()
        cmp_dir.mkdir()
        for i, frame in enumerate(queries):
            src = torch.zeros_like(frame["image"])[None, None].to(device)
            tgt = frame["image"][None, None].to(device)
            c2w = torch.cat([ref_c2w, frame["c2w"][None]], 0)[None].to(device)
            k = torch.cat([ref_k, frame["k"][None]], 0)[None].to(device)
            # One seed for the whole clip, so flicker is the camera, not the noise.
            pred = _forward(model, src, ref[None], c2w, k, tgt, steps, start, seed)[0, 0]
            gt = _match(frame["image"].to(device), pred)
            frame_ids.append(frame["frame"])
            p8, g8 = _uint8(pred), _uint8(gt)
            Image.fromarray(p8).save(pred_dir / f"frame_{i:04d}.png")
            Image.fromarray(np.concatenate([g8, p8], axis=1)).save(cmp_dir / f"frame_{i:04d}.png")
        _write_mp4(dest / "pred.mp4", pred_dir, fps)
        _write_mp4(dest / "compare.mp4", cmp_dir, fps)
    strip = np.concatenate([_uint8(r["image"]) for r in refs], axis=1)
    Image.fromarray(strip).save(dest / "refs.png")
    (dest / "frames.json").write_text(json.dumps({
        "refs": [r["frame"] for r in refs], "num_steps": steps, "start": start, "frames": frame_ids,
    }, indent=2))
    print(f"video_{start} {dest.name}: {len(queries)} frames", flush=True)


def main(
    data_root: str = "/home/jhu/zzhan288/projects_rchella4/zzhan288/data/dl3dv_points/test",
    outputs: str = "outputs",
    runs: str = ",".join(DEFAULT_RUNS),
    step: int = 0,
    num_scenes: int = 10,
    num_steps: int = 4,
    fps: float = 8.0,
    seed: int = 0,
    out_dir: str = "outputs/eval_dl3dv",
    nvs_starts: str = "black",
    near_depth_json: str | None = None,
) -> None:
    """``near_depth_json`` overrides the runs' own (their paths are this cluster's)."""
    root = Path(data_root)
    size = (432, 768)
    # Image size follows the first run; every 9B run here trains at 432x768.
    # Fire turns "a,b,c" into a tuple, so accept both.
    raw = runs.split(",") if isinstance(runs, str) else list(runs)
    run_names = [str(name).strip() for name in raw if str(name).strip()]
    raw = nvs_starts.split(",") if isinstance(nvs_starts, str) else list(nvs_starts)
    starts = tuple(str(s).strip() for s in raw if str(s).strip())
    if not set(starts) <= set(NVS_STARTS):
        raise ValueError(f"nvs_starts must be from {NVS_STARTS}, got {starts}")
    first_cfg = Config.from_yaml(Path(outputs) / run_names[0] / "config.yaml")
    size = (int(first_cfg.image_size[0]), int(first_cfg.image_size[1]))
    print(f"indexing {root} at {size[0]}x{size[1]}", flush=True)
    scenes = _scan(root)
    picked = _pick_scenes(sorted(scenes), num_scenes)
    print(f"{len(picked)} scenes, {sum(len(scenes[s]) for s in picked)} samples", flush=True)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "scenes.json").write_text(json.dumps(picked, indent=2))
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
        _eval_run(run_dir, weight, scenes, size, picked, num_steps, fps, seed, out_root, starts, near_depth_json)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
