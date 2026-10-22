"""Per-scene near depth for the DL3DV point-splat set (SCoPE's pose normalization).

SCoPE divides camera translation by a per-clip ``near_depth`` so scenes with
arbitrary COLMAP scale share one motion magnitude. This is SCoPE's
``scripts/estimate_near_depth.py`` (RAFT flow with a forward-backward check,
closed-form triangulation from the known relative pose, ``near_depth`` = median
over frame pairs of the 25th-percentile depth, DL3DV central crop 0.8), run on
the frames our samples already hold: every ``ref_*.png`` / ``gt_*.png`` of a
scene, deduplicated by frame index, with that frame's w2c and K. Pairs step
through the frames in index order, each at least ``--pair_stride`` frames
after the last (the original's stride on a dense clip).

Output (resumable, flushed every 50 scenes), keyed by scene name as
``near_depth_json`` expects::

    {"<scene_name>": {"near_depth": 1.23, "n_pairs_total": 20, ...}, ...}

Usage (GPU)::

    python scripts/estimate_near_depth_dl3dv_points.py \\
        --roots /path/dl3dv_points/train /path/dl3dv_points/test \\
        --output /path/dl3dv_points/near_depth.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raycrafter.data.dl3dv_points import _c2w, _scale_k  # noqa: E402

SCHEMA_VERSION = 1
FLUSH_EVERY = 50
CENTRAL_CROP = 0.8  # SCoPE's DL3DV setting: frames have mild edge distortion


# ── From SCoPE scripts/estimate_near_depth.py (Apache-2.0), unchanged ───────


class RAFTRunner:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _pad8(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, pad_h, pad_w

    @torch.no_grad()
    def flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        x1, pad_h, pad_w = self._pad8(img1)
        x2, _, _ = self._pad8(img2)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            flows = self.model(x1, x2)
        flow = flows[-1].float()
        if pad_h or pad_w:
            flow = flow[..., : img1.shape[-2], : img1.shape[-1]]
        return flow


def _build_grid(height: int, width: int, device) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def _fb_consistency(flow12: torch.Tensor, flow21: torch.Tensor) -> torch.Tensor:
    _, height, width = flow12.shape
    grid = _build_grid(height, width, flow12.device)
    warped = grid + flow12.permute(1, 2, 0)
    norm = warped.clone()
    norm[..., 0] = 2.0 * norm[..., 0] / max(width - 1, 1) - 1.0
    norm[..., 1] = 2.0 * norm[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        flow21.unsqueeze(0), norm.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=True,
    )[0]
    return torch.linalg.norm(flow12 + sampled, dim=0)


def _skew(t: torch.Tensor) -> torch.Tensor:
    x, y, z = t.unbind(-1)
    zero = torch.zeros_like(x)
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _triangulate_closed_form(p1_pix, p2_pix, K, R_12, t_12):
    height, width, _ = p1_pix.shape
    device = p1_pix.device
    ones = torch.ones(height, width, 1, device=device, dtype=p1_pix.dtype)
    p1h = torch.cat([p1_pix, ones], dim=-1)
    p2h = torch.cat([p2_pix, ones], dim=-1)
    k_inv = torch.linalg.inv(K).to(p1h.dtype)
    r1 = p1h @ k_inv.T
    r2 = p2h @ k_inv.T
    rr1 = r1 @ R_12.T
    A = torch.stack([rr1, -r2], dim=-1)
    ata = A.transpose(-1, -2) @ A
    rhs = (-t_12).expand(height, width, 3)
    atb = (A.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)
    det = ata[..., 0, 0] * ata[..., 1, 1] - ata[..., 0, 1] * ata[..., 1, 0]
    det_safe = det.clone()
    det_safe[det_safe.abs() < 1e-12] = 1e-12
    z1 = (ata[..., 1, 1] / det_safe) * atb[..., 0] + (-ata[..., 0, 1] / det_safe) * atb[..., 1]
    z2 = (-ata[..., 1, 0] / det_safe) * atb[..., 0] + (ata[..., 0, 0] / det_safe) * atb[..., 1]
    return z1, z2, det, det.abs() > 1e-8


def _epipolar_error_px(p1_pix, p2_pix, K, R_12, t_12):
    height, width, _ = p1_pix.shape
    device = p1_pix.device
    ones = torch.ones(height, width, 1, device=device, dtype=p1_pix.dtype)
    p1h = torch.cat([p1_pix, ones], dim=-1)
    p2h = torch.cat([p2_pix, ones], dim=-1)
    k_inv = torch.linalg.inv(K)
    fmat = k_inv.T @ _skew(t_12) @ R_12 @ k_inv
    l2 = p1h @ fmat.T
    n2 = torch.sqrt(l2[..., 0] ** 2 + l2[..., 1] ** 2 + 1e-12)
    d_fwd = (p2h * l2).sum(-1).abs() / n2
    l1 = p2h @ fmat
    n1 = torch.sqrt(l1[..., 0] ** 2 + l1[..., 1] ** 2 + 1e-12)
    d_bwd = (p1h * l1).sum(-1).abs() / n1
    return 0.5 * (d_fwd + d_bwd)


def _estimate_pair_p25(img1, img2, K, R_12, t_12, raft, central_crop_ratio, min_inliers=1000):
    device = img1.device
    height, width = img1.shape[-2], img1.shape[-1]
    f12 = raft.flow(img1, img2)[0]
    f21 = raft.flow(img2, img1)[0]
    fb = _fb_consistency(f12, f21)
    grid = _build_grid(height, width, device)
    p2 = grid + f12.permute(1, 2, 0)
    epi = _epipolar_error_px(grid, p2, K, R_12, t_12)
    z1, z2, _, ok_det = _triangulate_closed_form(grid, p2, K, R_12, t_12)
    thresh = 1.5 * max(height, width) / 832.0
    mask_base = (fb < 1.5) & (epi < thresh) & (z1 > 0) & (z2 > 0) & ok_det & (z1 < 1e4) & (z2 < 1e4)
    if central_crop_ratio < 1.0:
        pad_h = int(height * (1 - central_crop_ratio) / 2)
        pad_w = int(width * (1 - central_crop_ratio) / 2)
        cc = torch.zeros_like(mask_base)
        cc[pad_h : height - pad_h, pad_w : width - pad_w] = True
        mask = mask_base & cc
    else:
        mask = mask_base
    n_valid = int(mask.sum().item())
    if n_valid < min_inliers:
        return None
    p25 = float(torch.quantile(z1[mask], 0.25).item())
    if central_crop_ratio >= 1.0:
        denom = float(mask_base.numel())
    else:
        pad_h = int(height * (1 - central_crop_ratio) / 2)
        pad_w = int(width * (1 - central_crop_ratio) / 2)
        denom = float((height - 2 * pad_h) * (width - 2 * pad_w))
    if not np.isfinite(p25) or p25 <= 0:
        return None
    return p25, n_valid / max(denom, 1.0)


def _rel_pose(c2w_i: np.ndarray, c2w_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_i, r_j = c2w_i[:3, :3], c2w_j[:3, :3]
    c_i, c_j = c2w_i[:3, 3], c2w_j[:3, 3]
    return (r_j.T @ r_i).astype(np.float32), (r_j.T @ (c_i - c_j)).astype(np.float32)


def _pil_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return tensor.to(device, non_blocking=True)


def atomic_dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# ── DL3DV point-splat samples → per-scene frames ────────────────────────────


def _sample_frames(sample_dir: Path) -> tuple[str, list[tuple[int, Path, dict]]]:
    meta = json.loads((sample_dir / "metadata.json").read_text())
    frames = [(int(r["frame_index"]), sample_dir / f"ref_{i}.png", r) for i, r in enumerate(meta["references"])]
    targets = sorted(meta["targets"], key=lambda r: r["target_number"])
    frames += [(int(r["frame_index"]), sample_dir / f"gt_{i}.png", r) for i, r in enumerate(targets)]
    return meta["scene_name"], frames


def scan_scenes(roots: list[Path]) -> dict[str, dict[int, tuple[Path, dict]]]:
    """scene → {frame index: (clean image, metadata row)} over every sample of every root."""
    sample_dirs = [
        p for root in roots for p in sorted((root / "priors" / "samples").iterdir())
        if (p / "metadata.json").is_file()
    ]
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(_sample_frames, sample_dirs))
    scenes: dict[str, dict[int, tuple[Path, dict]]] = {}
    for scene, frames in results:
        for index, path, row in frames:
            scenes.setdefault(scene, {}).setdefault(index, (path, row))
    return scenes


def frame_pairs(indices: list[int], stride: int, max_pairs: int) -> list[tuple[int, int]]:
    """Consecutive pairs through the sorted frames, each frame ≥ ``stride`` after
    the last; consecutive frames if no two are that far apart."""
    pairs, last = [], indices[0]
    for index in indices[1:]:
        if index - last >= stride:
            pairs.append((last, index))
            last = index
            if len(pairs) >= max_pairs:
                break
    if not pairs:
        pairs = list(zip(indices[:-1], indices[1:]))[:max_pairs]
    return pairs


def process_scene(frames: dict[int, tuple[Path, dict]], raft, device, stride: int, max_pairs: int) -> dict:
    empty = {"near_depth": None, "n_pairs_total": 0, "n_pairs_valid": 0, "mean_inlier_ratio": 0.0,
             "version": SCHEMA_VERSION}
    indices = sorted(frames)
    if len(indices) < 2:
        return {**empty, "note": f"{len(indices)} frame(s)"}
    pairs = frame_pairs(indices, stride, max_pairs)
    images, poses, intrinsics = {}, {}, {}
    for index in sorted({i for pair in pairs for i in pair}):
        path, row = frames[index]
        images[index] = _pil_to_tensor(path, device)
        height, width = images[index].shape[-2:]
        fx, fy, cx, cy = _scale_k(row["camera"], (width, height), (height, width)).tolist()
        intrinsics[index] = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
        poses[index] = _c2w(row["w2c"]).numpy()
    p25_list, ratio_list = [], []
    for i, j in pairs:
        r_np, t_np = _rel_pose(poses[i], poses[j])
        if float(np.linalg.norm(t_np)) < 1e-6 or images[i].shape != images[j].shape:
            continue
        res = _estimate_pair_p25(
            images[i], images[j], intrinsics[i], torch.from_numpy(r_np).to(device),
            torch.from_numpy(t_np).to(device), raft=raft, central_crop_ratio=CENTRAL_CROP,
        )
        if res is not None:
            p25_list.append(res[0])
            ratio_list.append(res[1])
    if not p25_list:
        return {**empty, "n_pairs_total": len(pairs)}
    return {
        "near_depth": round(float(np.median(p25_list)), 6),
        "n_pairs_total": len(pairs),
        "n_pairs_valid": len(p25_list),
        "mean_inlier_ratio": round(float(np.mean(ratio_list)), 4),
        "version": SCHEMA_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="dl3dv_points split directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair_stride", type=int, default=5)
    parser.add_argument("--max_pairs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    scenes = scan_scenes(args.roots)
    names = sorted(scenes)[: args.limit]
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    print(f"[near_depth] {len(scenes)} scenes, {len(existing)} already in {args.output}", flush=True)

    torch.backends.cudnn.benchmark = True
    raft = RAFTRunner(args.device)
    device = torch.device(args.device)
    processed = nulls = 0
    start = time.time()
    for k, scene in enumerate(names):
        if scene in existing:
            continue
        record = process_scene(scenes[scene], raft, device, args.pair_stride, args.max_pairs)
        existing[scene] = record
        processed += 1
        nulls += record["near_depth"] is None
        if processed % FLUSH_EVERY == 0:
            atomic_dump(args.output, existing)
            rate = processed / (time.time() - start)
            print(f"[{k + 1}/{len(names)}] {rate:.2f} scenes/s, {nulls} without an estimate", flush=True)
    atomic_dump(args.output, existing)
    print(f"[done] {len(existing)} scenes in {args.output}; {processed} new, {nulls} without an estimate")


if __name__ == "__main__":
    main()
