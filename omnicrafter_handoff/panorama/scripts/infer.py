#!/usr/bin/env python3
"""One-process OmniCrafter inference; target ground truth is never model input."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from common import check_resolution, read_manifest, sha256, wan_assets, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--task", choices=("nvs", "cleanup"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--wan-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--vram-limit", type=float, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--validate-only", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    check_resolution(args.height, args.width)
    if args.steps < 1 or args.vram_limit <= 0:
        p.error("steps and vram-limit must be positive")
    manifest = read_manifest(args.manifest, args.task)
    print(f"Validated {len(manifest['sets'])} sets; task={args.task}; HxW={args.height}x{args.width}")
    if args.validate_only:
        return
    if any(v is None for v in (args.checkpoint, args.wan_root, args.output)):
        p.error("Inference requires --checkpoint, --wan-root and --output")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    wan_assets(args.wan_root)
    # New output only: never silently mix predictions from different models/tasks.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(manifest_sha256=sha256(args.manifest), checkpoint_sha256=sha256(args.checkpoint),
                  sigma_shift=5.0, timestamp=datetime.now(timezone.utc).isoformat(),
                  seed_policy="base_seed + manifest set index", target_gt_used=False)
    write_json(args.output / "run.json", config)

    import numpy as np
    import torch
    from PIL import Image
    from model_pose_conditioned_render_queries import PoseConditionedRenderQueryCrafter
    from diffsynth.core.data.dataset import get_equirectangular_plucker_rays, normalize_w2c_make_cam_last_origin
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = PoseConditionedRenderQueryCrafter(
        checkpoint_path=str(args.checkpoint), base_model_dir=str(args.wan_root),
        device=args.device, vram_limit=args.vram_limit, input_dim=420,
    )

    def rgb(path):
        with Image.open(path) as im:
            return im.convert("RGB").resize((args.width, args.height), Image.Resampling.BILINEAR)

    for index, item in enumerate(manifest["sets"]):
        refs = [rgb(v["image"]) for v in item["references"]]
        queries = ([Image.new("RGB", (args.width, args.height), (0, 0, 0)) for _ in item["targets"]]
                   if args.task == "nvs" else [rgb(v["noisy_image"]) for v in item["targets"]])
        poses = torch.from_numpy(np.asarray([v["w2c"] for v in item["references"] + item["targets"]], dtype=np.float32))
        _, c2w, scale = normalize_w2c_make_cam_last_origin(poses)
        rays = get_equirectangular_plucker_rays(c2w, height=args.height, width=args.width)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        predictions = model.generate_render_queries(refs, queries, rays, height=args.height, width=args.width,
            num_inference_steps=args.steps, seed=args.seed + index, cfg_scale=args.cfg_scale)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        if len(predictions) != len(item["targets"]):
            raise RuntimeError("Returned query count differs from manifest")
        destination = args.output / "color" / item["id"]
        destination.mkdir(parents=True)
        for q, prediction in enumerate(predictions):
            if prediction.size != (args.width, args.height):
                raise RuntimeError("Unexpected output size; refusing silent resize")
            prediction.save(destination / f"{q:06d}.png")
        write_json(destination / "metadata.json", {"id": item["id"], "task": args.task,
            "seed": args.seed + index, "pose_scale": float(scale), "seconds": elapsed,
            "target_fps": len(predictions) / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None})
        print(f"{index+1}/{len(manifest['sets'])}: {item['id']}, {elapsed:.2f} s", flush=True)
    (args.output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
