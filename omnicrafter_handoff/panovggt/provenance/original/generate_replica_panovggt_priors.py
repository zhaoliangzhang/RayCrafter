#!/usr/bin/env python3
"""Generate PanoVGGT point-cloud renders for a two-reference ERP benchmark.

Only the two benchmark reference panoramas are passed to PanoVGGT.  Its
predicted cameras and point cloud are aligned to the two known reference poses,
then raw z-buffered point splats are rendered at the three benchmark target
poses.  Target RGB images are never opened by this program.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panovggt_point_predictor import PanoVGGTPointPredictor
from render_jhu_heldout import align_cameras, render_erp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-sets",
        type=int,
        default=-1,
        help="Use only the first N benchmark sets; -1 means all sets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--degenerate-baseline-threshold",
        type=float,
        default=1e-4,
        help="Reference-center distance below which scale is unobservable.",
    )
    parser.add_argument(
        "--fallback-scale-calibration",
        type=Path,
        help="JSON produced by calibrate_replica_panovggt_scale.py.",
    )
    parser.add_argument(
        "--only-degenerate-baselines",
        action="store_true",
        help="Process only sets whose two reference centers are co-located.",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents)
    os.replace(temporary, path)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG", compress_level=4)
    os.replace(temporary, path)


def validate_benchmark(manifest: dict, height: int, width: int) -> list[dict]:
    if manifest.get("pose_convention") != "OpenCV world-to-camera":
        raise ValueError("Benchmark must use OpenCV world-to-camera poses")
    resolution = manifest.get("resolution", {})
    if (resolution.get("height"), resolution.get("width")) != (height, width):
        raise ValueError(
            f"Benchmark resolution is {resolution}, requested {(height, width)}"
        )
    sets = manifest.get("sets", [])
    if manifest.get("logical_sets") != len(sets):
        raise ValueError("Benchmark logical_sets does not match its set list")
    expected_targets = sum(len(item.get("targets", [])) for item in sets)
    if manifest.get("target_images") != expected_targets:
        raise ValueError(
            f"Manifest target count is {manifest.get('target_images')}, expected {expected_targets}"
        )
    seen = set()
    for expected_index, item in enumerate(sets):
        expected_id = f"sample_{expected_index:06d}"
        if item.get("id") != expected_id:
            raise ValueError(f"Expected {expected_id}, got {item.get('id')}")
        if item["id"] in seen:
            raise ValueError(f"Duplicate benchmark id: {item['id']}")
        seen.add(item["id"])
        if len(item.get("references", [])) != 2:
            raise ValueError(f"{item['id']} does not have exactly two references")
        if len(item.get("targets", [])) < 1:
            raise ValueError(f"{item['id']} has no targets")
        for number, target in enumerate(item["targets"]):
            if target.get("target_number") != number:
                raise ValueError(f"Non-contiguous targets in {item['id']}")
        for view in item["references"] + item["targets"]:
            pose = np.asarray(view["w2c"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid pose in {item['id']}")
            if not Path(view["image"]).is_file():
                raise FileNotFoundError(view["image"])
    return sets


def completed(sample_dir: Path, width: int, height: int, num_targets: int) -> bool:
    if not (sample_dir / "_SUCCESS").is_file():
        return False
    required = [sample_dir / f"prior_{index}.png" for index in range(num_targets)]
    required += [sample_dir / f"mask_{index}.png" for index in range(num_targets)]
    required.append(sample_dir / "metadata.json")
    if not all(path.is_file() for path in required):
        return False
    try:
        for path in required[: 2 * num_targets]:
            with Image.open(path) as image:
                if image.size != (width, height):
                    return False
    except OSError:
        return False
    return True


def reference_baseline(item: dict) -> float:
    w2c = np.stack(
        [np.asarray(view["w2c"], dtype=np.float64) for view in item["references"]]
    )
    centers = np.linalg.inv(w2c)[:, :3, 3]
    return float(np.linalg.norm(centers[1] - centers[0]))


def process_set(
    item: dict,
    predictor: PanoVGGTPointPredictor,
    output: Path,
    height: int,
    width: int,
    degenerate_baseline_threshold: float,
    fallback_scale: float | None,
) -> dict:
    started = time.monotonic()
    sample_dir = output / "samples" / item["id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    success = sample_dir / "_SUCCESS"
    if success.exists():
        success.unlink()

    reference_paths = [Path(view["image"]) for view in item["references"]]
    reference_w2c = np.stack(
        [np.asarray(view["w2c"], dtype=np.float64) for view in item["references"]]
    )
    reference_c2w = np.linalg.inv(reference_w2c)

    # No target path is passed to this call.  The only RGB input consists of
    # the two reference panoramas listed above.
    predictions = predictor.predict_from_paths(reference_paths)
    predicted_c2w = predictions["camera_poses"][0].astype(np.float64)
    if predicted_c2w.shape != (2, 4, 4):
        raise ValueError(
            f"PanoVGGT returned camera shape {predicted_c2w.shape}, expected (2,4,4)"
        )
    points = predictions["world_points"][0].reshape(-1, 3).astype(np.float64)
    colors = (
        predictions["images"][0]
        .transpose(0, 2, 3, 1)
        .reshape(-1, 3)
        .astype(np.float64)
    )
    if len(points) != len(colors):
        raise ValueError("PanoVGGT point/color counts do not match")

    scale, rotation, translation, alignment = align_cameras(
        predicted_c2w, reference_c2w
    )
    if alignment["gt_baseline"] < degenerate_baseline_threshold:
        if fallback_scale is None:
            alignment["scale_method"] = "degenerate reference baseline; uncalibrated"
        else:
            uncalibrated_scale = scale
            scale = fallback_scale
            predicted_centers = predicted_c2w[:, :3, 3]
            reference_centers = reference_c2w[:, :3, 3]
            translation = np.mean(
                reference_centers
                - scale * (rotation @ predicted_centers.T).T,
                axis=0,
            )
            aligned_centers = (
                scale * (rotation @ predicted_centers.T).T + translation
            )
            alignment.update(
                {
                    "uncalibrated_degenerate_scale": float(uncalibrated_scale),
                    "scale": float(scale),
                    "scale_method": (
                        "global median from non-degenerate benchmark reference pairs"
                    ),
                    "condition_center_errors": [
                        float(value)
                        for value in np.linalg.norm(
                            aligned_centers - reference_centers, axis=1
                        )
                    ],
                }
            )
    else:
        alignment["scale_method"] = "per-set two-reference baseline ratio"
    aligned_points = scale * (points @ rotation.T) + translation
    finite = (
        np.isfinite(aligned_points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
    )
    aligned_points = aligned_points[finite]
    colors = np.clip(colors[finite], 0.0, 1.0)
    if not len(aligned_points):
        raise ValueError(f"PanoVGGT produced no finite points for {item['id']}")

    target_records = []
    for number, target in enumerate(item["targets"]):
        target_w2c = np.asarray(target["w2c"], dtype=np.float64)
        render, _depth, mask = render_erp(
            aligned_points, colors, target_w2c, height, width
        )
        atomic_save_png(
            Image.fromarray(render), sample_dir / f"prior_{number}.png"
        )
        atomic_save_png(
            Image.fromarray(mask), sample_dir / f"mask_{number}.png"
        )
        target_records.append(
            {
                "target_number": number,
                "frame_index": int(target["frame_index"]),
                "target_pose_used": True,
                "target_rgb_accessed": False,
                "prior": str(sample_dir / f"prior_{number}.png"),
                "mask": str(sample_dir / f"mask_{number}.png"),
                "raw_pixel_coverage": float((mask > 0).mean()),
            }
        )

    record = {
        "set_id": item["id"],
        "source_sample": item["source_sample"],
        "scene_name": item["scene_name"],
        "input_reference_images": [str(path) for path in reference_paths],
        "num_reference_images": 2,
        "num_query_renders": len(item["targets"]),
        "render_type": "raw PanoVGGT point splat with nearest-depth z-buffer",
        "render_resolution": [height, width],
        "target_rgb_accessed": False,
        "finite_point_count": int(len(aligned_points)),
        "alignment": {
            **alignment,
            "similarity_rotation": rotation.tolist(),
            "similarity_translation": translation.tolist(),
        },
        "targets": target_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_text(
        sample_dir / "metadata.json",
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(success, "done\n")

    del predictions, predicted_c2w, points, colors, aligned_points
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    args = parse_args()
    if args.width != 2 * args.height:
        raise ValueError(f"Expected a 2:1 ERP, got {args.width}x{args.height}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.degenerate_baseline_threshold <= 0:
        raise ValueError("degenerate-baseline-threshold must be positive")
    if args.only_degenerate_baselines and args.fallback_scale_calibration is None:
        raise ValueError(
            "--only-degenerate-baselines requires --fallback-scale-calibration"
        )
    for required in (args.benchmark / "manifest.json", args.config, args.checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    fallback_scale = None
    calibration = None
    if args.fallback_scale_calibration is not None:
        if not args.fallback_scale_calibration.is_file():
            raise FileNotFoundError(args.fallback_scale_calibration)
        calibration = json.loads(args.fallback_scale_calibration.read_text())
        calibrated_threshold = float(calibration["degenerate_baseline_threshold"])
        if calibrated_threshold != args.degenerate_baseline_threshold:
            raise ValueError(
                "Calibration threshold does not match --degenerate-baseline-threshold"
            )
        fallback_scale = float(calibration["global_median_scale"])
        if not np.isfinite(fallback_scale) or fallback_scale <= 0:
            raise ValueError(f"Invalid calibrated fallback scale: {fallback_scale}")

    manifest_path = args.benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sets = validate_benchmark(manifest, args.height, args.width)
    if args.max_sets >= 0:
        sets = sets[: args.max_sets]
    indexed_sets = list(enumerate(sets))
    if args.only_degenerate_baselines:
        indexed_sets = [
            (index, item)
            for index, item in indexed_sets
            if reference_baseline(item) < args.degenerate_baseline_threshold
        ]
    assigned = [
        (index, item)
        for index, item in indexed_sets
        if index % args.num_shards == args.shard_index
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "program": str(Path(__file__).resolve()),
        "benchmark_manifest": str(manifest_path.resolve()),
        "panovggt_config": str(args.config.resolve()),
        "panovggt_checkpoint": str(args.checkpoint.resolve()),
        "panovggt_checkpoint_bytes": args.checkpoint.stat().st_size,
        "protocol": "two reference RGBs -> PanoVGGT cloud -> three known target poses",
        "target_rgb_accessed": False,
        "render_type": "raw point splat; no hole filling",
        "resolution": [args.height, args.width],
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_sets": len(assigned),
        "only_degenerate_baselines": args.only_degenerate_baselines,
        "degenerate_baseline_threshold": args.degenerate_baseline_threshold,
        "fallback_scale_calibration": (
            str(args.fallback_scale_calibration.resolve())
            if args.fallback_scale_calibration is not None
            else None
        ),
        "fallback_scale": fallback_scale,
    }
    file_prefix = "repair_degenerate_" if args.only_degenerate_baselines else ""
    atomic_write_text(
        args.output
        / f"{file_prefix}configuration_shard_{args.shard_index:02d}.json",
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
    )

    pending = [
        (index, item)
        for index, item in assigned
        if args.overwrite
        or not completed(
            args.output / "samples" / item["id"],
            args.width,
            args.height,
            len(item["targets"]),
        )
    ]
    print(
        json.dumps(
            {
                "shard": f"{args.shard_index}/{args.num_shards}",
                "assigned": len(assigned),
                "already_complete": len(assigned) - len(pending),
                "pending": len(pending),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        atomic_write_text(
            args.output
            / f"_SUCCESS.{file_prefix}shard{args.shard_index:02d}",
            "done\n",
        )
        return

    predictor = PanoVGGTPointPredictor(
        args.config, args.checkpoint, device=args.device
    )
    progress_path = (
        args.output
        / f"{file_prefix}progress_shard_{args.shard_index:02d}.jsonl"
    )
    with progress_path.open("a") as stream:
        for ordinal, (benchmark_index, item) in enumerate(pending, start=1):
            record = process_set(
                item,
                predictor,
                args.output,
                args.height,
                args.width,
                args.degenerate_baseline_threshold,
                fallback_scale,
            )
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "benchmark_index": benchmark_index,
                        "set_id": item["id"],
                        "progress": f"{ordinal}/{len(pending)}",
                        "coverage": [
                            target["raw_pixel_coverage"]
                            for target in record["targets"]
                        ],
                        "seconds": record["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    atomic_write_text(
        args.output / f"_SUCCESS.{file_prefix}shard{args.shard_index:02d}",
        "done\n",
    )


if __name__ == "__main__":
    main()
