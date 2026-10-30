#!/usr/bin/env python3
"""Calibrate PanoVGGT scale for co-located Replica reference cameras.

Two cameras at the same center provide no baseline from which to recover the
scale of PanoVGGT's normalized point cloud.  This program takes the robust
global median of the scale ratios from all non-degenerate *reference pairs*.
It reads alignment metadata and camera poses only; no target RGB is accessed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degenerate-baseline-threshold", type=float, default=1e-4)
    args = parser.parse_args()
    if args.degenerate_baseline_threshold <= 0:
        raise ValueError("degenerate-baseline-threshold must be positive")

    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    sets = manifest["sets"]
    if not sets:
        raise ValueError("Benchmark contains no sets")

    scales = []
    degenerate_ids = []
    for index, item in enumerate(sets):
        if item["id"] != f"sample_{index:06d}":
            raise ValueError(f"Unexpected set id at {index}: {item['id']}")
        poses = np.asarray([v["w2c"] for v in item["references"]], dtype=np.float64)
        centers = np.linalg.inv(poses)[:, :3, 3]
        if np.linalg.norm(centers[1] - centers[0]) < args.degenerate_baseline_threshold:
            degenerate_ids.append(item["id"])
            continue
        metadata_path = args.priors / "samples" / item["id"] / "metadata.json"
        success_path = args.priors / "samples" / item["id"] / "_SUCCESS"
        if not metadata_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(f"Incomplete PanoVGGT prior: {item['id']}")
        metadata = json.loads(metadata_path.read_text())
        alignment = metadata["alignment"]
        gt_baseline = float(alignment["gt_baseline"])
        scale = float(alignment["scale"])
        if gt_baseline < args.degenerate_baseline_threshold:
            degenerate_ids.append(item["id"])
        elif math.isfinite(scale) and scale > 0:
            scales.append(scale)
        else:
            raise ValueError(f"Invalid non-degenerate scale for {item['id']}: {scale}")

    if not scales:
        raise ValueError("No usable non-degenerate reference-pair scales")
    values = np.asarray(scales, dtype=np.float64)
    result = {
        "method": "global median of PanoVGGT-to-GT scale ratios from non-degenerate two-reference baselines",
        "benchmark_manifest": str((args.benchmark / "manifest.json").resolve()),
        "prior_root": str(args.priors.resolve()),
        "target_rgb_accessed": False,
        "degenerate_baseline_threshold": args.degenerate_baseline_threshold,
        "total_sets": len(sets),
        "non_degenerate_scale_count": len(scales),
        "degenerate_set_count": len(degenerate_ids),
        "degenerate_set_ids": degenerate_ids,
        "global_median_scale": float(np.median(values)),
        "scale_quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
