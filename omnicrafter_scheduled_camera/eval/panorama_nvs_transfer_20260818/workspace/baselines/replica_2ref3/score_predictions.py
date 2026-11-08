#!/usr/bin/env python3
"""Compute one consistent metric suite for all Replica baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=(
            "panogrf",
            "pansplat",
            "omnisplat",
            "pfgs360",
            "cylindersplat",
            "splatter360",
            "panovggt",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="replica_2ref3")
    parser.add_argument("--max-sets", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Scored height; defaults to the benchmark manifest resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Scored width; defaults to the benchmark manifest resolution.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def prediction_candidates(
    root: Path, layout: str, experiment: str, item: dict, target: dict
) -> list[Path]:
    set_id = item["id"]
    target_number = target["target_number"]
    if layout == "panogrf":
        return [root / "color" / set_id / f"{target_number:06d}.png"]
    if layout == "pansplat":
        relative = (
            Path("replica_val_0.5")
            / target["triplet_name"]
            / "color"
            / "000001.png"
        )
        return [root / "test" / relative, root / relative] + [
            root / f"shard_{shard}" / "test" / relative for shard in range(16)
        ]
    if layout == "omnisplat":
        relative = Path(set_id) / "color" / f"{target['frame_index']}.png"
        return [root / "ReplicaVal" / relative, root / relative] + [
            root / f"shard_{shard}" / "ReplicaVal" / relative
            for shard in range(16)
        ]
    if layout == "pfgs360":
        return [
            root
            / experiment
            / set_id
            / "pfgs360"
            / "results"
            / f"{target_number:06d}-pred.png"
        ]
    if layout == "cylindersplat":
        return [root / "color" / set_id / f"{target_number:06d}.png"]
    if layout == "splatter360":
        relative = (
            Path("ReplicaVal")
            / "render"
            / set_id
            / "color"
            / f"{target_number:06d}_erp.png"
        )
        return [root / relative] + [
            root / f"shard_{shard}" / relative for shard in range(16)
        ]
    if layout == "panovggt":
        return [
            root
            / "samples"
            / set_id
            / f"prior_{target_number}.png"
        ]
    raise ValueError(layout)


def resolve_prediction(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_rgb(
    path: Path,
    expected_size: tuple[int, int],
    *,
    resize: bool = False,
) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != expected_size:
            if not resize:
                raise ValueError(
                    f"Resolution mismatch for {path}: {image.size}, expected {expected_size}"
                )
            image = image.resize(expected_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    squared_error = (prediction - target) ** 2
    mse = float(squared_error.mean())
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    ssim = float(
        structural_similarity(
            target.transpose(2, 0, 1),
            prediction.transpose(2, 0, 1),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
    )
    height, width = target.shape[:2]
    latitude_weight = np.sin(
        (np.arange(height, dtype=np.float64) + 0.5) * np.pi / height
    )
    weighted_mse = float(
        np.sum(squared_error * latitude_weight[:, None, None])
        / (np.sum(latitude_weight) * width * 3)
    )
    ws_psnr = float(-10.0 * np.log10(max(weighted_mse, 1e-12)))
    return {"psnr": psnr, "ssim": ssim, "ws_psnr": ws_psnr}


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    sets = manifest["sets"]
    if args.max_sets >= 0:
        sets = sets[: args.max_sets]
    full_expected = sum(len(item["targets"]) for item in sets)
    sets = [
        item
        for index, item in enumerate(sets)
        if index % args.num_shards == args.shard_index
    ]
    native_height = int(manifest["resolution"]["height"])
    native_width = int(manifest["resolution"]["width"])
    height = native_height if args.height is None else int(args.height)
    width = native_width if args.width is None else int(args.width)
    if height <= 0 or width <= 0:
        raise ValueError("Scored height and width must be positive")

    entries, missing = [], []
    for item in sets:
        for target in item["targets"]:
            candidates = prediction_candidates(
                args.predictions, args.layout, args.experiment, item, target
            )
            prediction_path = resolve_prediction(candidates)
            if prediction_path is None:
                missing.append(
                    {
                        "set_id": item["id"],
                        "target_number": target["target_number"],
                        "candidates": [str(path) for path in candidates],
                    }
                )
                continue
            entries.append((item, target, prediction_path))

    expected = sum(len(item["targets"]) for item in sets)
    if missing and not args.allow_missing:
        preview = "\n".join(record["candidates"][0] for record in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)}/{expected} predictions; first paths:\n{preview}"
        )
    if not entries:
        raise ValueError("No predictions found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    records = []
    for start in range(0, len(entries), args.batch_size):
        chunk = entries[start : start + args.batch_size]
        predictions, targets = [], []
        for item, target, prediction_path in chunk:
            prediction = load_rgb(prediction_path, (width, height))
            target_image = load_rgb(
                Path(target["image"]),
                (width, height),
                resize=(width, height) != (native_width, native_height),
            )
            metrics = scalar_metrics(prediction, target_image)
            records.append(
                {
                    "set_id": item["id"],
                    "source_sample": item["source_sample"],
                    "target_number": target["target_number"],
                    "frame_index": target["frame_index"],
                    "prediction": str(prediction_path),
                    "target": target["image"],
                    **metrics,
                }
            )
            predictions.append(prediction.transpose(2, 0, 1))
            targets.append(target_image.transpose(2, 0, 1))
        prediction_tensor = torch.from_numpy(np.stack(predictions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        with torch.inference_mode():
            scores = (
                lpips_model(target_tensor, prediction_tensor, normalize=True)
                .reshape(-1)
                .cpu()
                .numpy()
            )
        for record, score in zip(records[-len(chunk) :], scores):
            record["lpips"] = float(score)
        print(f"Scored {min(start + len(chunk), len(entries))}/{len(entries)}", flush=True)

    metrics = {
        key: float(np.mean([record[key] for record in records]))
        for key in ("psnr", "lpips", "ssim", "ws_psnr")
    }
    result = {
        "layout": args.layout,
        "prediction_root": str(args.predictions),
        "resolution": [height, width],
        "benchmark_native_resolution": [native_height, native_width],
        "target_resize": (
            "none"
            if (height, width) == (native_height, native_width)
            else "PIL Lanczos from benchmark native resolution"
        ),
        "metric_definitions": {
            "psnr": "per-image RGB PSNR, then arithmetic mean",
            "lpips": "LPIPS VGG, normalize=True, then arithmetic mean",
            "ssim": "skimage SSIM, 11x11 Gaussian window, then arithmetic mean",
            "ws_psnr": "sin(latitude)-weighted RGB PSNR, then arithmetic mean",
        },
        "expected_targets": expected,
        "full_expected_targets": full_expected,
        "scored_targets": len(records),
        "missing_targets": len(missing),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "metrics": metrics,
        "records": records,
        "missing": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"coverage": f"{len(records)}/{expected}", **metrics}, indent=2))


if __name__ == "__main__":
    main()
