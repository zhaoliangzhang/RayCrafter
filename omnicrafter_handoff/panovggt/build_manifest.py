#!/usr/bin/env python3
"""Select an explicit reference pair and target frames from a posed sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_priors import atomic_write_text, validate_benchmark


def build(sequence_path: Path, references: list[int], targets: list[int], height: int, width: int) -> dict:
    sequence = json.loads(sequence_path.read_text())
    if sequence.get("pose_convention") != "OpenCV world-to-camera":
        raise ValueError("Convert poses to OpenCV world-to-camera first")
    if len(references) != 2 or len(set(references)) != 2:
        raise ValueError("Choose exactly two distinct reference frame indices")
    if not targets or len(set(targets)) != len(targets) or set(references) & set(targets):
        raise ValueError("Targets must be unique and disjoint from the references")
    if height <= 0 or width != 2 * height:
        raise ValueError("Output must have a positive 2:1 ERP resolution")
    frames = sequence["frames"]
    by_index = {v["frame_index"]: v for v in frames}
    if len(by_index) != len(frames):
        raise ValueError("Duplicate frame indices")

    def view(index: int) -> dict:
        if index not in by_index:
            raise ValueError(f"Unknown frame_index: {index}")
        record = dict(by_index[index])
        if record.get("image"):
            path = Path(record["image"]).expanduser()
            record["image"] = str(path.resolve() if path.is_absolute()
                                  else (sequence_path.parent / path).resolve())
        return record

    scene = sequence.get("scene_name", "scene")
    manifest = {
        "pose_convention": "OpenCV world-to-camera",
        "resolution": {"height": height, "width": width},
        "logical_sets": 1,
        "target_images": len(targets),
        "sets": [{
            "id": "sample_000000", "scene_name": scene, "source_sample": scene,
            "references": [view(i) for i in references],
            "targets": [dict(view(i), target_number=n) for n, i in enumerate(targets)],
        }],
    }
    validate_benchmark(manifest, height, width)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--references", type=int, nargs=2, required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    result = build(args.sequence.resolve(), args.references, args.targets, args.height, args.width)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}; choose a new output directory")
    atomic_write_text(destination, json.dumps(result, indent=2) + "\n")
    print(destination.resolve())


if __name__ == "__main__":
    main()
