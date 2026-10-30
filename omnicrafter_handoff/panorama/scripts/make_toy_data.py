#!/usr/bin/env python3
"""Generate synthetic schema/test fixtures, NOT meaningful training/evaluation data."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from common import write_json


def create(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    (root / "noisy").mkdir()
    (root / "poses").mkdir()
    rng = np.random.default_rng(42)
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    w2c[:, 0, 3] = -np.linspace(0, 1, 5)
    for frame in range(5):
        pixels = rng.integers(32, 256, (32, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{frame:06d}.png")
        noisy = pixels.copy()
        noisy[rng.random((32, 64)) < 0.3] = 0
        Image.fromarray(noisy).save(root / "noisy" / f"{frame:06d}.png")
    np.savez(root / "poses/scene.npz", w2c_poses=w2c)
    write_json(root / "poses/manifest.json", {"pose_convention": "w2c_opencv", "scenes": {
        "demo_scene": {"pose_path": "poses/scene.npz", "num_frames": 5}}})
    refs = ["images/000000.png", "images/000004.png"]
    rows = [{"source_sample": "demo_set", "scene_name": "demo_scene", "target_number": q,
             "refs": refs, "src": f"noisy/{frame:06d}.png", "target": f"images/{frame:06d}.png",
             "frame_index": frame} for q, frame in enumerate((1, 2, 3))]
    import json
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(root / "eval.json", {"pose_convention": "w2c_opencv", "sets": [{"id": "demo_set",
        "references": [{"image": refs[q], "w2c": w2c[frame].tolist()} for q, frame in enumerate((0, 4))],
        "targets": [{"image": f"images/{frame:06d}.png", "noisy_image": f"noisy/{frame:06d}.png",
                     "w2c": w2c[frame].tolist()} for frame in (1, 2, 3)]}]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)
