"""Portable paths and strict input validation; no model downloads on import."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_resolution(height, width):
    if height < 16 or width != 2 * height or height % 16:
        raise ValueError("Require a 2:1 ERP with H and W divisible by 16 (e.g. 256x512)")


def read_manifest(path, task=None):
    import numpy as np
    from PIL import Image
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("pose_convention") != "w2c_opencv":
        raise ValueError("Set pose_convention to w2c_opencv; do not supply c2w/OpenGL poses")
    sets = data["sets"]
    if not sets:
        raise ValueError("Empty evaluation manifest")
    seen = set()
    for item in sets:
        ident = item["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("Set IDs must contain only letters, digits, '_' or '-'")
        if ident in seen:
            raise ValueError(f"Duplicate set ID: {ident}")
        seen.add(ident)
        if len(item["references"]) != 2 or not item["targets"]:
            raise ValueError(f"{ident}: require exactly two references and at least one target")
        for view in item["references"] + item["targets"]:
            pose = np.asarray(view["w2c"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"{ident}: invalid 4x4 pose")
            if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
                raise ValueError(f"{ident}: invalid homogeneous bottom row")
            rotation = pose[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
                raise ValueError(f"{ident}: pose rotation must be a proper rigid rotation")
            for key in ("image", "noisy_image"):
                if key in view:
                    p = Path(view[key])
                    view[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
        # GT paths are deliberately not opened/required for inference.
        required = [v["image"] for v in item["references"]]
        if task == "cleanup":
            required += [v["noisy_image"] for v in item["targets"]]
        for image_path in required:
            with Image.open(image_path) as image:
                if image.width != 2 * image.height:
                    raise ValueError(f"Not a full 2:1 ERP: {image_path}")
                image.verify()
    return data


def wan_assets(root):
    root = Path(root).resolve()
    model = root / "Wan-AI/Wan2.1-I2V-14B-480P"
    shards = sorted(model.glob("diffusion_pytorch_model-*-of-00007.safetensors"))
    if len(shards) != 7:
        raise FileNotFoundError(f"Expected all seven Wan2.1 DiT shards in {model}")
    paths = [[str(p) for p in shards]]
    converted = root / "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
    for name in ("models_t5_umt5-xxl-enc-bf16", "Wan2.1_VAE", "models_clip_open-clip-xlm-roberta-large-vit-huge-14"):
        candidates = [converted / f"{name}.safetensors", model / f"{name}.pth"]
        candidate = next((p for p in candidates if p.is_file()), None)
        if candidate is None:
            raise FileNotFoundError(f"Missing {name}; tried {candidates}")
        paths.append(str(candidate))
    tokenizer = root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Missing complete tokenizer directory: {tokenizer}")
    return paths, str(tokenizer)
