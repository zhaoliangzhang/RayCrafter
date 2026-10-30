#!/usr/bin/env python3
"""Create a source-only ZIP and checksums; optionally verify original copied files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    allowed = {".py", ".md", ".txt", ".toml", ".json", ".jsonl", ".yaml"}
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Do not distribute external symlinks: {path}")
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name not in {"LICENSE", ".gitignore", "SHA256SUMS"} and path.suffix not in allowed:
            raise ValueError(f"Unexpected non-source file: {path}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpected large payload: {path}")
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, help="Optional verification against the production snapshot")
    parser.add_argument("--output", type=Path, help="Defaults to a sibling ZIP; refuses overwrite")
    args = parser.parse_args()
    if args.source_workspace:
        original = args.source_workspace / "third_party/FrameCrafter"
        records = []
        for path in sorted((ROOT / "code").rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT / "code")
            source_relative = Path("README.md") if str(relative) == "README_FrameCrafter.md" else relative
            source = original / source_relative
            if path.read_bytes() != source.read_bytes():
                raise ValueError(f"Copied source differs: {relative}")
            records.append({"package_path": str(path.relative_to(ROOT)),
                "source_relative_to_workspace": str(source.relative_to(args.source_workspace)), "sha256": digest(path)})
        aliases = {
            "configs/accelerate_zero2_gpu.yaml": "third_party/FrameCrafter/model_training/io105_original_zero2_gpuopt_bf16.yaml",
            "provenance/run_650k_original.sh.txt": "scripts/run_omnicrafter_q3_250k_hm3d_jhu_balanced_io105.sh",
            "provenance/run_continuation_original.sh.txt": "scripts/run_omnicrafter_hm3d_jhu_q3_100k_continue_io105.sh",
            "provenance/run_trainer_original.sh.txt": "scripts/run_framecrafter_model2_large_io95.sh",
        }
        for destination, relative in aliases.items():
            path, source = ROOT / destination, args.source_workspace / relative
            if path.read_bytes() != source.read_bytes():
                raise ValueError(f"Copied source differs: {destination}")
            records.append({"package_path": destination, "source_relative_to_workspace": relative, "sha256": digest(path)})
        manifest = {"snapshot_date": "2026-10-19", "source_label": "panorama_nvs_transfer_20260818/workspace",
                    "copied_files": records, "all_copied_files_byte_identical": True,
                    "new_files": "Root README, examples, portable scripts, tests, and sharing metadata were added separately."}
        (ROOT / "provenance/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payload = [p for p in files() if p.name != "SHA256SUMS"]
    (ROOT / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(ROOT)}\n" for p in payload))
    output = args.output or ROOT.with_suffix(".zip")
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Archive must be outside the source folder")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files():
            archive.write(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC validation failed")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": digest(output), "files": len(payload) + 1}))


if __name__ == "__main__":
    main()
