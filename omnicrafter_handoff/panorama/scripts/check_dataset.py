#!/usr/bin/env python3
"""CPU-only dataloader check; no backbone or model weights loaded."""
import argparse
from common import check_resolution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--pose-manifest", required=True)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--queries", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=8, help="Number of complete sets to load; 0 checks all")
    args = p.parse_args()
    check_resolution(args.height, args.width)
    if args.max_samples < 0 or args.queries < 1:
        p.error("max-samples must be nonnegative and queries positive")
    from diffsynth.core.data.pose_conditioned_render_query_dataset import WanPanoramaPoseConditionedRenderQueryDataset
    dataset = WanPanoramaPoseConditionedRenderQueryDataset(
        base_path=args.data_root, metadata_path=args.train_manifest, pose_manifest_path=args.pose_manifest,
        height=args.height, width=args.width, max_query_slots=args.queries,
        stream_metadata=True, deterministic_degradation=True, nvs_ratio=0.5, max_pixel_dropout=0.99)
    limit = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    for i in range(limit):
        sample = dataset[i]
        if len(sample["input_images"]) != 2 or len(sample["conditioning_images"]) != args.queries:
            raise ValueError(f"{sample['sample_id']}: expected 2 references and {args.queries} queries")
        expected = (2 + args.queries, 384, args.height // 8, args.width // 8)
        if tuple(sample["raymap"].shape) != expected:
            raise ValueError(f"Incorrect raymap shape: {sample['raymap'].shape}")
    print(f"PASS: loaded {limit}/{len(dataset)} sets; 2ref{args.queries}; {args.height}x{args.width}")


if __name__ == "__main__":
    main()
