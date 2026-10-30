#!/usr/bin/env python3
"""Full-image metrics at prediction resolution. Learned metrics are opt-in."""
import argparse
import importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
from common import check_resolution, read_manifest, write_json


def scalar_metrics(prediction, target):
    # Same scalar protocol as the in-house replica_2ref3/score_predictions.py.
    from skimage.metrics import structural_similarity
    error = (prediction - target) ** 2
    psnr = float(-10.0 * np.log10(max(float(error.mean()), 1e-12)))
    ssim = float(structural_similarity(target.transpose(2, 0, 1), prediction.transpose(2, 0, 1),
        win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0))
    h, w = target.shape[:2]
    weights = np.sin((np.arange(h, dtype=np.float64) + 0.5) * np.pi / h)
    wmse = float((error * weights[:, None, None]).sum() / (weights.sum() * w * 3))
    return {"psnr": psnr, "ssim": ssim, "ws_psnr": float(-10.0 * np.log10(max(wmse, 1e-12)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--learned", action="store_true", help="Also compute VGG-LPIPS and ensemble DreamSim; may download metric weights")
    p.add_argument("--device", default="cuda")
    p.add_argument("--metric-cache", default=".cache/dreamsim")
    args = p.parse_args()
    check_resolution(args.height, args.width)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_manifest(args.manifest)
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "scikit-image")}
    if args.learned:
        import torch
        import lpips
        from dreamsim import dreamsim
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        perceptual = lpips.LPIPS(net="vgg").to(args.device).eval().requires_grad_(False)
        dream, preprocess = dreamsim(pretrained=True, device=args.device, cache_dir=args.metric_cache, dreamsim_type="ensemble")
        dream.eval().requires_grad_(False)
        versions.update({n: importlib.metadata.version(n) for n in ("torch", "torchvision", "lpips", "dreamsim", "timm", "peft", "transformers")})
    rows = []
    for item in manifest["sets"]:
        for q, view in enumerate(item["targets"]):
            path = args.predictions / "color" / item["id"] / f"{q:06d}.png"
            with Image.open(path) as im:
                pred = im.convert("RGB")
                if pred.size != (args.width, args.height):
                    raise ValueError(f"Wrong prediction size: {path}")
            with Image.open(view["image"]) as im:
                gt = im.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
            arrays = [np.asarray(im, dtype=np.float32) / 255 for im in (pred, gt)]
            scores = scalar_metrics(*arrays)
            if args.learned:
                tensors = [torch.from_numpy(a.transpose(2, 0, 1).copy())[None].to(args.device) for a in arrays]
                with torch.inference_mode():
                    scores["lpips"] = float(perceptual(*tensors, normalize=True).item())
                    embeddings = dream.embed(torch.cat([preprocess(im) for im in (pred, gt)]).to(args.device))
                    scores["dreamsim"] = float((1 - torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2], dim=-1)).item())
            rows.append({"set_id": item["id"], "query": q, **scores})
    summary = {metric: float(np.mean([r[metric] for r in rows])) for metric in scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"mean_per_target": summary, "num_targets": len(rows), "records": rows,
        "protocol": {"height": args.height, "width": args.width, "full_image": True, "gt_resize": "Lanczos",
                     "lpips": "vgg" if args.learned else None, "dreamsim": "ensemble cosine distance" if args.learned else None,
                     "versions": versions}})
    print(summary)


if __name__ == "__main__":
    main()
