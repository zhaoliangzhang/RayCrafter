#!/usr/bin/env python3
"""Run a trained checkpoint on WebDataset shards and save src | pred | gt grids.

    python sample.py --config outputs/<run>/config.yaml \
        --ckpt outputs/<run>/transformer_final.safetensors \
        --mode nvs --num_steps 4 --max_batches 8

mode=refine: the degraded render is the input (noise_alpha=0).
mode=nvs:    the input latent is pure Gaussian noise (noise_alpha=1), i.e.
             novel-view synthesis from refs + cameras only. Needs noise_space=latent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from raycrafter.config import Config
from raycrafter.data import make_dataloader
from raycrafter.model import RayCrafterModel


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    top, left = (x.shape[-2] - h) // 2, (x.shape[-1] - w) // 2
    return x[..., top:top + h, left:left + w]


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy().transpose(1, 2, 0)


@torch.no_grad()
def main(
    config: str,
    ckpt: str,
    shards: Optional[str] = None,
    mode: str = "nvs",
    num_steps: int = 4,
    max_batches: int = 8,
    out_dir: Optional[str] = None,
    seed: int = 0,
) -> None:
    if mode not in ("refine", "nvs"):
        raise ValueError(f"mode must be refine or nvs, got {mode!r}")
    cfg = Config.from_yaml(config)
    if mode == "nvs" and cfg.noise_space != "latent":
        raise ValueError("mode=nvs needs noise_space=latent")
    cfg.resume_from = ckpt
    cfg.gradient_checkpointing = False
    cfg.lpips_loss_weight = 0.0
    device = torch.device("cuda")
    torch.manual_seed(seed)

    model = RayCrafterModel(cfg, device).eval()
    loader = make_dataloader(
        shards or cfg.val_shards, num_views=cfg.num_views, ref_size=cfg.ref_size,
        image_size=tuple(cfg.image_size), batch_size=1, num_workers=1,
        use_pose=cfg.use_pose, shuffle=False, repeat=False, window=cfg.window,
        data_format=cfg.data_format,
    )
    out = Path(out_dir or Path(ckpt).parent / f"samples_{Path(ckpt).stem}_{mode}_n{num_steps}")
    out.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src, tgt, ref = (batch[k].to(device) for k in ("src", "tgt", "ref"))
        cams = {k: batch[k].to(device) for k in ("camera_c2w", "camera_fxfycxcy") if k in batch}
        if cams:
            cams["camera_model"] = batch.get("camera_model", "pinhole")
        alpha = torch.full((src.shape[0],), 1.0 if mode == "nvs" else 0.0)
        pred = model.sample(src, ref, num_steps=num_steps, noise_alpha=alpha, **cams)
        h, w = pred.shape[-2:]
        crop = lambda x: _center_crop(x, h, w)  # VAE input is cropped to a multiple of 16
        for v in range(src.shape[1]):
            shown = torch.zeros_like(src[0, v]) if mode == "nvs" else src[0, v]
            row = np.concatenate([_to_uint8(crop(shown)), _to_uint8(pred[0, v]), _to_uint8(crop(tgt[0, v]))], 1)
            refs = np.concatenate([_to_uint8(crop(r)) for r in ref[0]], 1)
            refs = np.array(Image.fromarray(refs).resize((row.shape[1], row.shape[1] * h // refs.shape[1])))
            Image.fromarray(np.concatenate([row, refs], 0)).save(out / f"{i:04d}_v{v}.png")
        print(f"[{i}] {batch['key'][0]}", flush=True)
    print(f"Saved to {out}  (rows: src | pred | gt, then refs)")


if __name__ == "__main__":
    fire.Fire(main)
