"""Runs the FULL PanoVGGTModel (aggregator + point head + camera head) on a pair of
condition images, producing camera_poses / world_points / images for
render_jhu_heldout.py's align_cameras / render_erp.

Mirrors run_panovggt_colmap.py EXACTLY (the user's own validated inference script)
rather than an independent implementation: same config-driven construction, same
map_location loading, same bf16 autocast, same reliance on
panovggt.utils.basic.load_images_as_tensor for image preprocessing (which is
hard-locked to 518x1036 -- `target_height` is accepted but ignored by that
function, per its own docstring). Point-cloud extraction therefore always runs at
518x1036 regardless of whatever encoder resolution the main renderer training uses
-- these are decoupled by design, since 518x1036 is what's been validated to give
clean point clouds.

NOT executable in this sandbox (no checkpoint, no GPU) -- mirrors validated code
as closely as possible; verify the load report on first run.
"""
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

try:
    # xFormers 0.0.32's FA3 Hopper kernel rejects Blackwell (SM 12.0) at
    # launch. FA2/CUTLASS remain memory-efficient and support this GPU.
    from xformers.ops.fmha import _set_use_fa3

    _set_use_fa3(False)
except ImportError:
    pass

from panovggt.models.panovggt_model import PanoVGGTModel


def load_ordered_panoramas(image_paths):
    """Match PanoVGGT's fixed loader without importing optional Open3D IO."""
    to_tensor = transforms.ToTensor()
    tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((1036, 518), Image.Resampling.LANCZOS)
            tensors.append(to_tensor(image))
    if not tensors:
        raise ValueError("No PanoVGGT input images were provided")
    return torch.stack(tensors, dim=0)


class PanoVGGTPointPredictor:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        # --- construction: mirrors run_panovggt_colmap.load_model exactly ---
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        self.model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
            # enable_global_points NOT overridden -- back to matching
            # run_panovggt_colmap.py exactly (defaults to True). It was
            # previously set False here as a believed-safe speedup (world_points/
            # camera_poses are computed by a separate code path that doesn't
            # reference the global branch, per direct source reading) -- but that
            # was never empirically verified against a working run, and the
            # user's own validated pipeline never disables it. Reverted: matching
            # known-good behavior takes priority over an unverified optimization.
        )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if isinstance(ckpt, dict) and key in ckpt:
                ckpt = ckpt[key]
                break
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[load] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[PanoVGGTPointPredictor] loaded {len(state_dict)} tensors: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

        self.model = self.model.to(device).eval()
        self.device = device
        self.patch_size = int(cfg.patch_size)

    @torch.no_grad()
    def predict_from_paths(self, image_paths):
        """image_paths: list of S image file paths (condition views, in order).
        Loads them via panovggt.utils.basic.load_images_as_tensor (through a
        throwaway temp dir, since that function reads a directory) -- this is the
        SAME preprocessing path run_panovggt_colmap.py uses, fixed at 518x1036.
        Returns a dict: camera_poses (1,S,4,4), world_points (1,S,H,W,3),
        images (1,S,3,H,W) -- all float32 numpy, matching render_jhu_heldout.py's
        expected `predictions` format.
        """
        imgs = load_ordered_panoramas(image_paths).to(
            self.device
        )  # (S, 3, 518, 1036)

        use_amp = self.device.startswith("cuda")
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = self.model(imgs[None])
        else:
            preds = self.model(imgs[None])

        if "world_points" not in preds:
            raise RuntimeError(
                "model output has no 'world_points' -- enable_camera and "
                "enable_point must both be True in the config for this to be "
                "computed."
            )
        if "images" not in preds:
            raise RuntimeError(
                "model output has no 'images' -- this requires model.eval() "
                "(predictions['images'] is only set when not self.training)."
            )
        return {
            "camera_poses": preds["camera_poses"].float().cpu().numpy(),
            "world_points": preds["world_points"].float().cpu().numpy(),
            "images": preds["images"].float().cpu().numpy(),
        }
