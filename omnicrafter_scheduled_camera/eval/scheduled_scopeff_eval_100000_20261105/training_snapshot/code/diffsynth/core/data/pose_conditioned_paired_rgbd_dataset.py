"""Five-slot panorama RGB-D cleanup/NVS dataset.

Each physical camera occupies one temporal slot.  RGB and radial depth remain
separate rasters until the frozen VAE has encoded them; the pipeline then
concatenates their 16-channel latents into one 32-channel diffusion token.

The external observations are two clean RGB/depth references and three noisy
RGB query renders.  Query depth is deliberately not provided as conditioning.
Sky/operator exclusion masks are applied to all matching RGB/depth rasters and
are also returned for loss masking.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from .pose_conditioned_render_query_dataset import (
    WanPanoramaPoseConditionedRenderQueryDataset,
)
from .pose_conditioned_rgbd_query_dataset import (
    encode_log_radial_depth,
    horizontal_wrap_nearest_fill,
    unpack_bit_mask,
)
from .pose_conditioned_rgbd_reference_depth_dataset import apply_validity_mask


class WanPanoramaPoseConditionedPairedRGBDDataset(
    WanPanoramaPoseConditionedRenderQueryDataset
):
    """Return five physical slots with paired RGB/depth targets."""

    def __init__(
        self,
        *args,
        clean_target_root,
        semantic_exclusion_root,
        minimum_normalized_range=1.0 / 16.0,
        maximum_normalized_range=2048.0,
        **kwargs,
    ):
        if semantic_exclusion_root is None:
            raise ValueError("The paired RGB-D dataset requires semantic masks")
        self.clean_target_root = Path(clean_target_root)
        self.semantic_exclusion_root = Path(semantic_exclusion_root)
        self.minimum_normalized_range = float(minimum_normalized_range)
        self.maximum_normalized_range = float(maximum_normalized_range)
        super().__init__(*args, **kwargs)

        for sample in self.samples:
            if len(sample["ref_indices"]) != 2 or len(sample["target_indices"]) != 3:
                raise ValueError(
                    "Five-slot paired RGB-D requires exactly two references and "
                    f"three queries; {sample['sample_id']} has "
                    f"{len(sample['ref_indices'])} and {len(sample['target_indices'])}"
                )
            for frame_index in sample["ref_indices"] + sample["target_indices"]:
                depth_path = self._clean_depth_path(sample["scene_name"], frame_index)
                mask_path = self._semantic_mask_path(sample["scene_name"], frame_index)
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing clean depth pseudo-target: {depth_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing semantic exclusion mask: {mask_path}")

    def _clean_depth_path(self, scene_name, frame_index):
        return self.clean_target_root / scene_name / "depth" / f"{frame_index:06d}.npz"

    def _semantic_mask_path(self, scene_name, frame_index):
        return (
            self.semantic_exclusion_root
            / scene_name
            / "combined"
            / f"{frame_index:06d}.png"
        )

    def _load_semantic_validity(self, scene_name, frame_indices):
        views = []
        for frame_index in frame_indices:
            path = self._semantic_mask_path(scene_name, frame_index)
            exclusion = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if exclusion is None:
                raise FileNotFoundError(path)
            if exclusion.shape != (self.height, self.width):
                exclusion = cv2.resize(
                    exclusion,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )
            views.append(torch.from_numpy(exclusion <= 127))
        return torch.stack(views)

    def _load_depth_frames(
        self,
        scene_name,
        frame_indices,
        pose_scale,
        semantic_validity,
    ):
        encoded_views = []
        valid_views = []
        clipping = []
        for frame_index, semantic_valid in zip(frame_indices, semantic_validity):
            path = self._clean_depth_path(scene_name, frame_index)
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["scene_name"]) != scene_name:
                    raise ValueError(f"Clean depth scene mismatch in {path}")
                if int(archive["frame_index"]) != frame_index:
                    raise ValueError(f"Clean depth frame mismatch in {path}")
                model_hw = tuple(int(value) for value in archive["model_hw"])
                if model_hw != (self.height, self.width):
                    raise ValueError(
                        f"Clean depth {path} is {model_hw}, expected "
                        f"{(self.height, self.width)}"
                    )
                depth = archive["radial_depth_model"].astype(np.float32)
                valid_shape = tuple(
                    int(value) for value in archive["model_valid_mask_shape"]
                )
                valid = unpack_bit_mask(
                    archive["model_valid_mask_packed"],
                    valid_shape,
                    str(archive["mask_bitorder"]),
                )

            valid &= semantic_valid.cpu().numpy().astype(bool)
            encoded, stats = encode_log_radial_depth(
                depth,
                valid,
                pose_scale,
                self.minimum_normalized_range,
                self.maximum_normalized_range,
            )
            encoded = horizontal_wrap_nearest_fill(encoded, valid)
            valid_tensor = torch.from_numpy(valid)
            raster = torch.from_numpy(encoded).repeat(3, 1, 1)
            encoded_views.append(apply_validity_mask(raster, valid_tensor))
            valid_views.append(valid_tensor)
            clipping.append(stats)
        return encoded_views, torch.stack(valid_views), clipping

    @staticmethod
    def _rgb_tensor(image):
        return TF.pil_to_tensor(image).float().div_(127.5).sub_(1.0)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        sample = self.samples[index % len(self.samples)]
        num_references = len(sample["ref_indices"])
        num_queries = len(sample["target_indices"])
        frame_indices = sample["ref_indices"] + sample["target_indices"]

        semantic_validity = self._load_semantic_validity(
            sample["scene_name"], frame_indices
        )
        reference_validity = semantic_validity[:num_references]
        query_validity = semantic_validity[num_references:]

        reference_rgb = [self._rgb_tensor(image) for image in item["input_images"]]
        query_rgb_conditioning = [
            self._rgb_tensor(image) for image in item["conditioning_images"]
        ]
        target_rgb = [self._rgb_tensor(image) for image in item["target_images"]]

        reference_rgb = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(reference_rgb, reference_validity)
        ]
        query_rgb_conditioning = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(query_rgb_conditioning, query_validity)
        ]
        target_rgb = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(target_rgb, semantic_validity)
        ]

        target_depth, depth_validity, depth_clipping = self._load_depth_frames(
            sample["scene_name"],
            frame_indices,
            float(item["pose_scale"]),
            semantic_validity,
        )
        reference_depth = target_depth[:num_references]
        reference_depth_validity = depth_validity[:num_references]
        output_depth_validity = depth_validity[num_references:]

        task_types = [
            "nvs" if ratio >= 1.0 else "cleanup"
            for ratio in item["query_pixel_dropout_ratios"]
        ]
        return {
            "reference_rgb": reference_rgb,
            "reference_depth": reference_depth,
            "query_rgb_conditioning": query_rgb_conditioning,
            "target_rgb_rasters": target_rgb,
            "target_depth_rasters": target_depth,
            "reference_rgb_validity": reference_validity.float(),
            "reference_depth_validity": reference_depth_validity.float(),
            "output_rgb_validity": query_validity.float(),
            "output_depth_validity": output_depth_validity.float(),
            "raymap": item["raymap"],
            "prompt": item["prompt"],
            "sample_id": item["sample_id"],
            "scene_name": item["scene_name"],
            "slot_roles": ["reference"] * num_references + ["query"] * num_queries,
            "slot_frame_indices": frame_indices,
            "query_pixel_dropout_ratios": item["query_pixel_dropout_ratios"],
            "target_task_types": task_types,
            "pose_scale": item["pose_scale"],
            "depth_clipping": depth_clipping,
            "depth_encoding": {
                "representation": "normalized_log_radial_range",
                "minimum_normalized_range": self.minimum_normalized_range,
                "maximum_normalized_range": self.maximum_normalized_range,
            },
            "depth_sources": [
                str(self._clean_depth_path(sample["scene_name"], frame_index))
                for frame_index in frame_indices
            ],
            "semantic_exclusion_sources": [
                str(self._semantic_mask_path(sample["scene_name"], frame_index))
                for frame_index in frame_indices
            ],
        }
