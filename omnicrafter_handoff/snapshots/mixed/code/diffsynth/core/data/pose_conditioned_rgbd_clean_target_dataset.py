"""RGB-D query dataset with independent clean-frame depth pseudo-targets."""

from pathlib import Path

import cv2
import numpy as np
import torch

from .pose_conditioned_rgbd_query_dataset import (
    WanPanoramaPoseConditionedRGBDQueryDataset,
    encode_log_radial_depth,
    horizontal_wrap_nearest_fill,
    unpack_bit_mask,
)


class WanPanoramaPoseConditionedRGBDCleanTargetDataset(
    WanPanoramaPoseConditionedRGBDQueryDataset
):
    """Keep legacy conditioning but replace only depth supervision."""

    def __init__(
        self, *args, clean_target_root, semantic_exclusion_root=None, **kwargs
    ):
        self.clean_target_root = Path(clean_target_root)
        self.semantic_exclusion_root = (
            None
            if semantic_exclusion_root is None
            else Path(semantic_exclusion_root)
        )
        super().__init__(*args, **kwargs)
        for sample in self.samples:
            for frame_index in sample["target_indices"]:
                path = self._clean_depth_path(sample["scene_name"], frame_index)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing clean depth pseudo-target: {path}")
                if self.semantic_exclusion_root is not None:
                    mask_path = self._semantic_mask_path(
                        sample["scene_name"], frame_index
                    )
                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            f"Missing semantic exclusion mask: {mask_path}"
                        )

    def _clean_depth_path(self, scene_name, frame_index):
        return (
            self.clean_target_root
            / scene_name
            / "depth"
            / f"{frame_index:06d}.npz"
        )

    def _semantic_mask_path(self, scene_name, frame_index):
        if self.semantic_exclusion_root is None:
            return None
        return (
            self.semantic_exclusion_root
            / scene_name
            / "combined"
            / f"{frame_index:06d}.png"
        )

    def _load_clean_target_depth(self, sample, pose_scale):
        encoded_views = []
        valid_views = []
        clipping = []
        for frame_index in sample["target_indices"]:
            path = self._clean_depth_path(sample["scene_name"], frame_index)
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["scene_name"]) != sample["scene_name"]:
                    raise ValueError(f"Clean target scene mismatch in {path}")
                if int(archive["frame_index"]) != frame_index:
                    raise ValueError(f"Clean target frame mismatch in {path}")
                model_hw = tuple(int(value) for value in archive["model_hw"])
                if model_hw != (self.height, self.width):
                    raise ValueError(
                        f"Clean target {path} is {model_hw}, expected "
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
            if self.semantic_exclusion_root is not None:
                semantic_mask_path = self._semantic_mask_path(
                    sample["scene_name"], frame_index
                )
                semantic_mask = cv2.imread(
                    str(semantic_mask_path), cv2.IMREAD_GRAYSCALE
                )
                if semantic_mask is None:
                    raise FileNotFoundError(semantic_mask_path)
                if semantic_mask.shape != valid.shape:
                    semantic_mask = cv2.resize(
                        semantic_mask,
                        (valid.shape[1], valid.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                valid &= semantic_mask <= 127
            encoded, stats = encode_log_radial_depth(
                depth,
                valid,
                pose_scale,
                self.minimum_normalized_range,
                self.maximum_normalized_range,
            )
            encoded = horizontal_wrap_nearest_fill(encoded, valid)
            encoded_views.append(torch.from_numpy(encoded).repeat(3, 1, 1))
            valid_views.append(torch.from_numpy(valid))
            clipping.append(stats)
        return encoded_views, torch.stack(valid_views), clipping

    def __getitem__(self, index):
        item = super().__getitem__(index)
        sample = self.samples[index % len(self.samples)]
        clean_depth, clean_valid, clean_clipping = self._load_clean_target_depth(
            sample, float(item["pose_scale"])
        )
        num_queries = len(sample["target_indices"])
        item["target_rasters"][-num_queries:] = clean_depth
        item["output_validity_masks"][-num_queries:] = clean_valid.float()
        item["depth_clipping"] = {
            "conditioning_from_reference_render": item["depth_clipping"],
            "target_from_clean_pointmap": clean_clipping,
        }
        item["depth_condition_source"] = str(sample["geometry_path"])
        item["depth_target_source"] = [
            str(self._clean_depth_path(sample["scene_name"], frame_index))
            for frame_index in sample["target_indices"]
        ]
        item["depth_target_semantic_exclusion"] = (
            None
            if self.semantic_exclusion_root is None
            else [
                str(self._semantic_mask_path(sample["scene_name"], frame_index))
                for frame_index in sample["target_indices"]
            ]
        )
        return item
