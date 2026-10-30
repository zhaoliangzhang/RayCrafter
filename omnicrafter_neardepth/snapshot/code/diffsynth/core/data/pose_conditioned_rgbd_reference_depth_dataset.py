"""Masked 10-slot RGB-D cleanup/NVS dataset with clean reference depth.

The external observations are two clean reference RGB panoramas, two clean
reference radial-depth panoramas, and three paired noisy RGB/depth query
renders. Internally the diffusion sequence contains four known reference
slots followed by six generated query slots::

    ref_rgb[2], ref_depth[2], query_rgb[3], query_depth[3]

Sky and operator masks are exclusion masks. Excluded pixels are removed from
all conditioning/target rasters and from RGB/depth output supervision.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import (
    get_equirectangular_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)
from .pose_conditioned_rgbd_clean_target_dataset import (
    WanPanoramaPoseConditionedRGBDCleanTargetDataset,
)
from .pose_conditioned_rgbd_query_dataset import (
    DEPTH_MODALITY,
    RGB_MODALITY,
    encode_log_radial_depth,
    horizontal_wrap_nearest_fill,
    unpack_bit_mask,
)


def apply_validity_mask(
    raster: torch.Tensor,
    valid: torch.Tensor,
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Fill invalid pixels of a three-channel raster without changing input."""
    if raster.ndim != 3 or raster.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] raster, got {tuple(raster.shape)}")
    if valid.shape != raster.shape[-2:]:
        raise ValueError(
            f"Validity shape {tuple(valid.shape)} does not match "
            f"raster {tuple(raster.shape[-2:])}"
        )
    output = raster.clone()
    output[:, ~valid.bool()] = fill_value
    return output


class WanPanoramaPoseConditionedRGBDReferenceDepthDataset(
    WanPanoramaPoseConditionedRGBDCleanTargetDataset
):
    """Return four known reference slots and six conditioned query outputs."""

    def __init__(self, *args, semantic_exclusion_root, **kwargs):
        if semantic_exclusion_root is None:
            raise ValueError("The masked 10-slot dataset requires semantic masks")
        super().__init__(
            *args,
            semantic_exclusion_root=semantic_exclusion_root,
            **kwargs,
        )
        for sample in self.samples:
            for frame_index in sample["ref_indices"]:
                depth_path = self._clean_depth_path(sample["scene_name"], frame_index)
                mask_path = self._semantic_mask_path(sample["scene_name"], frame_index)
                if not depth_path.is_file():
                    raise FileNotFoundError(
                        f"Missing clean reference depth pseudo-target: {depth_path}"
                    )
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Missing reference semantic exclusion mask: {mask_path}"
                    )

    def _load_semantic_validity(self, scene_name, frame_indices):
        valid_views = []
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
            valid_views.append(torch.from_numpy(exclusion <= 127))
        return torch.stack(valid_views)

    def _load_clean_depth_frames(
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
            # Filling keeps VAE boundary behavior stable; apply_validity_mask
            # removes these values from both conditioning and supervision.
            encoded = horizontal_wrap_nearest_fill(encoded, valid)
            encoded_tensor = torch.from_numpy(encoded).repeat(3, 1, 1)
            valid_tensor = torch.from_numpy(valid)
            encoded_views.append(apply_validity_mask(encoded_tensor, valid_tensor))
            valid_views.append(valid_tensor)
            clipping.append(stats)
        return encoded_views, torch.stack(valid_views), clipping

    def _ten_slot_raymap(self, sample):
        physical_indices = list(
            dict.fromkeys(sample["ref_indices"] + sample["target_indices"])
        )
        scene_poses = self._scene_poses(sample["scene_name"])
        physical_w2cs = torch.from_numpy(scene_poses[physical_indices]).float()
        _, physical_c2w_norm, scale = normalize_w2c_make_cam_last_origin(
            physical_w2cs
        )
        normalized_pose_by_index = {
            frame_index: physical_c2w_norm[position]
            for position, frame_index in enumerate(physical_indices)
        }
        slot_indices = (
            sample["ref_indices"]
            + sample["ref_indices"]
            + sample["target_indices"]
            + sample["target_indices"]
        )
        slot_c2ws = torch.stack(
            [normalized_pose_by_index[frame_index] for frame_index in slot_indices]
        )
        raymap = get_equirectangular_plucker_rays(
            slot_c2ws,
            height=self.height,
            width=self.width,
            no_pixel_unshuffle=self.no_pixel_unshuffle,
        )
        return raymap, slot_indices, float(scale)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        sample = self.samples[index % len(self.samples)]
        num_references = len(sample["ref_indices"])
        num_queries = len(sample["target_indices"])
        if num_references != 2 or num_queries != 3:
            raise ValueError(
                "The 10-slot experiment requires exactly two references and "
                f"three queries, got {num_references} and {num_queries}"
            )

        reference_semantic_valid = self._load_semantic_validity(
            sample["scene_name"], sample["ref_indices"]
        )
        query_semantic_valid = self._load_semantic_validity(
            sample["scene_name"], sample["target_indices"]
        )
        raymap, slot_indices, pose_scale = self._ten_slot_raymap(sample)
        if not np.isclose(pose_scale, float(item["pose_scale"]), rtol=1e-6):
            raise ValueError("10-slot and legacy pose normalization scales differ")

        reference_depth, reference_depth_valid, reference_depth_clipping = (
            self._load_clean_depth_frames(
                sample["scene_name"],
                sample["ref_indices"],
                pose_scale,
                reference_semantic_valid,
            )
        )

        reference_rgb = item["input_rasters"]
        noisy_rgb = item["conditioning_rasters"][:num_queries]
        noisy_depth = item["conditioning_rasters"][num_queries:]
        target_rgb = item["target_rasters"][
            num_references : num_references + num_queries
        ]
        target_depth = item["target_rasters"][-num_queries:]

        dropout_valid = ~item["query_pixel_dropout_masks"].bool()
        # The old rendered-depth validity identifies point-cloud holes. Apply
        # it jointly to RGB/depth conditioning, then remove semantic regions.
        rendered_depth_valid = item["query_depth_conditioning_validity_masks"].bool()
        query_condition_valid = (
            query_semantic_valid & rendered_depth_valid & dropout_valid
        )
        target_depth_valid = item["output_validity_masks"][-num_queries:].bool()

        reference_rgb = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(reference_rgb, reference_semantic_valid)
        ]
        noisy_rgb = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(noisy_rgb, query_condition_valid)
        ]
        noisy_depth = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(noisy_depth, query_condition_valid)
        ]
        target_rgb = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(target_rgb, query_semantic_valid)
        ]
        target_depth = [
            apply_validity_mask(raster, valid)
            for raster, valid in zip(target_depth, target_depth_valid)
        ]

        item.update(
            {
                "input_rasters": reference_rgb + reference_depth,
                "conditioning_rasters": noisy_rgb + noisy_depth,
                "target_rasters": (
                    reference_rgb
                    + reference_depth
                    + target_rgb
                    + target_depth
                ),
                "raymap": raymap,
                "modality_ids": torch.tensor(
                    [RGB_MODALITY] * num_references
                    + [DEPTH_MODALITY] * num_references
                    + [RGB_MODALITY] * num_queries
                    + [DEPTH_MODALITY] * num_queries,
                    dtype=torch.long,
                ),
                "input_validity_masks": torch.cat(
                    [reference_semantic_valid, reference_depth_valid], dim=0
                ).float(),
                "conditioning_validity_masks": torch.cat(
                    [query_condition_valid, query_condition_valid], dim=0
                ).float(),
                "output_validity_masks": torch.cat(
                    [query_semantic_valid, target_depth_valid], dim=0
                ).float(),
                "slot_roles": ["reference_rgb"] * num_references
                + ["reference_depth"] * num_references
                + ["query_rgb"] * num_queries
                + ["query_depth"] * num_queries,
                "slot_frame_indices": slot_indices,
                "known_slot_count": 2 * num_references,
                "output_slot_count": 2 * num_queries,
                "reference_depth_clipping": reference_depth_clipping,
                "reference_depth_source": [
                    str(self._clean_depth_path(sample["scene_name"], frame_index))
                    for frame_index in sample["ref_indices"]
                ],
                "semantic_exclusion_source": [
                    str(self._semantic_mask_path(sample["scene_name"], frame_index))
                    for frame_index in slot_indices
                ],
            }
        )
        return item
