"""Inference wrapper for pose-conditioned render-query FrameCrafter models."""

from __future__ import annotations

import torch
from PIL import Image

from model import FrameCrafter


class PoseConditionedRenderQueryCrafter(FrameCrafter):
    """Generate clean target views from references, query renders, and poses."""

    def generate_render_queries(
        self,
        reference_images: list[Image.Image],
        query_images: list[Image.Image],
        raymap: torch.Tensor,
        height: int = 256,
        width: int = 512,
        num_inference_steps: int = 50,
        seed: int = 42,
        cfg_scale: float = 1.0,
        prompt: str = "",
    ) -> list[Image.Image]:
        if self.input_dim != self.IN_DIM:
            raise ValueError(
                f"Pose-conditioned render queries require {self.IN_DIM} input channels"
            )
        if not reference_images:
            raise ValueError("At least one clean reference image is required")
        if not query_images:
            raise ValueError("At least one query conditioning image is required")
        if width != 2 * height:
            raise ValueError(f"Expected a 2:1 panorama, got {width}x{height}")

        num_total = len(reference_images) + len(query_images)
        expected_shape = (num_total, 384, height // 8, width // 8)
        if tuple(raymap.shape) != expected_shape:
            raise ValueError(
                f"Expected ERP raymap shape {expected_shape}, got {tuple(raymap.shape)}"
            )

        references = [
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            for image in reference_images
        ]
        queries = [
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            for image in query_images
        ]
        video = self.pipe(
            prompt=prompt,
            negative_prompt="",
            input_image=references,
            conditioning_images=queries,
            raymap=raymap.to(self.device, dtype=torch.bfloat16),
            height=height,
            width=width,
            num_frames=num_total,
            num_latent_frames=num_total,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            tiled=True,
        )
        return video[-len(queries):]
