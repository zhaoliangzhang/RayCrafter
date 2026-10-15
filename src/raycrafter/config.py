"""Run configuration. Every field is documented in configs/example.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class Config:
    # ── Backbone ───────────────────────────────────────────────────────────
    backbone: str = "flux2"
    model_variant: str = "flux.2-klein-4b"
    model_path: Optional[str] = None
    ae_path: Optional[str] = None

    # ── LoRA ───────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj", "linear1", "linear2",
    ])

    # ── Views ──────────────────────────────────────────────────────────────
    num_views: int = 1
    ref_size: int = 6
    ref_t_scale: Optional[int] = 10
    sort_refs_by_distance: bool = True

    # ── Camera conditioning ────────────────────────────────────────────────
    camera_condition: str = "scope"   # "none" | "scope"
    rope_type: str = "rayrope"        # "flux" | "rayrope"

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 20000
    warmup_steps: int = 500
    lr_schedule: str = "cosine"       # "cosine" | "constant"
    time_limit_hours: Optional[float] = None
    gradient_clip_norm: float = 1.0
    t_min: float = 0.02
    t_max: float = 1.0
    t_one_prob: float = 0.5
    mixed_precision: str = "bf16"     # "bf16" | "fp32"
    lpips_loss_weight: float = 10.0
    l1_pixel_loss_weight: float = 1.0
    pixel_loss_latent_crop: Optional[int] = None
    pixel_loss_t_weighting: str = "none"   # "none" | "one_minus_t"
    gradient_checkpointing: bool = True

    # ── Curriculum / synthetic query noise ─────────────────────────────────
    enable_curriculum: bool = False
    curriculum_steps: Optional[int] = None
    curriculum_sigma: float = 0.16
    curriculum_min_mass: float = 0.10
    curriculum_start_center: float = 0.0
    curriculum_end_center: float = 0.52
    curriculum_start_probs: Optional[List[float]] = None
    curriculum_end_probs: Optional[List[float]] = None
    noise_levels: Optional[List[float]] = None
    noise_mode: str = "lerp"          # "lerp" | "additive"
    noise_space: str = "latent"       # "latent" | "pixel"

    # ── Data ───────────────────────────────────────────────────────────────
    train_shards: Union[str, List[str]] = ""
    val_shards: Union[str, List[str]] = ""
    image_size: List[int] = field(default_factory=lambda: [432, 768])
    num_workers: int = 1
    window: int = 15

    # ── Logging / checkpointing ────────────────────────────────────────────
    output_dir: str = "outputs/run"
    log_interval: int = 250
    val_interval: int = 500
    save_interval: int = 5000
    use_wandb: bool = False
    wandb_project: str = "raycrafter"
    wandb_run_name: str = ""

    # ── Resume ─────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None
    start_step: int = 0

    # ── Inference ──────────────────────────────────────────────────────────
    num_inference_steps: Union[int, List[int]] = field(default_factory=lambda: [1, 4])

    # ────────────────────────────────────────────────────────────────────────

    @property
    def uses_scope(self) -> bool:
        return self.camera_condition == "scope"

    @property
    def uses_rayrope(self) -> bool:
        return self.rope_type == "rayrope"

    @property
    def needs_cameras(self) -> bool:
        return self.uses_scope or self.uses_rayrope

    @property
    def use_pose(self) -> bool:
        """The dataloader reads pose.json whenever a feature consumes cameras."""
        return self.needs_cameras or self.sort_refs_by_distance

    def validate(self) -> None:
        choices = {
            "backbone": ("flux2",),
            "camera_condition": ("none", "scope"),
            "rope_type": ("flux", "rayrope"),
            "lr_schedule": ("cosine", "constant"),
            "mixed_precision": ("bf16", "fp32"),
            "pixel_loss_t_weighting": ("none", "one_minus_t"),
            "noise_mode": ("lerp", "additive"),
            "noise_space": ("latent", "pixel"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = cls()
        unknown = [k for k in d if not hasattr(cfg, k)]
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        for k, v in d.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=False)
