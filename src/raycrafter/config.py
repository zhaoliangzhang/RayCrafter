"""Run configuration. Every field is documented in configs/example.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class Config:
    # ── Backbone ───────────────────────────────────────────────────────────
    backbone: str = "flux2"           # "flux2" | "flux3"
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
    rope_type: str = "rayrope"        # "flux" | "rayrope" | "prope" | "unirope" | "unirope_anchor"
    # RayRoPE / UniRoPE: also rotate V and the attention output (the RayRoPE
    # paper's default). Off by default so older checkpoints evaluate as trained.
    rope_apply_vo: bool = False
    # UniRoPE: encode each token's patch footprint (three corner points at depth)
    # instead of only its center point. Off by default for older checkpoints.
    unirope_footprint: bool = False

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    # RayRoPE / UniRoPE depth heads: their own AdamW group. They start at zero
    # (log depth 0, sigma 3) and at the backbone's 1e-5 barely move (sigma was
    # still 2.96 after 90k steps); the reference RayRoPE trains them with the
    # whole model at 4e-4 and weight decay 0.5. None: learning_rate / weight_decay.
    depth_head_lr: Optional[float] = None
    depth_head_weight_decay: Optional[float] = None
    max_steps: int = 20000
    warmup_steps: int = 500
    lr_schedule: str = "cosine"       # "cosine" | "constant"
    time_limit_hours: Optional[float] = None
    gradient_clip_norm: float = 1.0
    t_min: float = 0.02
    t_max: float = 1.0
    t_one_prob: float = 0.5
    mixed_precision: str = "bf16"     # "bf16" | "fp32"
    lpips_loss_weight: float = 5.0
    l1_pixel_loss_weight: float = 1.0
    pixel_loss_latent_crop: Optional[int] = None
    pixel_loss_t_weighting: str = "none"   # "none" | "one_minus_t" | "one_minus_t_noise"
    gradient_checkpointing: bool = True
    shard_optimizer: bool = False

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
    data_format: str = "webdataset"   # "webdataset" | "dl3dv_points"
    train_shards: Union[str, List[str]] = ""
    val_shards: Union[str, List[str]] = ""
    # dl3dv_points only. Same draw as the panorama render-query loader:
    # each query's dropout percent is uniform on the integers in [min, max].
    min_pixel_dropout: float = 0.0
    max_pixel_dropout: float = 1.0
    nvs_probability: float = 0.0      # sample-level: every query becomes black
    nvs_ratio: float = 0.0            # per query: that query becomes black
    # dl3dv_points only: {scene_name: {"near_depth": d}} from
    # scripts/estimate_near_depth_dl3dv_points.py. Translations are divided by
    # d (SCoPE's near-depth normalization); scenes without one are dropped.
    near_depth_json: Optional[str] = None
    # Equirect panorama sets (data/panorama.py) mixed into dl3dv_points training:
    # each training batch is a panorama batch with probability panorama_ratio
    # (one camera model per batch). The dropout / NVS settings above apply to
    # both. Manifests are relative to panorama_root.
    panorama_ratio: float = 0.0
    panorama_root: Optional[str] = None
    panorama_train_manifests: List[str] = field(default_factory=list)
    panorama_val_manifests: List[str] = field(default_factory=list)
    # From scripts/estimate_near_depth_panorama.py; the same unit as near_depth_json.
    panorama_near_depth_json: Optional[str] = None
    panorama_image_size: List[int] = field(default_factory=lambda: [256, 512])
    image_size: List[int] = field(default_factory=lambda: [432, 768])
    num_workers: int = 1
    window: int = 15
    open_sequences: int = 4

    # ── Logging / checkpointing ────────────────────────────────────────────
    output_dir: str = "outputs/run"
    log_interval: int = 250
    val_interval: int = 500
    save_interval: int = 20000
    # One overwritten full state (fp32 weights + AdamW) so a killed job can resume.
    resume_interval: int = 2000
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
    def needs_cameras(self) -> bool:
        return self.uses_scope or self.rope_type != "flux"

    @property
    def use_pose(self) -> bool:
        """The dataloader reads pose.json whenever a feature consumes cameras."""
        return self.needs_cameras or self.sort_refs_by_distance

    def validate(self) -> None:
        choices = {
            "backbone": ("flux2", "flux3"),
            "camera_condition": ("none", "scope"),
            "rope_type": ("flux", "rayrope", "prope", "unirope", "unirope_anchor"),
            "lr_schedule": ("cosine", "constant"),
            "mixed_precision": ("bf16", "fp32"),
            "pixel_loss_t_weighting": ("none", "one_minus_t", "one_minus_t_noise"),
            "noise_mode": ("lerp", "additive"),
            "noise_space": ("latent", "pixel"),
            "data_format": ("webdataset", "dl3dv_points"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if not 0.0 <= float(self.min_pixel_dropout) <= float(self.max_pixel_dropout) <= 1.0:
            raise ValueError(
                "pixel dropout must satisfy 0 <= min_pixel_dropout <= max_pixel_dropout <= 1, "
                f"got {self.min_pixel_dropout} and {self.max_pixel_dropout}"
            )
        if not 0.0 <= float(self.nvs_probability) <= 1.0:
            raise ValueError(f"nvs_probability must be in [0, 1], got {self.nvs_probability}")
        if not 0.0 <= float(self.nvs_ratio) <= 1.0:
            raise ValueError(f"nvs_ratio must be in [0, 1], got {self.nvs_ratio}")
        if float(self.nvs_probability) > 0.0 and float(self.nvs_ratio) > 0.0:
            raise ValueError("nvs_probability (per sample) and nvs_ratio (per query) cannot both be set")
        if int(self.save_interval) <= 0 or int(self.resume_interval) <= 0:
            raise ValueError(
                "save_interval and resume_interval must be positive, "
                f"got {self.save_interval} and {self.resume_interval}"
            )
        if not 0.0 <= float(self.panorama_ratio) <= 1.0:
            raise ValueError(f"panorama_ratio must be in [0, 1], got {self.panorama_ratio}")
        if float(self.panorama_ratio) > 0.0:
            if self.data_format != "dl3dv_points":
                raise ValueError("panorama_ratio > 0 mixes panoramas into data_format: dl3dv_points")
            if not self.panorama_root or not self.panorama_train_manifests:
                raise ValueError("panorama_ratio > 0 needs panorama_root and panorama_train_manifests")
            # The VAE (2x2-patchified latents) maps 16 px to one token: other sizes
            # get center-cropped, which cuts an equirect image's longitude range.
            for name in ("image_size", "panorama_image_size"):
                h, w = (int(v) for v in getattr(self, name))
                if h % 16 or w % 16:
                    raise ValueError(f"{name} must be multiples of 16 when mixing panoramas, got {h}x{w}")
            h, w = (int(v) for v in self.panorama_image_size)
            if w != 2 * h:
                raise ValueError(f"panorama_image_size must be 2:1, got {h}x{w}")
        if self.data_format == "dl3dv_points" and (self.num_views != 3 or self.ref_size != 2):
            raise ValueError(
                "dl3dv_points samples are 2 reference images and 3 target views; "
                f"got ref_size={self.ref_size} num_views={self.num_views}"
            )

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
