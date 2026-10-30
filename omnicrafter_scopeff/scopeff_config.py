"""Run configuration. Data / schedule fields keep configs/flux3_mix_scope_ff.yaml's names."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    # ── Model: Wan2.1-I2V-14B + the published FrameCrafter LoRA ────────────
    wan_root: str = ''                # holds Wan-AI/Wan2.1-I2V-14B-480P and Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
    base_initialization: str = ''     # published framecrafter.safetensors (420-ch patch embedding)
    lora_rank: int = 32
    lora_target_modules: str = 'q,k,v,o,ffn.0,ffn.2'

    # ── Camera: scope-ff ────────────────────────────────────────────────────
    scope_dir_freqs: int = 8
    scope_patch_size: bool = True
    scope_mlp_hidden: int = 64
    sort_refs_by_distance: bool = True

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500           # linear, then constant
    max_steps: int = 50000
    time_limit_hours: Optional[float] = None
    seed: int = 20261030

    # ── Data (RayCrafter loaders) ───────────────────────────────────────────
    near_depth_json: str = ''
    min_pixel_dropout: float = 0.0
    max_pixel_dropout: float = 0.75
    nvs_ratio: float = 0.5
    train_shards: str = ''
    val_shards: str = ''
    image_size: List[int] = field(default_factory=lambda: [288, 512])
    num_workers: int = 1
    panorama_ratio: float = 0.5
    panorama_root: str = ''
    panorama_train_manifests: List[str] = field(default_factory=list)
    panorama_val_manifests: List[str] = field(default_factory=list)
    panorama_near_depth_json: str = ''
    panorama_image_size: List[int] = field(default_factory=lambda: [256, 512])

    # ── Validation: flow-matching loss at fixed timesteps and noise ────────
    val_sets: int = 10                # per camera model, evenly spaced over the val split
    val_t_indices: List[int] = field(default_factory=lambda: [250, 500, 750])

    # ── Logging / checkpointing ─────────────────────────────────────────────
    output_dir: str = ''
    log_interval: int = 10
    val_interval: int = 500
    save_interval: int = 10000        # bf16 adapter step-N.safetensors
    resume_interval: int = 2000       # rolling full state in output_dir/resume
    use_wandb: bool = True
    wandb_project: str = 'raycrafter'
    wandb_run_name: str = ''

    def fingerprint(self) -> dict:
        """Fields a full-state resume must not change."""
        return {k: getattr(self, k) for k in (
            'lora_rank', 'lora_target_modules', 'scope_dir_freqs', 'scope_patch_size', 'scope_mlp_hidden')}

    def validate(self) -> None:
        for name in ('wan_root', 'base_initialization', 'near_depth_json', 'train_shards', 'val_shards',
                     'panorama_root', 'panorama_near_depth_json', 'output_dir'):
            if not getattr(self, name):
                raise ValueError(f'{name} is required')
        if not self.panorama_train_manifests or not self.panorama_val_manifests:
            raise ValueError('panorama_train_manifests and panorama_val_manifests are required')
        if not 0.0 <= self.min_pixel_dropout <= self.max_pixel_dropout <= 1.0:
            raise ValueError('pixel dropout must satisfy 0 <= min <= max <= 1')
        if not 0.0 <= self.nvs_ratio <= 1.0 or not 0.0 <= self.panorama_ratio <= 1.0:
            raise ValueError('nvs_ratio and panorama_ratio must be in [0, 1]')
        for name in ('image_size', 'panorama_image_size'):
            h, w = (int(v) for v in getattr(self, name))
            if h % 16 or w % 16:
                raise ValueError(f'{name} must be multiples of 16 (one DiT token), got {h}x{w}')
        h, w = (int(v) for v in self.panorama_image_size)
        if w != 2 * h:
            raise ValueError(f'panorama_image_size must be 2:1, got {h}x{w}')
        for name in ('max_steps', 'log_interval', 'val_interval', 'save_interval', 'resume_interval'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.scope_dir_freqs < 0 or self.scope_mlp_hidden <= 0:
            raise ValueError('scope_dir_freqs must be >= 0 and scope_mlp_hidden > 0')
        if any(not 0 <= int(t) < 1000 for t in self.val_t_indices):
            raise ValueError('val_t_indices index the 1000 training timesteps')

    @classmethod
    def from_yaml(cls, path) -> 'Config':
        values = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        unknown = [k for k in values if not hasattr(cfg, k)]
        if unknown:
            raise ValueError(f'Unknown config keys in {path}: {unknown}')
        for k, v in values.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def save_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), default_flow_style=False, sort_keys=False))
