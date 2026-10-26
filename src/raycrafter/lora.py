"""Minimal LoRA (no peft dependency) and checkpoint helpers."""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

# Parameter-name fragments saved in a LoRA checkpoint: the adapters plus the
# camera modules, which are new full weights even in LoRA mode.
TRAINABLE_KEYS = ("lora_a", "lora_b", "scope_pe", "rayrope_depth", "ray_in.")


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter (B starts at zero)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        dev, dtype = base.weight.device, base.weight.dtype
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=dev, dtype=dtype)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=dev, dtype=dtype)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(self.dropout(x)))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight


def add_lora(
    model: nn.Module, target_suffixes: List[str], rank: int, alpha: float, dropout: float = 0.0
) -> Dict[str, LoRALinear]:
    """Wrap every nn.Linear whose dotted name ends with one of ``target_suffixes``."""
    replaced: Dict[str, LoRALinear] = {}
    candidates = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, mod in candidates:
        if any(name.endswith(s) for s in target_suffixes):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replaced[name] = LoRALinear(mod, rank, alpha, dropout)
            setattr(parent, child, replaced[name])
    return replaced


def freeze_base_model(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "lora_a" not in name and "lora_b" not in name:
            p.requires_grad_(False)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_checkpoint(model: nn.Module, path: str, lora_only: bool) -> None:
    """Save the transformer in bf16: the full state dict, or only TRAINABLE_KEYS for LoRA."""
    sd = {
        k: v.detach().to(torch.bfloat16).cpu()
        for k, v in model.state_dict().items()
        if not lora_only or any(key in k for key in TRAINABLE_KEYS)
    }
    save_file(sd, path)


def load_trainable_weights(model: nn.Module, path: str) -> None:
    """Load a LoRA checkpoint (adapters + camera modules) into a model that already has them."""
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if any(key in k for key in TRAINABLE_KEYS)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
