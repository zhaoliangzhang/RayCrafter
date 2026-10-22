#!/usr/bin/env python3
"""Train RayCrafter.

    # 1 GPU
    python train.py --config configs/my_run.yaml
    # N GPUs (DDP)
    torchrun --standalone --nproc_per_node=N train.py --config configs/my_run.yaml
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Before CUDA init: long multi-view sequences otherwise fragment memory and OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fire
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent / "src"))

from raycrafter.checkpoint import (
    finalize_resume_dir, load_training_state, save_training_state, validate_resume_meta,
)
from raycrafter.config import Config
from raycrafter.data import make_dataloader
from raycrafter.data.curriculum import make_progress_holder
from raycrafter.lora import count_parameters, save_checkpoint
from raycrafter.model import RayCrafterModel

try:
    import wandb
except ImportError:
    wandb = None

# Wall-clock origin for time_limit_hours (model loading counts against a Slurm limit too).
_PROCESS_START = time.time()


def _lr(cfg: Config, step: int) -> float:
    warm = min(1.0, step / max(1, cfg.warmup_steps))
    if cfg.lr_schedule == "constant" or step < cfg.warmup_steps:
        return cfg.learning_rate * warm
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _progress(cfg: Config, step: int) -> float:
    return min(1.0, step / max(cfg.curriculum_steps or cfg.max_steps, 1))


def _cameras(batch: dict, device):
    if "camera_c2w" not in batch:
        return {}
    return {"camera_c2w": batch["camera_c2w"].to(device),
            "camera_fxfycxcy": batch["camera_fxfycxcy"].to(device)}


def train(config: str) -> None:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    cfg = Config.from_yaml(config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = out_dir / "resume"
    if is_main:
        finalize_resume_dir(resume_dir)
    if world_size > 1:
        dist.barrier()
    has_resume = (resume_dir / "meta.json").is_file()
    if has_resume:
        # The rolling checkpoint has weights, AdamW and the step. A bf16
        # resume_from would just be overwritten, and its optimizer would be cold.
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        validate_resume_meta(resume_meta, world_size, world_size > 1 and cfg.shard_optimizer)
        cfg.resume_from = None
        cfg.start_step = int(resume_meta["step"])
    use_wandb = cfg.use_wandb and wandb is not None
    if is_main:
        cfg.save_yaml(out_dir / "config.yaml")
        print(f"variant={cfg.model_variant} lora={cfg.use_lora} views={cfg.num_views} refs={cfg.ref_size} "
              f"camera_condition={cfg.camera_condition} rope={cfg.rope_type} "
              f"steps={cfg.start_step}..{cfg.max_steps}")
        if has_resume:
            print(f"Resuming full state (weights + AdamW) from {resume_dir} at step {cfg.start_step}")
        if cfg.use_wandb and wandb is None:
            print("use_wandb=true but wandb is not importable; logging to stdout only")
        if use_wandb:
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name or None,
                       config=cfg.__dict__, dir=str(out_dir))

    model = RayCrafterModel(cfg, device)
    # fp32 master weights for everything trainable: AdamW updates of ~1e-5 are
    # below bf16's ulp and would round away. Forward still runs in bf16 (autocast).
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable, total = count_parameters(model.transformer)
    if is_main:
        print(f"Parameters: {trainable:,} trainable / {total:,} ({100 * trainable / total:.2f}%)")

    xfm = model.transformer
    if world_size > 1:
        # gradient_as_bucket_view: grads live in the allreduce buckets (no 2nd fp32 copy).
        model.transformer = DDP(xfm, device_ids=[device], broadcast_buffers=False,
                                find_unused_parameters=False, gradient_as_bucket_view=True)
    train_params = [p for p in xfm.parameters() if p.requires_grad]
    adamw = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    if world_size > 1 and cfg.shard_optimizer:
        # ZeRO-1: each rank keeps AdamW state for 1/world_size of the parameters and
        # broadcasts its updated shard. fused=True avoids shard-sized foreach temporaries.
        optimizer = ZeroRedundancyOptimizer(train_params, optimizer_class=AdamW, fused=True, **adamw)
    else:
        optimizer = AdamW(train_params, **adamw)
    if has_resume:
        load_training_state(xfm, optimizer, resume_dir, rank, world_size)

    saved_resume_at = -1

    def save_resume(at_step: int, force: bool = False) -> None:
        nonlocal saved_resume_at
        if at_step <= 0 or saved_resume_at == at_step:
            return
        if not force and at_step % cfg.resume_interval != 0:
            return
        save_training_state(xfm, optimizer, at_step, resume_dir, rank, world_size)
        saved_resume_at = at_step

    progress = (make_progress_holder(_progress(cfg, cfg.start_step))
                if (cfg.enable_curriculum or cfg.noise_levels) else None)
    common = dict(num_views=cfg.num_views, ref_size=cfg.ref_size, image_size=tuple(cfg.image_size),
                  batch_size=cfg.batch_size, use_pose=cfg.use_pose, window=cfg.window)
    points = dict(
        data_format=cfg.data_format, min_pixel_dropout=cfg.min_pixel_dropout,
        max_pixel_dropout=cfg.max_pixel_dropout, nvs_probability=cfg.nvs_probability,
        nvs_ratio=cfg.nvs_ratio, near_depth_json=cfg.near_depth_json,
    )
    train_loader = make_dataloader(
        cfg.train_shards, num_workers=cfg.num_workers, shuffle=True,
        enable_curriculum=cfg.enable_curriculum, curriculum_sigma=cfg.curriculum_sigma,
        curriculum_min_mass=cfg.curriculum_min_mass,
        curriculum_start_center=cfg.curriculum_start_center,
        curriculum_end_center=cfg.curriculum_end_center,
        curriculum_start_probs=cfg.curriculum_start_probs,
        curriculum_end_probs=cfg.curriculum_end_probs,
        noise_levels=cfg.noise_levels, noise_mode=cfg.noise_mode, noise_space=cfg.noise_space,
        curriculum_progress=progress, open_sequences=cfg.open_sequences, **points, **common,
    )
    val_loader = None
    if cfg.val_shards:
        val_loader = make_dataloader(
            cfg.val_shards, num_workers=1, shuffle=False, repeat=False, **points, **common,
        )

    steps = cfg.num_inference_steps if isinstance(cfg.num_inference_steps, list) else [cfg.num_inference_steps]
    train_iter = iter(train_loader)
    step = int(cfg.start_step)
    t_last = time.time()
    model.train()

    while step < cfg.max_steps:
        step += 1
        if progress is not None:
            progress.value = _progress(cfg, step)
        lr = _lr(cfg, step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        src, tgt, ref = (batch[k].to(device) for k in ("src", "tgt", "ref"))
        cams = _cameras(batch, device)

        # In-place zeroing keeps grads as views of the DDP buckets.
        optimizer.zero_grad(set_to_none=False)
        out = model.training_step(src, tgt, ref, noise_alpha=batch["noise_alpha"], **cams)
        out["loss"].backward()
        if cfg.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(train_params, cfg.gradient_clip_norm)
        optimizer.step()

        if is_main:
            now = time.time()
            log = {"train/loss": out["loss"].item(), "train/lr": lr,
                   "train/noise_frac": (batch["noise_alpha"] > 0).float().mean().item(),
                   "perf/steps_per_sec": 1.0 / max(now - t_last, 1e-6)}
            t_last = now
            for k in ("latent_loss", "lpips_loss", "l1_pixel_loss", "v_mag"):
                if k in out:
                    log[f"train/{k}"] = out[k]
            if use_wandb:
                wandb.log(log, step=step)
            if step == 1 or step % 50 == 0:
                mem = torch.cuda.max_memory_allocated(device) / 2**30
                print(f"step {step:>7d} | " + " | ".join(f"{k}={v:.4g}" for k, v in log.items())
                      + f" | peak_mem={mem:.1f}GiB", flush=True)

        # Every rank must enter sample() while the transformer is DDP-wrapped.
        if use_wandb and (step - 1) % cfg.log_interval == 0:
            _log_images(model, src, tgt, ref, batch["noise_alpha"], cams, steps, step, upload=is_main)

        if val_loader is not None and step % cfg.val_interval == 0:
            val_loss, val_loss_nvs = _validate(model, val_loader, device)
            if is_main:
                print(f"step {step:>7d} | val/loss={val_loss:.4f} | val/loss_nvs={val_loss_nvs:.4f}", flush=True)
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/loss_nvs": val_loss_nvs}, step=step)

        if is_main and step % cfg.save_interval == 0:
            save_checkpoint(xfm, str(out_dir / f"transformer_step{step}.safetensors"), cfg.use_lora)
        # All ranks: one overwritten resume/ with fp32 weights and this rank's AdamW shard.
        save_resume(step)

        if world_size > 1:
            dist.barrier()

        if cfg.time_limit_hours is not None:
            # Rank 0 decides so every rank leaves on the same step.
            stop = torch.tensor(float(time.time() - _PROCESS_START >= cfg.time_limit_hours * 3600), device=device)
            if world_size > 1:
                dist.broadcast(stop, src=0)
            if stop.item():
                if is_main:
                    print(f"time_limit_hours={cfg.time_limit_hours} reached at step {step}", flush=True)
                save_resume(step, force=True)
                break

    save_resume(step, force=True)
    if is_main:
        save_checkpoint(xfm, str(out_dir / "transformer_final.safetensors"), cfg.use_lora)
        print(f"Done at step {step}. Weights → {out_dir}")
        if use_wandb:
            wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


@torch.no_grad()
def _log_images(model, src, tgt, ref, noise_alpha, cams, steps, step, upload: bool) -> None:
    import numpy as np

    def to_uint8(t):
        return ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)

    model.eval()
    preds = {n: model.sample(src, ref, num_steps=n, noise_alpha=noise_alpha, tgt_imgs=tgt, **cams)
             for n in steps}
    model.train()
    if not upload:
        return

    logs = {f"val/refined_vs_gt_mae_{n}step": (p - tgt).abs().mean().item() for n, p in preds.items()}
    latent_noise = model.cfg.noise_space == "latent"
    for n, p in preds.items():
        rows = []
        for b in range(min(src.shape[0], 2)):
            a = float(noise_alpha[b])
            for v in range(min(src.shape[1], 3)):
                # Latent noise never touches src pixels: show gray instead of the unused render.
                s = torch.zeros_like(src[b, v]) if (latent_noise and a > 0) else src[b, v]
                row = np.concatenate([to_uint8(s), to_uint8(p[b, v]), to_uint8(tgt[b, v])], axis=1)
                rows.append(wandb.Image(row, caption=f"b{b} v{v} noise_alpha={a:g}  src | pred | gt"))
        logs[f"val/samples_{n}step"] = rows
    wandb.log(logs, step=step)


@torch.no_grad()
def _validate(model, loader, device, max_batches: int = 20):
    """Mean loss over the val batches as refinement (their degraded renders) and as
    NVS (pure-noise input, noise_alpha=1; meaningful with noise_space=latent)."""
    model.eval()
    total, total_nvs, count = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src, tgt, ref = (batch[k].to(device) for k in ("src", "tgt", "ref"))
        cams = _cameras(batch, device)
        total += model.training_step(src, tgt, ref, noise_alpha=batch["noise_alpha"], **cams)["loss"].item()
        nvs = torch.ones_like(batch["noise_alpha"])
        total_nvs += model.training_step(src, tgt, ref, noise_alpha=nvs, **cams)["loss"].item()
        count += 1
    model.train()
    return total / max(count, 1), total_nvs / max(count, 1)


if __name__ == "__main__":
    fire.Fire(train)
