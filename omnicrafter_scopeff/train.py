"""OmniCrafter (Wan2.1-I2V-14B + FrameCrafter LoRA) with scope-ff, perspective + panorama.

ZeRO-2 bf16 through accelerate + DeepSpeed (4 GPUs x 1 set for the 50k run), as the near-depth run's
``snapshot/train.py``: the same module construction, frozen-encoder release and
flow-matching step. Data, camera encoding and schedule follow
``configs/flux3_mix_scope_ff.yaml``: RayCrafter's dl3dv_points + panorama loaders
(one camera model per rank and step), scope-ff rays on near-depth translations,
AdamW 1e-5 with a 500-step warmup.

Checkpoints in ``output_dir``:
- ``step-N.safetensors``: bf16 trainable weights every ``save_interval`` and at the end.
- ``resume/``: the full training state (DeepSpeed model + per-rank ZeRO-2 optimizer
  shards with fp32 masters and AdamW moments, LR scheduler, per-rank RNG, ``meta.json``),
  rewritten every ``resume_interval``, at ``time_limit_hours`` and at the end. A
  restart with the same config and world size continues from it.
- ``resume_step<max_steps>/``: hard links to the final full state, kept when a later
  run continues in the same directory.

    accelerate launch --config_file omnicrafter_scopeff/accelerate_zero2.yaml --num_processes 4 \\
        --mixed_precision bf16 omnicrafter_scopeff/train.py --config <yaml>
    python omnicrafter_scopeff/train.py --config <yaml> --dry-run   # CPU: paths, data, rays
"""
import argparse
import fcntl
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Wall-clock origin for time_limit_hours (model loading counts against the Slurm limit too).
_PROCESS_START = time.time()

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import scopeff_paths  # noqa: F401
from scopeff_config import Config
from scopeff_model import ScopeFFTrainingModule, black_fraction, to_omnicrafter
from common import model_paths
from frozen_encoders import release_frozen_encoders
from handoff_common import PUBLISHED_SHA256, read_json, sha256, write_json
from raycrafter.data.dataset import MixedLoader, _collate
from raycrafter.data.dl3dv_points import Dl3dvPoints, _split_root, load_near_depth, make_points_dataloader
from raycrafter.data.panorama import PanoramaSets, make_panorama_dataloader

LORA_TENSORS = 800
PATCH_TENSORS = 2
# eq.0, eq.2, ek.0, ek.2, norm_pe_q, norm_pe_k, scale_gate (2 weights, 2 biases), alpha_q, alpha_k.
SCOPE_TENSORS_PER_BLOCK = 12
NUM_BLOCKS = 40
INITIALIZATION = 'initialization/framecrafter_image36.safetensors'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


# ── Data ──────────────────────────────────────────────────────────────────────

def build_train_loader(cfg, rank, world, start_step):
    common = dict(num_views=3, ref_size=2, batch_size=1, num_workers=cfg.num_workers, use_pose=True,
                  shuffle=True, min_pixel_dropout=cfg.min_pixel_dropout,
                  max_pixel_dropout=cfg.max_pixel_dropout, nvs_ratio=cfg.nvs_ratio)
    points = make_points_dataloader(cfg.train_shards, image_size=tuple(cfg.image_size),
                                    near_depth_json=cfg.near_depth_json, **common)
    panoramas = make_panorama_dataloader(root=cfg.panorama_root, manifests=cfg.panorama_train_manifests,
                                         image_size=tuple(cfg.panorama_image_size),
                                         near_depth_json=cfg.panorama_near_depth_json, **common)
    # Per-rank source draw, as flux3 (ranks see different camera models in a step);
    # a resumed job draws a new sequence instead of replaying the first one.
    return MixedLoader([points, panoramas], [1.0 - cfg.panorama_ratio, cfg.panorama_ratio],
                       seed=rank + world * start_step)


def build_val_sets(cfg, rank, world):
    """This rank's share of the fixed validation sets, evenly spaced over the
    dl3dv_points test split and the held-out JHU panoramas, with the stored renders
    (no dropout): ``[(set_id, cleanup set, NVS set)]``."""
    points = Dl3dvPoints(_split_root(cfg.val_shards), image_size=tuple(cfg.image_size), use_pose=True,
                         min_pixel_dropout=0.0, max_pixel_dropout=0.0,
                         near_depth=load_near_depth(cfg.near_depth_json))
    panoramas = PanoramaSets(cfg.panorama_root, cfg.panorama_val_manifests,
                             image_size=tuple(cfg.panorama_image_size), use_pose=True, num_views=3,
                             min_pixel_dropout=0.0, max_pixel_dropout=0.0,
                             near_depth=load_near_depth(cfg.panorama_near_depth_json))
    sets = []
    for offset, dataset in enumerate((points, panoramas)):
        for i, index in enumerate(np.linspace(0, len(dataset) - 1, cfg.val_sets).round().astype(int)):
            set_id = offset * cfg.val_sets + i
            if set_id % world != rank:
                continue
            batch = _collate([dataset[int(index)]])
            sets.append((set_id,
                         to_omnicrafter(batch, cfg.scope_patch_size, cfg.sort_refs_by_distance),
                         to_omnicrafter(batch, cfg.scope_patch_size, cfg.sort_refs_by_distance, black_queries=True)))
    return sets


@torch.no_grad()
def validate(module, val_sets, cfg, accelerator):
    """Mean flow-matching loss per camera model and task, at fixed timesteps and
    seeded noise, so curves compare across steps."""
    sums = torch.zeros(2, 2, 2, dtype=torch.float64, device=accelerator.device)
    for set_id, cleanup, nvs in val_sets:
        camera = int(cleanup['camera_model'] == 'equirect')
        for task, data in enumerate((cleanup, nvs)):
            for t_index in cfg.val_t_indices:
                generator = torch.Generator().manual_seed(cfg.seed + 1000 * set_id + int(t_index))
                sums[camera, task, 0] += module.val_loss(data, int(t_index), generator).double()
                sums[camera, task, 1] += 1
    sums = accelerator.reduce(sums, reduction='sum')
    metrics = {}
    for i, camera in enumerate(('pinhole', 'equirect')):
        for j, task in enumerate(('cleanup', 'nvs')):
            if sums[i, j, 1] > 0:
                metrics[f'val/loss_{camera}_{task}'] = (sums[i, j, 0] / sums[i, j, 1]).item()
    return metrics


# ── Checkpoints ───────────────────────────────────────────────────────────────

def trainable_state(module):
    return {n: p for n, p in module.pipe.dit.named_parameters() if p.requires_grad}


def save_adapter(accelerator, module, out, step, cfg, init_sha):
    """bf16 trainable weights, the near-depth run's step-N.safetensors layout.
    ZeRO-2 keeps the full bf16 parameters current on every rank."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {n: p.detach().to('cpu', copy=True).contiguous() for n, p in trainable_state(module).items()}
        destination = out/f'step-{step}.safetensors'
        temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
        save_file(state, str(temporary), metadata={
            'step': str(step), 'camera_injection': 'scope_ff',
            'scope': json.dumps({'mlp_hidden': cfg.scope_mlp_hidden, 'dir_freqs': cfg.scope_dir_freqs,
                                 'patch_size': cfg.scope_patch_size}),
            'projections': 'panorama,perspective',
            'resolutions_hw': f'panorama={cfg.panorama_image_size},perspective={cfg.image_size}',
            'model': 'omnicrafter-2ref3-scope-ff', 'batch_per_gpu': '1', 'global_batch': str(accelerator.num_processes),
            'pose_normalization': 'per_scene_near_depth_relative_to_nearest_ref',
            'nvs_ratio': str(cfg.nvs_ratio), 'initialization_sha256': init_sha})
        temporary.replace(destination)
        print(f'{now()} saved {destination}', flush=True)
    accelerator.wait_for_everyone()


def full_state_checksum(model, module, accelerator):
    """Sums of squares of the bf16 trainable weights, the fp32 master partitions and
    both AdamW moments over all ranks: equal before a save and after its load."""
    weights = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    for p in trainable_state(module).values():
        weights += p.detach().double().square().sum()
    inner = model.optimizer.optimizer
    local = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    for group in inner.param_groups:
        for p in group['params']:
            local[0] += p.detach().double().square().sum()
            state = inner.state.get(p, {})
            for i, name in ((1, 'exp_avg'), (2, 'exp_avg_sq')):
                if name in state:
                    local[i] += state[name].detach().double().square().sum()
    local = accelerator.reduce(local, reduction='sum')
    return {'trainable_weights': weights.item(), 'fp32_master': local[0].item(),
            'exp_avg': local[1].item(), 'exp_avg_sq': local[2].item()}


def save_full_state(accelerator, model, module, out, step, cfg, init_sha):
    """Write ``resume.partial/`` on all ranks, then swap it in for ``resume/`` by rename."""
    partial, current, old = out/'resume.partial', out/'resume', out/'resume.old'
    if accelerator.is_main_process and partial.exists():
        shutil.rmtree(partial)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(partial), exclude_frozen_parameters=True)
    checksum = full_state_checksum(model, module, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        import accelerate
        import deepspeed
        write_json(partial/'meta.json', {
            'time': now(), 'step': step, 'world_size': accelerator.num_processes,
            'fingerprint': cfg.fingerprint(), 'initialization_sha256': init_sha, 'checksum': checksum,
            'deepspeed': deepspeed.__version__, 'accelerate': accelerate.__version__})
        if old.exists():
            shutil.rmtree(old)
        if current.exists():
            current.rename(old)
        partial.rename(current)
        if old.exists():
            shutil.rmtree(old)
        print(f'{now()} full state (weights + AdamW + scheduler + RNG) → {current} (step {step})', flush=True)
    accelerator.wait_for_everyone()


def find_resume(out):
    """``resume/``, or ``resume.old/`` if a swap was cut between its two renames.
    ``meta.json`` is written last, so its presence marks a complete state."""
    for name in ('resume', 'resume.old'):
        if (out/name/'meta.json').is_file():
            return out/name
    return None


def keep_final_state(out, step):
    """``resume_step<step>/``: hard links to ``resume/`` (copies where links fail).
    Saves replace ``resume/`` by rename and never rewrite a file, so the links stay intact."""
    destination = out/f'resume_step{step}'
    if destination.exists():
        return destination
    source, temporary = out/'resume', out/f'resume_step{step}.partial'
    if temporary.exists():
        shutil.rmtree(temporary)
    for path in sorted(source.rglob('*')):
        target = temporary/path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    temporary.rename(destination)
    return destination


# ── Dry run ───────────────────────────────────────────────────────────────────

def dry_run(cfg):
    """Check paths, weights, data and rays on the CPU; build no model."""
    paths, tokenizer = model_paths(cfg.wan_root)
    print(f'Wan: {len(json.loads(paths)[0])} DiT shards + T5/VAE/CLIP, tokenizer {tokenizer}')
    assert sha256(cfg.base_initialization) == PUBLISHED_SHA256, 'base_initialization is not the published FrameCrafter adapter'
    print(f'FrameCrafter initialization {cfg.base_initialization}: sha256 OK')
    loader = build_train_loader(cfg, 0, 1, 0)
    val_sets = build_val_sets(cfg, 0, 1)
    print(f'validation: {len(val_sets)} sets x {len(cfg.val_t_indices)} timesteps x (cleanup, nvs)')
    for source in loader.loaders:
        data = to_omnicrafter(next(iter(source)), cfg.scope_patch_size, cfg.sort_refs_by_distance)
        size = data['input_images'][0].size
        print(f"train batch {data['key']}: {data['camera_model']} {size[0]}x{size[1]}, "
              f"rays {tuple(data['scope_rays'].shape)}, black queries {black_fraction(data):.2f}")
    print('dry run OK')


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dry-run', action='store_true', help='Check inputs on the CPU; no model, no GPU')
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.dry_run:
        dry_run(cfg)
        return

    from accelerate import Accelerator
    from accelerate.utils import DistributedType, set_seed
    torch.set_num_threads(4)
    # DeepSpeed does not own a LambdaLR, so accelerate's default would step it num_processes times per
    # training step and shorten the warmup by that factor. One step per training step, for any world size.
    accelerator = Accelerator(gradient_accumulation_steps=1, step_scheduler_with_optimizer=False)
    assert accelerator.distributed_type == DistributedType.DEEPSPEED
    # One set per GPU; the 50k run uses 4 GPUs (global batch 4), a smoke test may use fewer.
    assert accelerator.mixed_precision == 'bf16'
    is_main, rank = accelerator.is_main_process, accelerator.process_index
    out = Path(cfg.output_dir)

    success = out/'_SUCCESS'
    if success.is_file() and read_json(success)['step'] == cfg.max_steps:
        if is_main:
            print(f'Complete: step {cfg.max_steps} in {out}', flush=True)
        return
    init = out/INITIALIZATION
    if is_main:
        for name in ('reports', 'runtime', 'initialization'):
            (out/name).mkdir(parents=True, exist_ok=True)
        if not init.exists():
            from scope_initialization import prepare_scope_initialization
            prepare_scope_initialization(cfg.base_initialization, init)
        cfg.save_yaml(out/'config.yaml')
    accelerator.wait_for_everyone()
    init_sha = sha256(init)

    resume_dir = find_resume(out)
    meta = read_json(resume_dir/'meta.json') if resume_dir else None
    if meta:
        assert meta['world_size'] == accelerator.num_processes, f"resume needs {meta['world_size']} GPUs"
        assert meta['fingerprint'] == cfg.fingerprint(), 'scope-ff / LoRA settings differ from the saved state'
        assert meta['initialization_sha256'] == init_sha
        if meta['step'] > cfg.max_steps:
            raise ValueError(f"saved state is at step {meta['step']}, past max_steps={cfg.max_steps}")
        if meta['step'] == cfg.max_steps:
            # Stopped after the final save but before _SUCCESS.
            if is_main:
                kept = keep_final_state(out, cfg.max_steps)
                write_json(success, {'time': now(), 'complete': True, 'step': cfg.max_steps,
                                     'final_full_state': str(kept)})
                print(f'Complete: step {cfg.max_steps} in {out}', flush=True)
            accelerator.wait_for_everyone()
            return
    start_step = meta['step'] if meta else 0

    set_seed(cfg.seed, device_specific=True)
    train_loader = build_train_loader(cfg, rank, accelerator.num_processes, start_step)
    val_sets = build_val_sets(cfg, rank, accelerator.num_processes)

    paths, tokenizer = model_paths(cfg.wan_root)
    # One rank at a time builds the 14B pipeline on the CPU (host memory).
    with (out/'runtime/model_init.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f'{now()} rank={rank} CPU initialization', flush=True)
        module = ScopeFFTrainingModule(
            model_paths=paths, tokenizer_path=tokenizer,
            scope_kwargs=dict(mlp_hidden=cfg.scope_mlp_hidden, dir_freqs=cfg.scope_dir_freqs,
                              patch_size=cfg.scope_patch_size),
            lora_base_model='dit', lora_target_modules=cfg.lora_target_modules, lora_rank=cfg.lora_rank,
            lora_checkpoint=str(init), device='cpu', modify_channels=True, new_in_dim=36,
            individual_encoding=True, extra_inputs='conditioning_images',
            use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
        source = load_file(str(init), device='cpu')
        state = module.pipe.dit.state_dict()
        assert all(torch.equal(state[n], v.to(state[n].dtype)) for n, v in source.items()), \
            'FrameCrafter initialization was not loaded exactly'
        del source, state
        names = {n: p.numel() for n, p in trainable_state(module).items()}
        assert len(module.pipe.dit.blocks) == NUM_BLOCKS
        assert sum('lora_' in n for n in names) == LORA_TENSORS
        assert sum(n.startswith('patch_embedding.') for n in names) == PATCH_TENSORS
        assert sum('.scope_pe.' in n for n in names) == SCOPE_TENSORS_PER_BLOCK * NUM_BLOCKS
        assert len(names) == LORA_TENSORS + PATCH_TENSORS + SCOPE_TENSORS_PER_BLOCK * NUM_BLOCKS
        if is_main:
            write_json(out/'reports/trainable_parameters.json', {'tensors': names, 'total': sum(names.values())})
            print(f'{now()} trainable: {len(names)} tensors, {sum(names.values()):,} parameters', flush=True)
        encoder_report = release_frozen_encoders(module.pipe, accelerator.device)
        print(f'{now()} rank={rank} frozen encoders: {json.dumps(encoder_report)}', flush=True)

    optimizer = torch.optim.AdamW(module.trainable_modules(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup = max(1, cfg.warmup_steps)
    # Optimizer step k (1-based) uses min(1, k / warmup) of the LR, as flux3's _lr.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda done: min(1.0, (done + 1) / warmup))
    model, optimizer, scheduler = accelerator.prepare(module, optimizer, scheduler)
    module = accelerator.unwrap_model(model)
    module.pipe.dit.train()
    assert module.pipe.text_encoder is None and module.pipe.image_encoder is None
    assert all(block.self_attn.scope_pe.training for block in module.pipe.dit.blocks)
    module.pipe.vae.eval()
    assert not any(p.requires_grad for p in module.pipe.vae.parameters())
    zero = accelerator.deepspeed_config['zero_optimization']
    assert zero['stage'] == 2 and zero.get('offload_optimizer', {}).get('device', 'none') == 'none'
    assert model.train_micro_batch_size_per_gpu() == 1 and model.train_batch_size() == accelerator.num_processes

    if meta:
        accelerator.load_state(str(resume_dir), load_module_strict=False)
        checksum = full_state_checksum(model, module, accelerator)
        for key, value in meta['checksum'].items():
            assert math.isclose(checksum[key], value, rel_tol=1e-9), f'{key}: loaded {checksum[key]} != saved {value}'
        assert model.global_steps == start_step, f'DeepSpeed is at step {model.global_steps}, meta at {start_step}'
        if is_main:
            print(f'{now()} resumed full state (weights + AdamW + scheduler + RNG) from {resume_dir} '
                  f'at step {start_step}, checksum OK, lr {model.get_lr()[0]:.3g}', flush=True)

    run = None
    if is_main and cfg.use_wandb:
        import wandb
        id_file = out/'runtime/wandb_id'
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:8])
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name or out.name, id=id_file.read_text().strip(),
                         resume='allow', dir=str(out), config=asdict(cfg),
                         mode=os.environ.get('WANDB_MODE', 'offline'))

    def log(metrics, step):
        if is_main:
            with (out/'reports/train_log.jsonl').open('a') as handle:
                handle.write(json.dumps({'time': now(), 'step': step, **metrics}, allow_nan=False) + '\n')
            if run is not None:
                run.log(metrics, step=step)

    if start_step == 0:
        log(validate(module, val_sets, cfg, accelerator), 0)
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    if is_main:
        print(f'{now()} training steps {start_step}..{cfg.max_steps}, world {accelerator.num_processes}', flush=True)
    # After any resume: the restored RNG seeds a fresh shuffle.
    train_iter = iter(train_loader)
    interval = {'loss': [], 'pinhole': [], 'equirect': [], 'black': [], 'seconds': []}
    step, stop = start_step, False
    while step < cfg.max_steps:
        tick = time.monotonic()
        data = to_omnicrafter(next(train_iter), cfg.scope_patch_size, cfg.sort_refs_by_distance)
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss = model([data])
            assert torch.isfinite(loss).all(), 'Non-finite loss'
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
        step += 1
        assert model.global_steps == step
        over = cfg.time_limit_hours is not None and time.time() - _PROCESS_START >= cfg.time_limit_hours * 3600
        rows = accelerator.gather(torch.tensor(
            [[loss.detach().float().item(), float(data['camera_model'] == 'equirect'), black_fraction(data), float(over)]],
            device=accelerator.device)).cpu()
        stop = bool(rows[:, 3].any())
        interval['loss'] += rows[:, 0].tolist()
        interval['black'] += rows[:, 2].tolist()
        for value, equirect in rows[:, :2].tolist():
            interval['equirect' if equirect else 'pinhole'].append(value)
        interval['seconds'].append(time.monotonic() - tick)
        if is_main:
            with (out/'reports/train_loss.jsonl').open('a') as handle:
                handle.write(json.dumps({'step': step, 'loss': rows[:, 0].mean().item(),
                                         'rank_losses': rows[:, 0].tolist(), 'rank_equirect': rows[:, 1].tolist(),
                                         'step_seconds': interval['seconds'][-1]}) + '\n')
        if step % cfg.log_interval == 0 or step <= start_step + 3:
            metrics = {'train/loss': float(np.mean(interval['loss'])), 'lr': model.get_lr()[0],
                       'train/black_query_fraction': float(np.mean(interval['black'])),
                       'perf/steps_per_sec': 1.0 / float(np.mean(interval['seconds'])),
                       'perf/peak_mem_gib': torch.cuda.max_memory_allocated(accelerator.device) / 2**30}
            for camera in ('pinhole', 'equirect'):
                if interval[camera]:
                    metrics[f'train/loss_{camera}'] = float(np.mean(interval[camera]))
            log(metrics, step)
            if is_main:
                print(f'{now()} step {step} ' + ' '.join(f'{k}={v:.4g}' for k, v in metrics.items()), flush=True)
            interval = {k: [] for k in interval}
        if step % cfg.val_interval == 0:
            metrics = validate(module, val_sets, cfg, accelerator)
            log(metrics, step)
            if is_main:
                print(f'{now()} step {step} ' + ' '.join(f'{k}={v:.4g}' for k, v in metrics.items()), flush=True)
        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_adapter(accelerator, module, out, step, cfg, init_sha)
        if step % cfg.resume_interval == 0 or step == cfg.max_steps or stop:
            save_full_state(accelerator, model, module, out, step, cfg, init_sha)
        if stop:
            if is_main:
                print(f'{now()} time_limit_hours={cfg.time_limit_hours} reached at step {step}; '
                      'resubmit to continue', flush=True)
            break

    if step == cfg.max_steps and is_main:
        kept = keep_final_state(out, step)
        write_json(success, {'time': now(), 'complete': True, 'step': step, 'final_full_state': str(kept)})
        print(f'{now()} complete at step {step}; final full state kept in {kept}', flush=True)
    accelerator.wait_for_everyone()
    if run is not None:
        run.finish()
    accelerator.end_training()


if __name__ == '__main__':
    main()
