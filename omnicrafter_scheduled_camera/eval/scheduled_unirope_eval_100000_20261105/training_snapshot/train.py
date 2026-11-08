"""Two separate scheduled concat continuations, four GPUs each."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from contract import CAMERA_CONTRACT, verify_vendor
from dataset import MixedProjectionDataset, collate
from checkpoints import prepare_initialization, save_adapter, sha256
from training_model import MixedTrainingModule
from wan_camera import optimizer_groups, depth_statistics
from config import BASE_STEP, RAMP_STEPS, VARIANT, injection_weight


def write_status(output, status):
    temporary = output / f'.progress.{os.getpid()}.tmp'
    temporary.write_text(json.dumps({'time': datetime.now().astimezone().isoformat(), **status}, indent=2)+'\n')
    temporary.replace(output/'progress.json')


def zero2_config():
    return {'bf16': {'enabled': True}, 'fp16': {'enabled': False},
            'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
            'train_batch_size': 'auto', 'gradient_clipping': 1.,
            'zero_allow_untested_optimizer': True,
            'zero_optimization': {'stage': 2, 'offload_optimizer': {'device': 'none'},
                'offload_param': {'device': 'none'}, 'reduce_scatter': True,
                'contiguous_gradients': True, 'overlap_comm': False,
                'allgather_partitions': True, 'reduce_bucket_size': 10000000,
                'allgather_bucket_size': 10000000}}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wan', required=True, type=Path, help='Directory containing Wan-AI/')
    parser.add_argument('--initialization-checkpoint', required=True, type=Path)
    parser.add_argument('--ramp-steps', default=RAMP_STEPS, type=int)
    parser.add_argument('--manifest', required=True, type=Path, help='Existing mixed train.jsonl with near_depth per row')
    parser.add_argument('--offsets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New experiment directory, never an old camera run')
    parser.add_argument('--max-steps', required=True, type=int, help='Absolute final optimizer step, not extra steps')
    parser.add_argument('--save-every', default=2000, type=int)
    parser.add_argument('--restore-checkpoint', type=Path)
    parser.add_argument('--nvs-probability', default=.5, type=float, help='Per SAMPLE; all three queries black together')
    parser.add_argument('--seed', default=20261023, type=int)
    parser.add_argument('--path-map', type=Path, help='Optional JSON old image root -> new root; no manifest rewrite')
    return parser.parse_args()


def wan_paths(root):
    model = root / 'Wan-AI/Wan2.1-I2V-14B-480P'
    shards = sorted(model.glob('diffusion_pytorch_model-*-of-00007.safetensors'))
    pieces = [model / name for name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth',
                                       'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    tokenizer = root / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'
    if len(shards) != 7 or not all(p.is_file() for p in shards + pieces) or not (tokenizer/'tokenizer_config.json').is_file():
        raise FileNotFoundError('Incomplete WAN directory or tokenizer')
    return json.dumps([shards] + pieces, default=str), str(tokenizer)


def main():
    args = arguments()
    if args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError('Step counts must be positive')
    verify_vendor()
    torch.set_num_threads(4)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=zero2_config()))
    if accelerator.num_processes != 4 or args.ramp_steps <= 0:
        raise ValueError('Require four GPUs and a positive ramp length')
    set_seed(args.seed, device_specific=True)
    args.output.mkdir(parents=True, exist_ok=True)
    # One job per output. Lock persists for the life of rank zero.
    run_lock = None
    if accelerator.is_main_process:
        run_lock = (args.output/'.run.lock').open('a')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list(args.output.glob('step-*.safetensors')) and not args.restore_checkpoint:
            raise ValueError('Output already has checkpoints; specify a matching restore or use a new output')
        write_status(args.output, {'stage':'preparing_initialization','target_step':args.max_steps})
        prepare_initialization(args.initialization_checkpoint, args.output/'initialization/concat_step50000_full420.safetensors')
    accelerator.wait_for_everyone()
    paths, tokenizer = wan_paths(args.wan)
    data = MixedProjectionDataset(args.manifest, args.offsets, args.nvs_probability,
                                  json.loads(args.path_map.read_text()) if args.path_map else None)
    data.prepare_cleanup_groups()
    loader = DataLoader(data, batch_size=1, shuffle=True, drop_last=True, num_workers=0,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    # Serialize CPU loading to avoid concurrent eight-copy host RAM peaks.
    with (args.output/'.initialization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f'{datetime.now().astimezone().isoformat()} rank={accelerator.process_index} CPU model initialization',flush=True)
        if accelerator.is_main_process:
            write_status(args.output, {'stage':'loading_WAN','target_step':args.max_steps})
        module = MixedTrainingModule(model_paths=paths, tokenizer_path=tokenizer,
            restore_checkpoint=args.restore_checkpoint, ramp_steps=args.ramp_steps, lora_base_model='dit',
            lora_target_modules='q,k,v,o,ffn.0,ffn.2', lora_rank=32,
            lora_checkpoint=str(args.output/'initialization/concat_step50000_full420.safetensors'),
            device='cpu', modify_channels=True, new_in_dim=420, individual_encoding=True,
            extra_inputs='conditioning_images', use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False)
        print(f'{datetime.now().astimezone().isoformat()} rank={accelerator.process_index} CPU model loaded',flush=True)
    step = module.initial_step
    if not 0 <= step < args.max_steps:
        raise ValueError('Final step must exceed the restored checkpoint step')
    groups = optimizer_groups(module.pipe.dit)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    protocol = {'camera': CAMERA_CONTRACT, 'manifest': str(args.manifest), 'sets': len(data),
                'batch_per_gpu': 1, 'global_batch': accelerator.num_processes,
                'nvs_probability': args.nvs_probability, 'task_sampling': 'per_sample_all_three_black',
                'cleanup_pixel_dropout': [0, 100], 'main_lr': 1e-5,
                'cleanup_all_black_guard': 'redraw_dropout_keep_at_least_one_nonempty_query',
                'empty_cleanup_source_policy': 'resample_entire_train_set_same_scene_first_else_same_dataset_projection_resolution',
                'empty_cleanup_source_search_limits': {'same_scene': 64, 'same_dataset': 256},
                'manifest_modified': False,
                'initialization': str(args.initialization_checkpoint),
                'initialization_sha256': sha256(args.initialization_checkpoint) if accelerator.is_main_process else None,
                'base_step': BASE_STEP, 'ramp_steps': args.ramp_steps,
                'injection_schedule': 'linear 0 to 1 over first ramp_steps additional updates; positive gate on first update',
                'inherited_parameters': 'standard LoRA and patch embedding trainable throughout; no extra backbone unfreezing',
                'depth_lr': 4e-4 if VARIANT=='unirope' else None,
                'depth_weight_decay': .5 if VARIANT=='unirope' else None, 'lr_schedule': 'constant',
                'zero_stage': 2, 'optimizer_device': 'cuda', 'cpu_offload': False,
                'resolutions_hw': {'panorama': [256, 512], 'perspective': [288, 512]},
                'initial_step': step, 'max_steps': args.max_steps, 'save_every': args.save_every,
                'restore_checkpoint': str(args.restore_checkpoint) if args.restore_checkpoint else None}
    if accelerator.is_main_process:
        protocol['manifest_sha256'] = sha256(args.manifest)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        (args.output/f'protocol_{stamp}.json').write_text(json.dumps(protocol, indent=2)+'\n')
        print(json.dumps(protocol), flush=True)
        print(f'{VARIANT}: new-camera LR={groups[1]["lr"]}; GPU AdamW, no CPU offload', flush=True)
        names = {n:p.numel() for n,p in module.pipe.dit.named_parameters() if p.requires_grad}
        if not all('lora_' in n or n.startswith('patch_embedding.') or '.scope_pe.' in n or '.unirope_depth.' in n for n in names):
            raise RuntimeError('Unexpected trainable backbone parameters')
        (args.output/'trainable_parameters.json').write_text(json.dumps({'total':sum(names.values()),'tensors':names},indent=2)+'\n')
        write_status(args.output, {'stage':'preparing_GPU_optimizer','target_step':args.max_steps})
    model, optimizer, loader, scheduler = accelerator.prepare(module, optimizer, loader, scheduler)
    module = accelerator.unwrap_model(model)
    # Full 14B frozen-weight forward parity, both real grids and both task
    # types, before the first optimizer update. This is a correctness check,
    # not a pilot training run or benchmark.
    if step == BASE_STEP:
        from verify_full_model import verify_full_model
        verify_full_model(module, data, accelerator, args.output)
    for name in ('vae', 'text_encoder', 'image_encoder'):
        encoder = getattr(module.pipe, name)
        encoder.eval()
        if any(p.requires_grad for p in encoder.parameters()):
            raise RuntimeError(f'Unexpected trainable frozen encoder: {name}')
        if any(p.device.type != 'cuda' for p in encoder.parameters()):
            raise RuntimeError(f'Encoder unexpectedly offloaded: {name}')
    if module.pipe.vram_management_enabled:
        raise RuntimeError('Encoder offloading is disabled for this run')
    effective = accelerator.deepspeed_config['zero_optimization']
    if effective['stage'] != 2 or any(effective.get(k, {}).get('device', 'none') != 'none'
                                     for k in ('offload_optimizer', 'offload_param')):
        raise RuntimeError('Expected ZeRO2 with GPU optimization')
    epoch = 0
    counts = torch.zeros(11,device=accelerator.device,dtype=torch.float64)
    while step < args.max_steps:
        loader.set_epoch(epoch)
        for batch in loader:
            start = time.monotonic()
            module.pipe.dit.injection_strength = injection_weight(step + 1, args.ramp_steps)
            row=batch[0]
            counts += torch.tensor([1,int(row['sample_task_type']=='nvs'),int(row['dataset']=='DL3DV'),
                int(row['dataset']=='HM3D'),int(row['dataset']=='JHU'),sum(v==1 for v in row['query_pixel_dropout_ratios']),
                int(row['all_queries_black']), row['cleanup_dropout_redraws'],
                int(row['cleanup_source_resampled']), int(row['cleanup_scene_changed']),
                row['cleanup_resample_attempts']],
                device=accelerator.device)
            optimizer.zero_grad()
            loss = model(batch)
            if not torch.isfinite(loss).all():
                raise RuntimeError('Non-finite training loss')
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            step += 1
            if model.global_steps != step-module.initial_step:
                raise RuntimeError('Engine and optimizer step counts diverged')
            if step == module.initial_step + 1:
                states = [v for s in model.optimizer.optimizer.state.values() for k, v in s.items()
                          if k in ('exp_avg', 'exp_avg_sq')]
                if not states or not all(v.is_cuda for v in states):
                    raise RuntimeError('AdamW state is not on GPU')
                (args.output/f'gpu_optimizer_rank{accelerator.process_index}.json').write_text(json.dumps({
                    'rank':accelerator.process_index,'optimizer_state_device':'cuda','zero_stage':2,
                    'batch_per_gpu':model.train_micro_batch_size_per_gpu(),'global_batch':model.train_batch_size(),
                    'trainable_tensors':sum(p.requires_grad for p in module.pipe.dit.parameters())},indent=2)+'\n')
            value = accelerator.reduce(loss.detach().float(), reduction='mean').item()
            if step % 10 == 0 or step == module.initial_step + 1:
                totals=accelerator.reduce(counts,reduction='sum').tolist()
                record = {'time': datetime.now().astimezone().isoformat(), 'step': step,
                          'loss': value, 'seconds': time.monotonic()-start,
                          'rank0_dataset':row['dataset'],'rank0_task':row['sample_task_type'],
                          'nvs_fraction':totals[1]/totals[0],'perspective_fraction':totals[2]/totals[0],
                          'all_three_black_fraction':totals[6]/totals[0],
                          'cleanup_dropout_redraws_total':totals[7],
                          'cleanup_source_resampled_total':totals[8],
                          'cleanup_scene_changed_total':totals[9],
                          'cleanup_resample_attempts_total':totals[10],
                          'initial_step':module.initial_step,
                          'additional_updates':step-BASE_STEP,
                          'injection_weight':module.pipe.dit.injection_strength,
                          'hm3d_fraction':totals[3]/totals[0],'jhu_fraction':totals[4]/totals[0],
                          'allocated_GiB':torch.cuda.memory_allocated()/1024**3,
                          'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                          **depth_statistics(module.pipe.dit)}
                if accelerator.is_main_process:
                    print(json.dumps(record), flush=True)
                    with (args.output/'train_loss.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    write_status(args.output,{**record,'stage':'training','target_step':args.max_steps})
            if step % args.save_every == 0 or step == args.max_steps:
                if accelerator.is_main_process:
                    print(f'{datetime.now().astimezone().isoformat()} saving step {step}',flush=True)
                save_adapter(accelerator, model, args.output/f'step-{step}.safetensors', step, protocol)
                if accelerator.is_main_process:
                    print(f'{datetime.now().astimezone().isoformat()} saved {args.output}/step-{step}.safetensors',flush=True)
            if step == args.max_steps:
                break
        epoch += 1
    if accelerator.is_main_process:
        write_status(args.output, {'stage':'complete','step':step,'target_step':args.max_steps})
    accelerator.end_training()
    if run_lock is not None:
        run_lock.close()


if __name__ == '__main__':
    main()
