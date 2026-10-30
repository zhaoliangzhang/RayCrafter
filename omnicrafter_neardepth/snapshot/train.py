"""Four-GPU ZeRO-2, GPU AdamW, near-depth-normalized mixed-projection SCoPE."""
import argparse
import fcntl
import json
import time
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from common import *
from mixed_dataset import MixedProjectionDataset
from mixed_model import MixedTrainingModule
from frozen_encoders import release_frozen_encoders
from runtime_guard import LocalTrace, IdleMemoryReservation, event


def collate(rows):
    return rows


def save_adapter(accelerator,model,step,architecture):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        module = accelerator.unwrap_model(model)
        state = module.export_trainable_state_dict(accelerator.get_state_dict(model),remove_prefix='pipe.dit.')
        expected = {n for n,p in module.pipe.dit.named_parameters() if p.requires_grad}
        assert set(state) == expected
        assert len(state) == (1202 if architecture == 'scope' else 802)
        state = {n:v.detach().cpu().contiguous() for n,v in state.items()}
        destination = OUT/f'step-{step}.safetensors'
        assert not destination.exists(), f'Refusing to overwrite {destination}'
        temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
        save_file(state,str(temporary),metadata={'step':str(step),'camera_injection':architecture,
            'projections':'panorama,perspective','resolutions_wh':'panorama=512x256,perspective=512x288',
            'model':'mixed-projection-2ref3','batch_per_gpu':'1','global_batch':str(GLOBAL_BATCH),
            'frozen_encoder_policy':FROZEN_ENCODER_POLICY,
            'pose_normalization':POSE_NORMALIZATION,
            'near_depth_source_sha256':json.dumps(json.loads((DATA/'summary.json').read_text())['near_depth_source_sha256'],sort_keys=True),
            'nvs_probability':'0.5','manifest_sha256':json.loads((DATA/'summary.json').read_text())['sha256']})
        temporary.replace(destination)
        print(f'{now()} saved {destination}',flush=True)
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wan',required=True)
    parser.add_argument('--architecture',choices=['scope','concat'],required=True)
    parser.add_argument('--base-initialization',required=True)
    parser.add_argument('--restore-checkpoint')
    parser.add_argument('--initial-step',type=int,default=0)
    parser.add_argument('--reserve-gpu-headroom-gib',type=int,default=0,
        help='Opt-in idle allocation after 100 warmup steps; leave at least 16 GiB free')
    args = parser.parse_args()
    assert 0 <= args.initial_step < FINAL_STEP
    assert args.reserve_gpu_headroom_gib == 0, 'No memory parking for this experiment'
    torch.set_num_threads(4)
    accelerator = Accelerator(gradient_accumulation_steps=1)
    assert accelerator.distributed_type == DistributedType.DEEPSPEED
    assert accelerator.num_processes == WORLD_SIZE and accelerator.mixed_precision == 'bf16'
    trace = LocalTrace(os.environ['OC_LOCAL_RUNTIME'],accelerator.process_index) if os.environ.get('OC_LOCAL_RUNTIME') else None
    event('initialization',step=args.initial_step)
    set_seed(SEED,device_specific=True)
    dataset = MixedProjectionDataset(args.architecture)
    assert len(dataset) == 200000
    loader = DataLoader(dataset,batch_size=1,shuffle=True,drop_last=True,collate_fn=collate,
        num_workers=0,generator=torch.Generator().manual_seed(SEED))
    paths,tokenizer = model_paths(args.wan)
    with (OUT/'runtime/model_init.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(f'{now()} rank={accelerator.process_index} CPU initialization',flush=True)
        module = MixedTrainingModule(model_paths=paths,tokenizer_path=tokenizer,
            architecture=args.architecture,restore_checkpoint=args.restore_checkpoint,
            lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=32,
            lora_checkpoint=args.base_initialization,device='cpu',modify_channels=True,
            new_in_dim=36 if args.architecture=='scope' else 420,individual_encoding=True,
            extra_inputs='conditioning_images',use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False)
        source = load_file(args.restore_checkpoint or args.base_initialization,device='cpu')
        state = module.pipe.dit.state_dict()
        assert all(torch.equal(state[n],v.to(state[n].dtype)) for n,v in source.items())
        del source,state
        names = {n:p.numel() for n,p in module.pipe.dit.named_parameters() if p.requires_grad}
        assert len(names) == (1202 if args.architecture=='scope' else 802)
        assert all('lora_' in n or n.startswith('patch_embedding.') or '.scope_pe.' in n for n in names)
        if accelerator.is_main_process:
            write_json(OUT/'reports/trainable_parameters.json',{'tensors':names,'total':sum(names.values())})
        event('cache_empty_prompt_release_encoders')
        encoder_report = release_frozen_encoders(module.pipe, accelerator.device)
        write_json(OUT/f'runtime/frozen_encoders_rank{accelerator.process_index}.json',
                   {'time':now(), **encoder_report})
        print(f'{now()} rank={accelerator.process_index} frozen encoder optimization: {json.dumps(encoder_report)}',flush=True)
    optimizer = torch.optim.AdamW(module.trainable_modules(),lr=1e-5,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model,optimizer,loader,scheduler = accelerator.prepare(module,optimizer,loader,scheduler)
    module = accelerator.unwrap_model(model)
    assert module.pipe.text_encoder is None and module.pipe.image_encoder is None
    assert module.pipe.empty_prompt_context.is_cuda and not module.pipe.empty_prompt_context.requires_grad
    for name in ('vae',):
        encoder = getattr(module.pipe,name)
        encoder.eval()
        assert not any(p.requires_grad for p in encoder.parameters())
    zero = accelerator.deepspeed_config['zero_optimization']
    assert zero['stage'] == 2
    assert zero.get('offload_optimizer',{}).get('device','none') == 'none'
    assert zero.get('offload_param',{}).get('device','none') == 'none'
    assert model.train_micro_batch_size_per_gpu() == BATCH and model.train_batch_size() == GLOBAL_BATCH
    assert model.gradient_accumulation_steps() == 1 and model.global_steps == 0
    if accelerator.is_main_process:
        write_json(OUT/'runtime/effective_deepspeed_config.json',config_for_json(accelerator.deepspeed_config))
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    reservation = IdleMemoryReservation(torch,accelerator.device,args.reserve_gpu_headroom_gib) if args.reserve_gpu_headroom_gib else None
    start_time = time.monotonic()
    step,epoch = args.initial_step,0
    # Counts: all sets, NVS sets, DL3DV sets, HM3D sets, JHU sets, black queries.
    cumulative = torch.zeros(6,dtype=torch.float64,device=accelerator.device)
    initial_patch = module.pipe.dit.patch_embedding.weight.detach().flatten()[:128].clone()
    initial_scope = module.pipe.dit.blocks[0].self_attn.scope_pe.eq.weight.detach().clone() if args.architecture=='scope' else None
    seen_cases = set()
    while step < FINAL_STEP:
        loader.set_epoch(epoch)
        event('awaiting_batch',step=step+1,epoch=epoch)
        if trace:
            trace.watchdog()
        for batch in loader:
            tick = time.monotonic()
            assert len(batch) == 1
            row = batch[0]
            event('batch_ready',step=step+1,sample_id=row['sample_id'],projection=row['projection'])
            seen_cases.add((row['projection'],row['sample_task_type']))
            cumulative += torch.tensor([1,int(row['sample_task_type']=='nvs'),int(row['dataset']=='DL3DV'),
                int(row['dataset']=='HM3D'),int(row['dataset']=='JHU'),
                sum(v==1 for v in row['query_pixel_dropout_ratios'])],device=accelerator.device)
            previous = model.global_steps
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                event('forward')
                loss = model(batch)
                assert torch.isfinite(loss).all(),'Non-finite loss'
                event('backward')
                accelerator.backward(loss)
                event('optimizer')
                optimizer.step()
                scheduler.step()
            step += 1
            assert model.global_steps == previous+1 == step-args.initial_step
            event('loss_allreduce',step=step)
            loss_value = accelerator.reduce(loss.detach().float(),reduction='mean').item()
            record = {'time':now(),'step':step,'epoch':epoch,'loss':loss_value,
                'step_seconds':time.monotonic()-tick,'rank0_dataset':row['dataset'],
                'rank0_projection':row['projection'],'rank0_task':row['sample_task_type'],
                'batch_size_per_gpu':BATCH,'global_batch_size':GLOBAL_BATCH,'engine_steps_since_launch':model.global_steps}
            event('shared_progress_write')
            if accelerator.is_main_process:
                with (OUT/'reports/train_loss.jsonl').open('a') as handle:
                    handle.write(json.dumps(record,allow_nan=False)+'\n')
                if step <= args.initial_step+3 or step%10 == 0:
                    print(json.dumps(record),flush=True)
            if step <= args.initial_step+3 or step%100 == 0:
                counts = accelerator.reduce(cumulative,reduction='sum').tolist()
                states = [v for s in model.optimizer.optimizer.state.values() for k,v in s.items()
                          if k in ('exp_avg','exp_avg_sq') and isinstance(v,torch.Tensor)]
                assert states and all(v.is_cuda for v in states)
                assert module.last_batch_shapes['input_latents'][2:] == [5,32 if row['projection']=='panorama' else 36,64]
                write_json(OUT/f'runtime/rank{accelerator.process_index}.json',{
                    'time':now(),'step':step,'optimizer_state_device':'cuda','batch_shapes':module.last_batch_shapes,
                    'projection':row['projection'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'idle_reservation_bytes':reservation.bytes if reservation else 0,
                    'frozen_encoder_policy':FROZEN_ENCODER_POLICY,
                    'seen_projection_task_cases':sorted(seen_cases)})
                if accelerator.is_main_process:
                    write_json(OUT/'runtime/progress.json',{**record,'target_step':FINAL_STEP,'stage':'training',
                        'elapsed_seconds':time.monotonic()-start_time,'sets_seen_since_launch':int(counts[0]),
                        'nvs_sample_fraction':counts[1]/counts[0],'perspective_fraction':counts[2]/counts[0],
                        'hm3d_fraction':counts[3]/counts[0],'jhu_fraction':counts[4]/counts[0],
                        'black_query_fraction':counts[5]/(3*counts[0])})
            if step == args.initial_step+3:
                assert not torch.equal(initial_patch,module.pipe.dit.patch_embedding.weight.detach().flatten()[:128])
                if initial_scope is not None:
                    assert not torch.equal(initial_scope,module.pipe.dit.blocks[0].self_attn.scope_pe.eq.weight.detach())
            if step == args.initial_step+100:
                assert seen_cases == {('panorama','nvs'),('panorama','cleanup'),('perspective','nvs'),('perspective','cleanup')}
                if reservation:
                    event('idle_memory_reservation')
                    accelerator.wait_for_everyone()
                    result = reservation.acquire(GPU_UUIDS[accelerator.process_index])
                    event('idle_memory_reservation_result',reservation=result)
                    write_json(OUT/f'runtime/reservation_rank{accelerator.process_index}.json',
                        {'time':now(),'step':step,'pid':os.getpid(),**result})
                    print(f'{now()} rank={accelerator.process_index} memory reservation: {json.dumps(result)}',flush=True)
                    accelerator.wait_for_everyone()
                write_json(OUT/f'runtime/smoke_rank{accelerator.process_index}.json',{
                    'time':now(),'passed':True,'step':step,'optimizer_device':'cuda',
                    'cases':sorted(seen_cases),'projection_specific_resolutions':True})
            if reservation and step%100 == 0:
                released = reservation.release_under_pressure()
                if released:
                    event('idle_memory_released',reservation=released)
                    print(f'{now()} rank={accelerator.process_index} memory reservation: {json.dumps(released)}',flush=True)
            if step % SAVE_STEPS == 0:
                event('checkpoint_save')
                save_adapter(accelerator,model,step,args.architecture)
            event('step_complete',step=step)
            if trace:
                trace.watchdog()
            if step == FINAL_STEP:
                break
            event('awaiting_batch',step=step+1)
        epoch += 1
    if accelerator.is_main_process:
        write_json(OUT/'_SUCCESS',{'time':now(),'complete':True,'step':step})
    if reservation:
        reservation.release()
    if trace:
        trace.close()
    accelerator.end_training()


if __name__ == '__main__':
    main()
