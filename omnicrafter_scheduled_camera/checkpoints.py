"""Strict concat warm-start; isolated scheduled adapters with resume metadata.

The full training state (``resume/``) lets a run continue across Slurm jobs as
if it had not stopped: DeepSpeed weights and AdamW, scheduler, every RNG, and
each rank's position in the shuffled data.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from config import BASE_STEP, VARIANT, injection_weight, schedule_type
from contract import contract_json, check_contract

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()

def prepare_initialization(source, destination):
    with safe_open(str(source),framework='pt',device='cpu') as ckpt:
        metadata = ckpt.metadata()
        if metadata.get('camera_injection') != 'concat' or int(metadata['step']) != BASE_STEP:
            raise ValueError('Require the trained concat-only step-50000 checkpoint, NOT published initialization')
        if len(ckpt.keys()) != 802 or ckpt.get_slice('patch_embedding.weight').get_shape() != [5120,420,1,2,2]:
            raise ValueError('Unexpected concat adapter shape')
    state = load_file(str(source),device='cpu')
    destination = Path(destination)
    if destination.exists():
        existing = load_file(str(destination),device='cpu')
        if set(state) != set(existing) or not all(torch.equal(existing[k],v) for k,v in state.items()):
            raise ValueError('Existing initialization differs')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        save_atomic(state,destination,dict(metadata,source_sha256=sha256(source)))
    return destination

def save_atomic(state,destination,metadata):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(temporary),metadata=metadata)
    os.link(temporary,destination)
    temporary.unlink()

def warm_start(dit,path):
    state = load_file(str(path),device='cpu')
    params = dict(dit.named_parameters())
    inherited = {n for n,p in params.items() if p.requires_grad and '.scope_pe.' not in n and '.rayrope_depth.' not in n}
    if len(inherited) != 802 or set(state) != inherited:
        raise ValueError(f'Warm-start tensor mismatch: missing={inherited-set(state)}, extra={set(state)-inherited}')
    if any(state[n].shape != params[n].shape for n in inherited):
        raise ValueError('Warm-start shape mismatch')
    result = dit.load_state_dict(state,strict=False)
    if result.unexpected_keys or inherited.intersection(result.missing_keys):
        raise ValueError('Incomplete inherited-weight load')
    if not all(torch.equal(params[n],v.to(params[n])) for n,v in state.items()):
        raise ValueError('Inherited parameters differ after warm-start')
    print(f'Exactly restored all {len(inherited)} concat-only tensors; new {VARIANT} initialized separately',flush=True)

def restore_adapter(dit,path,ramp_steps):
    with safe_open(str(path),framework='pt',device='cpu') as ckpt:
        metadata = ckpt.metadata()
    check_contract(metadata)
    step = int(metadata['step'])
    schedule = json.loads(metadata['injection_schedule'])
    if schedule != {'base_step':BASE_STEP,'ramp_steps':ramp_steps,'type':schedule_type(ramp_steps),
                    'completed_updates':step-BASE_STEP,'weight':injection_weight(step,ramp_steps)}:
        raise ValueError('Resume schedule/step mismatch')
    state = load_file(str(path),device='cpu')
    expected = {n for n,p in dit.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Resume tensor keys do not match this variant')
    result = dit.load_state_dict(state,strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):
        raise ValueError('Incomplete scheduled adapter load')
    return step

def save_adapter(accelerator,model,destination,step,protocol):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        module = accelerator.unwrap_model(model)
        state = module.export_trainable_state_dict(state,remove_prefix='pipe.dit.')
        expected = {n for n,p in module.pipe.dit.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('Incomplete trainable export')
        schedule = dict(base_step=BASE_STEP,ramp_steps=protocol['ramp_steps'],type=schedule_type(protocol['ramp_steps']),
                        completed_updates=step-BASE_STEP,weight=injection_weight(step,protocol['ramp_steps']))
        save_atomic(state,destination,{'step':str(step),'camera_contract':contract_json(),
            'injection_schedule':json.dumps(schedule,sort_keys=True),
            'training_protocol':json.dumps(protocol,sort_keys=True),
            'resume_semantics':'trainable weights and gate/global-step restored; optimizer moments and RNG reset'})
    accelerator.wait_for_everyone()

def full_state_checksum(model,module,accelerator):
    """Sums of squares of the trainable weights, the fp32 master partitions and
    both AdamW moments over all ranks: equal before a save and after its load."""
    weights = torch.zeros((),dtype=torch.float64,device=accelerator.device)
    for p in module.pipe.dit.parameters():
        if p.requires_grad:
            weights += p.detach().double().square().sum()
    inner = model.optimizer.optimizer
    local = torch.zeros(3,dtype=torch.float64,device=accelerator.device)
    for group in inner.param_groups:
        for p in group['params']:
            local[0] += p.detach().double().square().sum()
            state = inner.state.get(p,{})
            for i,name in ((1,'exp_avg'),(2,'exp_avg_sq')):
                if name in state:
                    local[i] += state[name].detach().double().square().sum()
    local = accelerator.reduce(local,reduction='sum')
    return {'trainable_weights':weights.item(),'fp32_master':local[0].item(),
            'exp_avg':local[1].item(),'exp_avg_sq':local[2].item()}

def save_full_state(accelerator,model,module,output,step,data_state,meta):
    """Write ``resume.partial/`` on all ranks, then swap it in for ``resume/`` by rename.
    ``data_state`` is this rank's loader position and Python RNG (see ``train.py``)."""
    partial,current,old = output/'resume.partial',output/'resume',output/'resume.old'
    if accelerator.is_main_process and partial.exists():
        shutil.rmtree(partial)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(partial),exclude_frozen_parameters=True)
    partial.mkdir(parents=True,exist_ok=True)
    torch.save(data_state,partial/f'data_state_rank{accelerator.process_index}.pt')
    checksum = full_state_checksum(model,module,accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # meta.json is written last: its presence marks a complete state.
        (partial/'meta.json').write_text(json.dumps(dict(meta,time=datetime.now().astimezone().isoformat(),
            step=step,world_size=accelerator.num_processes,camera_contract=contract_json(),checksum=checksum),indent=2)+'\n')
        if old.exists():
            shutil.rmtree(old)
        if current.exists():
            current.rename(old)
        partial.rename(current)
        if old.exists():
            shutil.rmtree(old)
        print(f'{datetime.now().astimezone().isoformat()} full state (weights + AdamW + scheduler + RNG + data position) '
              f'-> {current} (step {step})',flush=True)
    accelerator.wait_for_everyone()

def find_resume(output):
    """``resume/``, or ``resume.old/`` if a swap was cut between its two renames."""
    for name in ('resume','resume.old'):
        if (output/name/'meta.json').is_file():
            return output/name
    return None

def load_full_state(accelerator,model,module,directory):
    """Restore a ``save_full_state`` directory; returns its meta and this rank's data state."""
    meta = json.loads((directory/'meta.json').read_text())
    if meta['world_size'] != accelerator.num_processes:
        raise ValueError(f"Resume needs {meta['world_size']} GPUs")
    if meta['camera_contract'] != contract_json():
        raise ValueError('Full state does not match this scheduled camera architecture')
    accelerator.load_state(str(directory),load_module_strict=False)
    checksum = full_state_checksum(model,module,accelerator)
    for key,value in meta['checksum'].items():
        if abs(checksum[key]-value) > 1e-9*abs(value):
            raise RuntimeError(f'Full-state checksum {key}: loaded {checksum[key]} != saved {value}')
    data_state = torch.load(directory/f'data_state_rank{accelerator.process_index}.pt',weights_only=False)
    return meta,data_state
