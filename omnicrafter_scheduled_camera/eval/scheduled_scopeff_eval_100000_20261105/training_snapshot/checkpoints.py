"""Strict concat warm-start; isolated scheduled adapters with resume metadata."""
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from config import BASE_STEP, VARIANT, injection_weight
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
    inherited = {n for n,p in params.items() if p.requires_grad and '.scope_pe.' not in n and '.unirope_depth.' not in n}
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
    if schedule != {'base_step':BASE_STEP,'ramp_steps':ramp_steps,'type':'linear',
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
        schedule = dict(base_step=BASE_STEP,ramp_steps=protocol['ramp_steps'],type='linear',
                        completed_updates=step-BASE_STEP,weight=injection_weight(step,protocol['ramp_steps']))
        save_atomic(state,destination,{'step':str(step),'camera_contract':contract_json(),
            'injection_schedule':json.dumps(schedule,sort_keys=True),
            'training_protocol':json.dumps(protocol,sort_keys=True),
            'resume_semantics':'trainable weights and gate/global-step restored; optimizer moments and RNG reset'})
    accelerator.wait_for_everyone()
