"""Gate-zero correctness check on the actual initialized 14B model, no updates."""
import json
import random
import types
import numpy as np
import torch
from diffsynth.models.wan_video_dit import SelfAttention
from diffsynth.pipelines.wan_video import model_fn_wan_video
from wan_camera import model_fn

@torch.no_grad()
def verify_full_model(module,data,accelerator,output):
    cases = {}
    for i in range(len(data)):
        row = data.record(i)
        cases.setdefault(row['projection'],i)
        if len(cases)==2:
            break
    if len(cases)!=2 or accelerator.num_processes!=4:
        raise ValueError('Expected both projections and four ranks for full-model parity')
    projection = 'panorama' if accelerator.process_index < 2 else 'perspective'
    task = 'nvs' if accelerator.process_index % 2 == 0 else 'cleanup'
    rng_py,rng_np = random.getstate(),np.random.get_state()
    probability,was_training = data.nvs_probability,module.training
    dit = module.pipe.dit
    previous_strength = dit.injection_strength
    forwards = [b.self_attn.forward for b in dit.blocks]
    try:
        module.eval(); data.nvs_probability = 1. if task=='nvs' else 0.
        with torch.random.fork_rng(devices=[accelerator.device.index]):
            batch = data[cases[projection]]
            packed = module.encode_set(batch)
            latents = torch.randn_like(packed.pop('input_latents'))
            t = module.pipe.scheduler.timesteps[0].reshape(1).to(latents)
            dit.injection_strength = 0.
            rng_before = torch.random.get_rng_state().clone()
            gpu_rng_before = torch.cuda.get_rng_state().clone()
            proposed = model_fn(dit,latents=latents,timestep=t,**packed)
            assert torch.equal(rng_before,torch.random.get_rng_state())
            assert torch.equal(gpu_rng_before,torch.cuda.get_rng_state())
            for block in dit.blocks:
                block.self_attn.forward = types.MethodType(SelfAttention.forward,block.self_attn)
            baseline = model_fn_wan_video(dit,latents=latents,timestep=t,
                context=packed['context'],y=packed['y'],raymap=packed['camera_context']['raymap'].to(latents))
            error = (proposed.float()-baseline.float()).abs().max().item()
            if not torch.equal(proposed,baseline):
                raise RuntimeError(f'Gate-zero full-model parity FAILED: max_abs_error={error}')
            report = dict(passed=True,rank=accelerator.process_index,projection=projection,task=task,
                image_hw=list(batch['camera_inputs']['image_hw']),bitwise_equal=True,max_abs_error=error,
                gate=0.,RNG_unchanged=True,optimizer_updates=0)
            (output/f'gate_zero_full_model_rank{accelerator.process_index}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'event':'full_model_gate_zero_parity',**report}),flush=True)
    finally:
        for block,forward in zip(dit.blocks,forwards):
            block.self_attn.forward=forward
        dit.injection_strength=previous_strength
        data.nvs_probability=probability
        module.train(was_training)
        random.setstate(rng_py); np.random.set_state(rng_np)
    accelerator.wait_for_everyone()
