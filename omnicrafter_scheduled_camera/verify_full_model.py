"""Gate-zero and zero-init lambda-1 correctness checks on the actual initialized 14B model, no updates."""
import json
import random
import types
import numpy as np
import torch
from diffsynth.models.wan_video_dit import SelfAttention
from diffsynth.pipelines.wan_video import model_fn_wan_video
from config import VARIANT
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
            # No schedule: the first update runs at lambda 1. The freshly initialized
            # SCoPE-FF output layer is zero, so the camera path must still be the concat model.
            if VARIANT == 'scope_ff':
                dit.injection_strength = 1.
                full = model_fn(dit,latents=latents,timestep=t,**packed)
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
            if VARIANT == 'scope_ff':
                full_error = (full.float()-baseline.float()).abs().max().item()
                scale = baseline.float().abs().max().item()
                # Bitwise is expected (SCoPE-FF adds exact zeros); a real mismatch, such as
                # a different RoPE in the camera path, is far above this bf16-level bound.
                if not torch.equal(full,baseline) and full_error > 1e-2*scale:
                    raise RuntimeError(f'Zero-init lambda-1 full-model parity FAILED: max_abs_error={full_error}, max_abs={scale}')
                report.update(lambda_one_zero_init_bitwise_equal=torch.equal(full,baseline),
                              lambda_one_zero_init_max_abs_error=full_error,baseline_max_abs=scale)
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
