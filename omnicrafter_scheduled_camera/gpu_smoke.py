"""Short four-rank correctness check before WAN loading; no experiment saves."""
import json
import os
import torch
from accelerate import Accelerator,DeepSpeedPlugin
from config import SOURCE,OUT,VARIANT
from dataset import MixedProjectionDataset
from camera import ordered_cameras,prepare_camera_context
from train import zero2_config
from wan_camera import install_camera_attention,model_fn,optimizer_groups
from diffsynth.models.wan_video_dit import WanModel

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit=WanModel(dim=128,in_dim=420,ffn_dim=192,out_dim=16,text_dim=16,freq_dim=16,
            eps=1e-6,patch_size=(1,2,2),num_heads=1,num_layers=2,has_image_input=False,require_clip_embedding=False)
        install_camera_attention(self.dit)
        self.to(dtype=torch.bfloat16)

    def forward(self,cameras,strength):
        self.dit.injection_strength=strength
        device=next(self.parameters()).device
        h,w=[v//8 for v in cameras['image_hw']]
        x=torch.randn(1,16,5,h,w,device=device,dtype=torch.bfloat16)
        y=torch.randn(1,20,5,h,w,device=device,dtype=torch.bfloat16)
        out=model_fn(self.dit,latents=x,y=y,context=torch.randn(1,6,16,device=device,dtype=torch.bfloat16),
            timestep=torch.ones(1,device=device,dtype=torch.bfloat16),
            camera_context=prepare_camera_context(cameras,device),use_gradient_checkpointing=True)
        return out[:,:,-3:].float().square().mean()

def main():
    torch.set_num_threads(4)
    acc=Accelerator(mixed_precision='bf16',deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=zero2_config()))
    if acc.num_processes!=4:raise ValueError('Require four ranks')
    data=MixedProjectionDataset(SOURCE/'train.jsonl',SOURCE/'train.offsets.npy')
    cases={}
    for i in range(len(data)):
        row=data.record(i);cases.setdefault(row['projection'],ordered_cameras(row)[1])
        if len(cases)==2:break
    module=Tiny();optimizer=torch.optim.AdamW(optimizer_groups(module.dit))
    model,optimizer=acc.prepare(module,optimizer)
    initial={n:p.detach().clone() for n,p in module.dit.named_parameters() if '.scope_pe.' in n or '.rayrope_depth.' in n}
    results=[]
    for projection,strength in [('panorama',.0001),('perspective',.2),('panorama',1.)]:
        optimizer.zero_grad()
        loss=model(cases[projection],strength)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite compiled GPU loss')
        acc.backward(loss);optimizer.step()
        states=[v for s in model.optimizer.optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')]
        if not states or not all(v.is_cuda for v in states):raise RuntimeError('Optimizer offload detected')
        results.append(dict(projection=projection,strength=strength,loss=loss.item()))
    params=dict(acc.unwrap_model(model).dit.named_parameters())
    if not any(not torch.equal(v,params[n]) for n,v in initial.items()):
        raise RuntimeError('New camera parameters did not update')
    report=dict(passed=True,variant=VARIANT,rank=acc.process_index,optimizer_device='cuda',zero_stage=2,
                real_resolution_grids=True,compiled_geometry=os.environ.get('TORCH_COMPILE_DISABLE')!='1',new_parameters_updated=True,cases=results)
    (OUT/f'gpu_correctness_rank{acc.process_index}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    acc.wait_for_everyone();acc.end_training()

if __name__=='__main__':main()
