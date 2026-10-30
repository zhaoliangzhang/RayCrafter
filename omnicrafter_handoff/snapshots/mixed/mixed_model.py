"""Batch-one mixed-projection training, sharing the established flow objective."""
import sys
import types
import torch
from common import REPO
sys.path.insert(0,str(REPO))
from model_training.train import WanTrainingModule
from scope_model import install_scope, model_fn_scope


def batched_head_forward(self,x,t_mod):
    if t_mod.ndim == 2:
        t_mod = t_mod[:,None,:]
    shift,scale = (self.modulation[None].to(t_mod)+t_mod.unsqueeze(2)).chunk(2,dim=2)
    return self.head(self.norm(x)*(1+scale.squeeze(2))+shift.squeeze(2))


def install_batch_head(dit):
    dit.head.forward = types.MethodType(batched_head_forward,dit.head)


def target_loss(prediction,target,weight):
    error = (prediction[:,:,-3:].float()-target[:,:,-3:].float()).square()
    # WAN's scheduler stores timestep weights on CPU. Move the weight before
    # reshaping/multiplication so every rank computes its loss on its own GPU.
    weight = weight.to(device=error.device,dtype=error.dtype).reshape(-1)
    return (error.flatten(1).mean(1)*weight).mean()


class MixedTrainingModule(WanTrainingModule):
    def __init__(self,*args,architecture,restore_checkpoint=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.architecture = architecture
        expected_channels = 36 if architecture == 'scope' else 420
        assert self.pipe.dit.in_dim == expected_channels
        install_batch_head(self.pipe.dit)
        if architecture == 'scope':
            install_scope(self.pipe.dit)
            self.pipe.model_fn = model_fn_scope
        if restore_checkpoint:
            from safetensors.torch import load_file
            source = load_file(str(restore_checkpoint),device='cpu')
            expected = {n for n,p in self.pipe.dit.named_parameters() if p.requires_grad}
            assert set(source) == expected, 'Resume checkpoint architecture differs'
            self.pipe.dit.load_state_dict(source,strict=False)
            state = self.pipe.dit.state_dict()
            assert all(torch.equal(state[n],v.to(state[n].dtype)) for n,v in source.items())
            print(f'Exactly restored {len(source)} tensors from {restore_checkpoint}',flush=True)
        self.last_batch_shapes = None

    @torch.no_grad()
    def encode_set(self,data):
        assert len(data['input_images']) == 2 and len(data['conditioning_images']) == 3
        assert len(data['target_images']) == 5 and data['prompt'] == ''
        inputs = self.transfer_data_to_device(self._get_pipeline_inputs(data),self.pipe.device,self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit,self.pipe,*inputs)
        shared,positive,_ = inputs
        assert shared['mask_loss'] and shared['num_output_frames'] == 3
        packed = {k:({**shared,**positive})[k] for k in ('input_latents','y','context')}
        if self.architecture == 'scope':
            assert 'raymap' not in data
            packed['scope_rays'] = data['scope_rays'].to(self.pipe.device,dtype=torch.float32).unsqueeze(0)
        else:
            assert 'scope_rays' not in data
            packed['raymap'] = data['raymap'].to(self.pipe.device,dtype=self.pipe.torch_dtype).permute(1,0,2,3).unsqueeze(0)
        return packed

    def forward(self,batch):
        assert isinstance(batch,list) and len(batch) == 1
        packed = self.encode_set(batch[0])
        self.last_batch_shapes = {k:list(v.shape) for k,v in packed.items()}
        clean = packed.pop('input_latents')
        assert clean.shape[0] == 1 and clean.shape[2] == 5
        index = torch.randint(0,len(self.pipe.scheduler.timesteps),(1,))
        t = self.pipe.scheduler.timesteps[index].to(self.pipe.device,dtype=self.pipe.torch_dtype)
        noise = torch.randn_like(clean)
        noisy = self.pipe.scheduler.add_noise(clean,noise,t)
        target = self.pipe.scheduler.training_target(clean,noise,t)
        weight = self.pipe.scheduler.training_weight(t)
        prediction = self.pipe.model_fn(dit=self.pipe.dit,latents=noisy,timestep=t,
            **packed,use_gradient_checkpointing=True,use_gradient_checkpointing_offload=False)
        assert prediction.shape == clean.shape
        return target_loss(prediction,target,weight)
