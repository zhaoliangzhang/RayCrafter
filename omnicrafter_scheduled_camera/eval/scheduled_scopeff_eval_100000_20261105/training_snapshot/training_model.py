"""OmniCrafter's existing WAN flow-matching loss, replacing only camera injection."""
import torch
import bootstrap
from model_training.train import WanTrainingModule
from camera import prepare_camera_context
from wan_camera import install_camera_attention, model_fn
from checkpoints import restore_adapter, warm_start
from config import BASE_STEP, RAMP_STEPS


def target_loss(prediction, target, weight):
    error = (prediction[:, :, -3:].float() - target[:, :, -3:].float()).square()
    weight = weight.to(device=error.device, dtype=error.dtype).reshape(-1)
    return (error.flatten(1).mean(1) * weight).mean()


class MixedTrainingModule(WanTrainingModule):
    def __init__(self, *args, restore_checkpoint=None, ramp_steps=RAMP_STEPS, **kwargs):
        super().__init__(*args, **kwargs)
        install_camera_attention(self.pipe.dit)
        self.pipe.model_fn = model_fn
        warm_start(self.pipe.dit, kwargs['lora_checkpoint'])
        self.initial_step = restore_adapter(self.pipe.dit, restore_checkpoint, ramp_steps) if restore_checkpoint else BASE_STEP

    @torch.no_grad()
    def encode_set(self, data):
        if (len(data['input_images']), len(data['conditioning_images']), len(data['target_images'])) != (2, 3, 5):
            raise ValueError('Expected two references and three targets')
        inputs = self.transfer_data_to_device(self._get_pipeline_inputs(data), self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _ = inputs
        if not shared['mask_loss'] or shared['num_output_frames'] != 3:
            raise ValueError('The WAN pipeline must mask loss to the three targets')
        packed = {k: {**shared, **positive}[k] for k in ('input_latents', 'y', 'context')}
        packed['camera_context'] = prepare_camera_context(data['camera_inputs'], self.pipe.device)
        return packed

    def forward(self, batch):
        if len(batch) != 1:
            raise ValueError('Mixed projection loader expects one logical set per GPU')
        packed = self.encode_set(batch[0])
        clean = packed.pop('input_latents')
        if clean.shape[0] != 1 or clean.shape[2] != 5:
            raise ValueError('Each image must be encoded individually: two refs then three queries')
        index = torch.randint(0, len(self.pipe.scheduler.timesteps), (1,))
        t = self.pipe.scheduler.timesteps[index].to(self.pipe.device, dtype=self.pipe.torch_dtype)
        noise = torch.randn_like(clean)
        noisy = self.pipe.scheduler.add_noise(clean, noise, t)
        target = self.pipe.scheduler.training_target(clean, noise, t)
        prediction = model_fn(self.pipe.dit, latents=noisy, timestep=t, **packed,
                              use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
        return target_loss(prediction, target, self.pipe.scheduler.training_weight(t))
