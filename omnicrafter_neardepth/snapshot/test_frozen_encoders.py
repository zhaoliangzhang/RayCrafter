"""Regression tests for the encoder lifecycle, full-resolution inputs and queue."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from common import REPO, GPU_UUIDS
sys.path.insert(0, str(REPO))
from diffsynth.pipelines.wan_video import WanVideoPipeline
from mixed_model import MixedTrainingModule
from frozen_encoders import release_frozen_encoders, CachedEmptyPrompt


class FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        assert prompt == ''
        return torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]])


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(32).reshape(4, 8).float(), requires_grad=False)

    def forward(self, ids, mask):
        return self.weight[ids]


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.model = SimpleNamespace(z_dim=16)
        self.upsampling_factor = 8

    def encode(self, images, **kwargs):
        def encode_one(x):
            return x.float().mean(0, keepdim=True)[:, :, ::8, ::8].repeat(16, 1, 1, 1)
        return torch.stack([encode_one(x) for x in images])


def make_module():
    module = MixedTrainingModule.__new__(MixedTrainingModule)
    nn.Module.__init__(module)
    module.pipe = WanVideoPipeline(device='cpu', torch_dtype=torch.float32)
    module.pipe.text_encoder = FakeText()
    module.pipe.tokenizer = FakeTokenizer()
    module.pipe.image_encoder = nn.Linear(2, 2).requires_grad_(False)
    module.pipe.vae = FakeVAE()
    module.pipe.dit = nn.Linear(2, 2)
    module.pipe.dit.individual_encoding = True
    module.pipe.dit.require_vae_embedding = True
    module.pipe.dit.require_clip_embedding = True
    module.pipe.scheduler.set_timesteps(1000, training=True)
    module.architecture = 'scope'
    module.extra_inputs = ['conditioning_images']
    module.use_gradient_checkpointing = True
    module.use_gradient_checkpointing_offload = False
    module.max_timestep_boundary = 1.
    module.min_timestep_boundary = 0.
    return module


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_full_resolution_packed_inputs_identical_for_all_cases(self):
        original = make_module()
        optimized = copy.deepcopy(original)
        rng = torch.get_rng_state().clone()
        report = release_frozen_encoders(optimized.pipe, 'cpu')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(report['repeat_embedding_bit_exact'])
        self.assertIsNone(optimized.pipe.text_encoder)
        self.assertIsNone(optimized.pipe.image_encoder)
        self.assertNotIn('empty_prompt_context', optimized.pipe.state_dict())
        self.assertTrue(torch.equal(original.pipe.dit.weight, optimized.pipe.dit.weight))
        for height in (256, 288):
            for color in (0, 173):
                images = [Image.new('RGB', (512, height), (i*30, 51, 89)) for i in range(5)]
                row = {'input_images':images[:2], 'target_images':images,
                       'conditioning_images':[Image.new('RGB', (512, height), (color,)*3) for _ in range(3)],
                       'prompt':'', 'scope_rays':torch.ones(5*(height//16)*32, 6)}
                torch.manual_seed(19)
                before = original.encode_set(row)
                torch.manual_seed(19)
                after = optimized.encode_set(row)
                self.assertEqual(set(before), set(after))
                for key in before:
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
                self.assertEqual(list(after['input_latents'].shape), [1,16,5,height//8,64])
                self.assertEqual(list(after['y'].shape), [1,20,5,height//8,64])

    def test_buffer_follows_pipeline_and_nonempty_prompt_rejected(self):
        module = make_module()
        release_frozen_encoders(module.pipe, 'cpu')
        module.to(dtype=torch.bfloat16)
        self.assertEqual(module.pipe.empty_prompt_context.dtype, torch.bfloat16)
        self.assertFalse(module.pipe.empty_prompt_context.requires_grad)
        unit = next(u for u in module.pipe.units if isinstance(u, CachedEmptyPrompt))
        with self.assertRaises(ValueError):
            unit.process(module.pipe, 'not an empty prompt', True)

    def test_reject_active_clip_or_trainable_text(self):
        module = make_module()
        module.pipe.dit.individual_encoding = False
        with self.assertRaises(AssertionError):
            release_frozen_encoders(module.pipe, 'cpu')
        module.pipe.dit.individual_encoding = True
        module.pipe.text_encoder.weight.requires_grad_(True)
        with self.assertRaises(AssertionError):
            release_frozen_encoders(module.pipe, 'cpu')

    def test_queue_ignores_nonselected_cards_but_waits_for_selected_card(self):
        from queue_io100 import inventory
        rows = '\n'.join(f'{i}, {uuid}, A6000, 2, 0' for i,uuid in enumerate(GPU_UUIDS))
        rows += '\n4, GPU-other, A6000, 45000, 100'
        with patch('queue_io100.subprocess.check_output', side_effect=[rows, '123, GPU-other, 45000']):
            self.assertTrue(inventory()['idle'])
        with patch('queue_io100.subprocess.check_output', side_effect=[rows, f'123, {GPU_UUIDS[0]}, 45000']):
            self.assertFalse(inventory()['idle'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
