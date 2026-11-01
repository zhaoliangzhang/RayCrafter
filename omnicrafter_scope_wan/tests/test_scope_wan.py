"""CPU checks of the Wan-init SCoPE wiring: no Wan weights, no GPU.

    python -m unittest discover -s omnicrafter_scope_wan/tests -v
"""
import sys
import unittest
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import scope_wan_paths  # noqa: E402,F401
from diffsynth.diffusion.training_module import DiffusionTrainingModule  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from diffsynth.pipelines.wan_video import model_fn_wan_video  # noqa: E402
from raycrafter.cameras.scope import plucker_tokens  # noqa: E402
from scope_ff import install_scope_ff, model_fn_scope_ff  # noqa: E402
from scope_wan_config import Config  # noqa: E402
from scopeff_model import PATCH, to_omnicrafter  # noqa: E402

CONFIG = HERE/'configs/omnicrafter_mix_scope_wan_4gpu_50k.yaml'
PLAIN = dict(mlp_hidden=64, dir_freqs=0, patch_size=False)


def tiny_wan():
    torch.manual_seed(0)
    return WanModel(dim=96, in_dim=36, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=32, eps=1e-6,
                    patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=True).eval()


def wan_inputs():
    torch.manual_seed(1)
    return dict(latents=torch.randn(1, 16, 5, 8, 8), y=torch.randn(1, 20, 5, 8, 8),
                context=torch.randn(1, 300, 32), timestep=torch.tensor([500.0]))


def cameras():
    c2w = torch.eye(4).repeat(1, 5, 1, 1)
    c2w[0, 0, 0, 3], c2w[0, 1, 0, 3] = 2.0, 0.5
    c2w[0, 2:, 1, 3] = torch.tensor([0.0, 0.1, -0.1])
    return c2w


def tiny_rays():
    return plucker_tokens(cameras(), torch.tensor([[60.0, 60.0, 32.0, 32.0]]).repeat(1, 5, 1), (64, 64), (4, 4), PATCH)


def add_lora(dit):
    trainer = DiffusionTrainingModule()
    return trainer.add_lora_to_model(dit, trainer.parse_lora_target_modules('q,k,v,o,ffn.0,ffn.2'), 8)


class WanInitTest(unittest.TestCase):
    def test_step_zero_is_exactly_wan(self):
        dit, inputs = tiny_wan(), wan_inputs()
        with torch.no_grad():
            stock = model_fn_wan_video(dit, **inputs)
            dit = add_lora(dit)
            install_scope_ff(dit, **PLAIN)
            start = model_fn_scope_ff(dit, scope_rays=tiny_rays(), **inputs)
        self.assertTrue(torch.equal(stock, start), 'zero LoRA and zero SCoPE must leave Wan unchanged')
        for name, parameter in dit.named_parameters():
            if 'lora_B' in name:
                self.assertFalse(parameter.any(), name)

    def test_lora_and_scope_can_move_the_output(self):
        dit, inputs = add_lora(tiny_wan()), wan_inputs()
        install_scope_ff(dit, **PLAIN)
        for name, parameter in dit.named_parameters():
            parameter.requires_grad = 'lora_' in name or '.scope_pe.' in name
        loss = model_fn_scope_ff(dit, scope_rays=tiny_rays(), **inputs).square().mean()
        loss.backward()
        grads = {n: p.grad for n, p in dit.named_parameters() if p.requires_grad}
        self.assertTrue(any(g is not None and g.any() for n, g in grads.items() if 'lora_B' in n))
        # SCoPE's output layers start at zero, so they are the first of its weights to get a gradient.
        for layer in ('.scope_pe.eq.2.', '.scope_pe.ek.2.'):
            self.assertTrue(any(g is not None and g.any() for n, g in grads.items() if layer in n), layer)


class PlainScopeTest(unittest.TestCase):
    def test_seven_released_inputs(self):
        dit = tiny_wan()
        install_scope_ff(dit, **PLAIN)
        pe = dit.blocks[0].self_attn.scope_pe
        self.assertEqual(len(list(pe.parameters())), 12)
        self.assertEqual(tuple(pe.eq[0].weight.shape), (64, 7))
        with self.assertRaises(ValueError):
            model_fn_scope_ff(dit, scope_rays=torch.zeros(1, 80, 7), **wan_inputs())
        with self.assertRaises(ValueError):
            model_fn_scope_ff(dit, scope_rays=tiny_rays(), raymap=torch.zeros(1), **wan_inputs())

    def test_adapter_builds_six_channel_rays(self):
        for (height, width), model in (((288, 512), 'pinhole'), ((256, 512), 'equirect')):
            def images(n):
                return torch.zeros(1, n, 3, height, width)
            batch = {'ref': images(2), 'src': images(3), 'tgt': images(3), 'key': ['synthetic'], 'camera_c2w': cameras(),
                     'camera_fxfycxcy': torch.tensor([[300.0, 300.0, width / 2, height / 2]]).repeat(1, 5, 1)}
            if model == 'equirect':
                batch['camera_model'] = model
            rays = to_omnicrafter(batch, patch_size=False)['scope_rays']
            self.assertEqual(tuple(rays.shape), (5 * (height // PATCH) * (width // PATCH), 6))
            self.assertTrue(torch.isfinite(rays).all())


class ConfigTest(unittest.TestCase):
    def test_run_config(self):
        cfg = Config.from_yaml(CONFIG)
        self.assertEqual((cfg.max_steps, cfg.lora_rank, cfg.learning_rate), (50000, 32, 1e-5))
        self.assertIn('scope_wan', cfg.output_dir)

    def test_framecrafter_and_scope_ff_keys_are_rejected(self):
        values = yaml.safe_load(CONFIG.read_text())
        for key, value in (('base_initialization', 'x'), ('scope_dir_freqs', 8), ('scope_patch_size', True)):
            path = Path(__file__).with_name(f'.reject_{key}.yaml')
            path.write_text(yaml.safe_dump({**values, key: value}))
            try:
                with self.assertRaises(ValueError):
                    Config.from_yaml(path)
            finally:
                path.unlink()


if __name__ == '__main__':
    unittest.main()
