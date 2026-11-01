"""CPU checks of the Wan-init concat wiring: no Wan weights, no GPU.

    python -m unittest discover -s omnicrafter_concat_wan/tests -v
"""
import sys
import unittest
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import concat_wan_paths  # noqa: E402,F401
from concat_wan_config import Config  # noqa: E402
from concat_wan_model import IN_CHANNELS, RAY_CHANNELS, WAN_CHANNELS, expand_patch_embedding, to_concat_set  # noqa: E402
from diffsynth.core.data.dataset import (get_equirectangular_plucker_rays, get_plucker_rays,  # noqa: E402
                                         normalize_w2c_make_cam_last_origin)
from diffsynth.diffusion.training_module import DiffusionTrainingModule  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from diffsynth.pipelines.wan_video import model_fn_wan_video  # noqa: E402

CONFIG = HERE/'configs/omnicrafter_mix_concat_wan_4gpu_50k.yaml'


def tiny_wan():
    torch.manual_seed(0)
    return WanModel(dim=96, in_dim=36, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=32, eps=1e-6,
                    patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=True).eval()


def wan_inputs():
    torch.manual_seed(1)
    return dict(latents=torch.randn(1, 16, 5, 8, 8), y=torch.randn(1, 20, 5, 8, 8),
                context=torch.randn(1, 300, 32), timestep=torch.tensor([500.0]))


def tiny_raymap():
    torch.manual_seed(2)
    return torch.randn(1, RAY_CHANNELS, 5, 8, 8)


def add_lora(dit):
    trainer = DiffusionTrainingModule()
    return trainer.add_lora_to_model(dit, trainer.parse_lora_target_modules('q,k,v,o,ffn.0,ffn.2'), 8)


def cameras():
    """A generic [1, 5, 4, 4] c2w: refs far then near (the sort swaps them), distinct rotations."""
    torch.manual_seed(3)
    c2w = torch.eye(4, dtype=torch.float64).repeat(1, 5, 1, 1)
    for i in range(5):
        q, _ = torch.linalg.qr(torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(3, 3, dtype=torch.float64))
        c2w[0, i, :3, :3] = q * torch.sign(torch.linalg.det(q))
    c2w[0, 0, :3, 3] = torch.tensor([3.0, 0.2, 0.0], dtype=torch.float64)
    c2w[0, 1, :3, 3] = torch.tensor([0.5, -0.1, 0.1], dtype=torch.float64)
    c2w[0, 2:, :3, 3] = torch.tensor([[0.0, 0.1, 0.0], [0.1, -0.1, 0.05], [-0.1, 0.0, 0.1]], dtype=torch.float64)
    return c2w.float()


def batch_of(height, width, camera_model):
    def images(n):
        return torch.zeros(1, n, 3, height, width)
    batch = {'ref': images(2), 'src': images(3), 'tgt': images(3), 'key': ['synthetic'], 'camera_c2w': cameras(),
             'camera_fxfycxcy': torch.tensor([[300.0, 310.0, width / 2, height / 2]]).repeat(1, 5, 1)}
    if camera_model == 'equirect':
        batch['camera_model'] = camera_model
    return batch


class WanInitTest(unittest.TestCase):
    def test_layout(self):
        dit = tiny_wan()
        wan = dit.patch_embedding.weight.detach().clone()
        expand_patch_embedding(dit)
        weight = dit.patch_embedding.weight
        self.assertEqual((WAN_CHANNELS, RAY_CHANNELS, IN_CHANNELS), (36, 384, 420))
        self.assertEqual(dit.in_dim, 420)
        self.assertEqual(weight.shape[1], 420)
        self.assertTrue(torch.equal(weight[:, :36], wan))
        self.assertFalse(weight[:, 36:].any())

    def test_step_zero_is_exactly_wan_with_a_nonzero_raymap(self):
        dit, inputs = tiny_wan(), wan_inputs()
        with torch.no_grad():
            stock = model_fn_wan_video(dit, **inputs)
            dit = add_lora(dit)
            expand_patch_embedding(dit)
            start = model_fn_wan_video(dit, raymap=tiny_raymap(), **inputs)
        self.assertTrue(torch.equal(stock, start), 'zero LoRA and zero ray channels must leave Wan unchanged')

    def test_ray_channels_get_a_gradient_and_then_move_the_output(self):
        dit = add_lora(tiny_wan())
        expand_patch_embedding(dit)
        inputs = wan_inputs()
        model_fn_wan_video(dit, raymap=tiny_raymap(), **inputs).square().mean().backward()
        self.assertTrue(dit.patch_embedding.weight.grad[:, 36:].any())
        with torch.no_grad():
            first = model_fn_wan_video(dit, raymap=tiny_raymap(), **inputs)
            dit.patch_embedding.weight[:, 36:] = 1e-3
            second = model_fn_wan_video(dit, raymap=tiny_raymap(), **inputs)
        self.assertFalse(torch.equal(first, second))


class RayMapTest(unittest.TestCase):
    def omnicrafter_gauge(self, sorted_c2w):
        _, c2w, _ = normalize_w2c_make_cam_last_origin(torch.linalg.inv(sorted_c2w[0].double()))
        return c2w.float()

    def test_pinhole_matches_omnicrafter(self):
        height, width = 288, 512
        data = to_concat_set(batch_of(height, width, 'pinhole'), sort_refs=False)
        self.assertEqual(tuple(data['raymap'].shape), (5, 384, height // 8, width // 8))
        K = torch.tensor([[300.0, 0, width / 2], [0, 310.0, height / 2], [0, 0, 1.0]]).repeat(5, 1, 1)
        reference = get_plucker_rays(self.omnicrafter_gauge(cameras()), K, height, width)
        self.assertLess((data['raymap'] - reference).abs().max().item(), 1e-5)

    def test_equirect_matches_omnicrafter(self):
        height, width = 256, 512
        data = to_concat_set(batch_of(height, width, 'equirect'), sort_refs=False)
        self.assertEqual(data['camera_model'], 'equirect')
        reference = get_equirectangular_plucker_rays(self.omnicrafter_gauge(cameras()), height, width)
        self.assertEqual(tuple(reference.shape), (5, 384, 32, 64))
        self.assertLess((data['raymap'] - reference).abs().max().item(), 1e-5)

    def test_sort_moves_the_nearer_ref_first_and_black_queries(self):
        batch = batch_of(288, 512, 'pinhole')
        plain, ordered = to_concat_set(batch, sort_refs=False), to_concat_set(batch, sort_refs=True, black_queries=True)
        self.assertTrue(torch.equal(plain['raymap'][0], ordered['raymap'][1]))
        self.assertTrue(torch.equal(plain['raymap'][1], ordered['raymap'][0]))
        self.assertTrue(torch.equal(plain['raymap'][2:], ordered['raymap'][2:]))
        self.assertFalse(any(image.getbbox() for image in ordered['conditioning_images']))

    def test_near_depth_unit_cancels(self):
        batch = batch_of(288, 512, 'pinhole')
        scaled = dict(batch, camera_c2w=batch['camera_c2w'].clone())
        scaled['camera_c2w'][..., :3, 3] /= 3.39
        difference = to_concat_set(batch)['raymap'] - to_concat_set(scaled)['raymap']
        self.assertLess(difference.abs().max().item(), 1e-5)


class ConfigTest(unittest.TestCase):
    def test_run_config(self):
        cfg = Config.from_yaml(CONFIG)
        self.assertEqual((cfg.max_steps, cfg.lora_rank, cfg.learning_rate, cfg.warmup_steps), (50000, 32, 1e-5, 500))
        self.assertEqual((cfg.nvs_ratio, cfg.panorama_ratio, cfg.max_pixel_dropout), (0.5, 0.5, 0.75))
        self.assertIn('concat_wan', cfg.output_dir)

    def test_scope_keys_are_rejected(self):
        values = yaml.safe_load(CONFIG.read_text())
        path = Path(__file__).with_name('.reject_scope.yaml')
        path.write_text(yaml.safe_dump({**values, 'scope_mlp_hidden': 64}))
        try:
            with self.assertRaises(ValueError):
                Config.from_yaml(path)
        finally:
            path.unlink()


if __name__ == '__main__':
    unittest.main()
