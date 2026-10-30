"""CPU checks of the scope-ff wiring: no Wan weights, no GPU.

    python -m unittest discover -s omnicrafter_scopeff/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scopeff_paths  # noqa: E402,F401
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from diffsynth.pipelines.wan_video import model_fn_wan_video  # noqa: E402
from raycrafter.cameras.scope import SightlineCoordinatePE, plucker_tokens  # noqa: E402
from scope_ff import install_scope_ff, model_fn_scope_ff  # noqa: E402
from scopeff_model import PATCH, to_omnicrafter  # noqa: E402

DATA = Path('/weka/projects/rchella4/zzhan288/data')


def tiny_wan():
    torch.manual_seed(0)
    return WanModel(dim=96, in_dim=36, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=32, eps=1e-6,
                    patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=True).eval()


def cameras(ref_distances=(2.0, 0.5)):
    """c2w [1, 5, 4, 4]: two refs on the x axis, three queries near the origin."""
    c2w = torch.eye(4).repeat(1, 5, 1, 1)
    c2w[0, 0, 0, 3], c2w[0, 1, 0, 3] = ref_distances
    c2w[0, 2:, 1, 3] = torch.tensor([0.0, 0.1, -0.1])
    return c2w


def wan_inputs():
    torch.manual_seed(1)
    # 5 frames of 8x8 latents (4x4 tokens), 16 noisy + 20 conditioning channels;
    # the context has the 257 image-token prefix the I2V cross-attention splits off.
    return dict(latents=torch.randn(1, 16, 5, 8, 8), y=torch.randn(1, 20, 5, 8, 8),
                context=torch.randn(1, 300, 32), timestep=torch.tensor([500.0]))


def tiny_rays(channels=7):
    rays = plucker_tokens(cameras(), torch.tensor([[60.0, 60.0, 32.0, 32.0]]).repeat(1, 5, 1),
                          (64, 64), (4, 4), PATCH, patch_size=True)
    return rays[..., :channels]


def synthetic_batch(height, width, camera_model):
    torch.manual_seed(2)
    def images(n):
        return torch.randint(0, 256, (1, n, 3, height, width), dtype=torch.uint8).float() / 255 * 2 - 1
    batch = {'ref': images(2), 'src': images(3), 'tgt': images(3), 'key': ['synthetic'],
             'camera_c2w': cameras(), 'camera_fxfycxcy': torch.tensor([[300.0, 300.0, width / 2, height / 2]]).repeat(1, 5, 1)}
    if camera_model == 'equirect':
        batch['camera_model'] = 'equirect'
        batch['camera_fxfycxcy'] = torch.zeros(1, 5, 4)
    return batch


class ScopeFFModelTest(unittest.TestCase):
    def test_identity_at_initialization(self):
        dit, inputs = tiny_wan(), wan_inputs()
        with torch.no_grad():
            stock = model_fn_wan_video(dit, **inputs)
            install_scope_ff(dit, mlp_hidden=64, dir_freqs=8, patch_size=True)
            scoped = model_fn_scope_ff(dit, scope_rays=tiny_rays(), **inputs)
            self.assertTrue(torch.equal(stock, scoped), 'zero output layers must leave the DiT unchanged')
            torch.nn.init.normal_(dit.blocks[0].self_attn.scope_pe.eq[2].weight)
            moved = model_fn_scope_ff(dit, scope_rays=tiny_rays(), **inputs)
        self.assertFalse(torch.allclose(stock, moved), 'a trained E_q must change the output')

    def test_parameters_and_ray_width(self):
        dit = tiny_wan()
        install_scope_ff(dit, mlp_hidden=64, dir_freqs=8, patch_size=True)
        pe = dit.blocks[0].self_attn.scope_pe
        self.assertEqual(len(list(pe.parameters())), 12)
        self.assertEqual(tuple(pe.eq[0].weight.shape), (64, 7 + 6 * 8 + 1))
        with self.assertRaises(ValueError):
            model_fn_scope_ff(dit, scope_rays=tiny_rays(6), **wan_inputs())
        with self.assertRaises(RuntimeError):
            install_scope_ff(dit)

    def test_fourier_frequencies_stay_fp32(self):
        dit = tiny_wan()
        install_scope_ff(dit, mlp_hidden=64, dir_freqs=8, patch_size=True)
        dit.bfloat16()  # what DeepSpeed's bf16 engine does to the module
        pe = dit.blocks[1].self_attn.scope_pe
        self.assertEqual(pe.eq[0].weight.dtype, torch.bfloat16)
        self.assertEqual(pe.dir_omega.dtype, torch.float32)
        self.assertTrue(torch.equal(pe.dir_omega, SightlineCoordinatePE(96, dir_freqs=8).dir_omega))


class AdapterTest(unittest.TestCase):
    def test_pinhole_and_equirect_sets(self):
        for (height, width), camera_model in (((288, 512), 'pinhole'), ((256, 512), 'equirect')):
            batch = synthetic_batch(height, width, camera_model)
            data = to_omnicrafter(batch)
            self.assertEqual(data['camera_model'], camera_model)
            self.assertEqual(tuple(data['scope_rays'].shape), (5 * (height // 16) * (width // 16), 7))
            self.assertTrue(torch.isfinite(data['scope_rays']).all())
            # The nearer ref (index 1, 0.5 from the queries) comes first and is the gauge origin.
            self.assertTrue(torch.equal(TF.to_tensor(data['input_images'][0]) * 2 - 1, batch['ref'][0, 1]))
            first = data['scope_rays'][:(height // 16) * (width // 16)]
            self.assertTrue(torch.allclose(first[:, 3:6], torch.zeros(1)), 'the anchor ref has zero moments')
            expected = plucker_tokens(batch['camera_c2w'][:, [1, 0, 2, 3, 4]], batch['camera_fxfycxcy'][:, [1, 0, 2, 3, 4]],
                                      (height, width), (height // 16, width // 16), 16,
                                      camera_model=camera_model, patch_size=True)[0]
            self.assertTrue(torch.equal(data['scope_rays'], expected))
            # PIL round trip is exact for uint8 images.
            for image, tensor in zip(data['target_images'][2:], batch['tgt'][0]):
                self.assertLess((TF.to_tensor(image) * 2 - 1 - tensor).abs().max().item(), 1e-6)
            self.assertEqual(len(data['target_images']), 5)
            black = to_omnicrafter(batch, black_queries=True)
            self.assertFalse(any(np.asarray(image).any() for image in black['conditioning_images']))

    @unittest.skipUnless(DATA.is_dir(), 'needs the dl3dv_points / framecrafter_5input data')
    def test_real_sets(self):
        from raycrafter.data.dataset import _collate
        from raycrafter.data.dl3dv_points import Dl3dvPoints, load_near_depth
        from raycrafter.data.panorama import PanoramaSets
        points = Dl3dvPoints(DATA/'dl3dv_points/test', (288, 512), use_pose=True, max_pixel_dropout=0.0,
                             near_depth=load_near_depth(DATA/'dl3dv_points/near_depth.json'))
        panoramas = PanoramaSets(DATA/'framecrafter_5input',
                                 ['manifests/jhu_holdout10_seed20260822/jhu_test_2ref1_10_scenes.jsonl'],
                                 use_pose=True, max_pixel_dropout=0.0,
                                 near_depth=load_near_depth(DATA/'framecrafter_5input/near_depth.json'))
        for dataset, tokens in ((points, 18 * 32), (panoramas, 16 * 32)):
            batch = _collate([dataset[0]])
            data = to_omnicrafter(batch)
            self.assertEqual(tuple(data['scope_rays'].shape), (5 * tokens, 7))
            for image, tensor in zip(data['conditioning_images'], batch['src'][0]):
                self.assertLess((TF.to_tensor(image) * 2 - 1 - tensor).abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
