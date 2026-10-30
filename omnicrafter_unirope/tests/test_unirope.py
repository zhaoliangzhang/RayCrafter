"""Checks of the UniRoPE wiring: no Wan weights. Runs on a GPU if there is one.
As the launcher, without torch.compile (it needs Python.h, which this env's system Python lacks):

    TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=4 python -m unittest discover -s omnicrafter_unirope/tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unirope_paths  # noqa: E402,F401
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from raycrafter.cameras.unirope import _direction_freqs, prepare_unirope  # noqa: E402
from unirope_model import optimizer_groups, to_omnicrafter_unirope  # noqa: E402
from unirope_wan import install_unirope, model_fn_unirope, unirope_geometry  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tiny_wan(dim=256, install=True):
    """Two blocks with 128-d heads, the head size UniRoPE needs (as the 14B's 5120 / 40)."""
    torch.manual_seed(0)
    dit = WanModel(dim=dim, in_dim=36, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=32, eps=1e-6,
                   patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=True).to(DEVICE).eval()
    if install:
        install_unirope(dit, apply_vo=True)
    return dit


def cameras(ref_distances=(2.0, 0.5)):
    """c2w [1, 5, 4, 4]: two refs on the x axis, three queries near the origin, each rotated differently."""
    c2w = torch.eye(4).repeat(1, 5, 1, 1)
    c2w[0, 0, 0, 3], c2w[0, 1, 0, 3] = ref_distances
    c2w[0, 2:, 1, 3] = torch.tensor([0.0, 0.1, -0.1])
    for index, angle in enumerate((0.1, -0.2, 0.3, 0.2, -0.1)):
        c2w[0, index, :3, :3] = rotation_y(angle)
    return c2w


def rotation_y(angle):
    angle = torch.tensor(angle)
    return torch.tensor([[angle.cos(), 0.0, angle.sin()], [0.0, 1.0, 0.0], [-angle.sin(), 0.0, angle.cos()]])


def wan_inputs(width=8, dtype=torch.float32):
    torch.manual_seed(1)
    # 5 frames of 8 x width latents, 16 noisy + 20 conditioning channels; the
    # context has the 257 image-token prefix the I2V cross-attention splits off.
    inputs = dict(latents=torch.randn(1, 16, 5, 8, width, dtype=dtype), y=torch.randn(1, 20, 5, 8, width, dtype=dtype),
                  context=torch.randn(1, 300, 32, dtype=dtype), timestep=torch.tensor([500.0], dtype=dtype))
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def geometry(c2w=None, camera_model='pinhole', width=8):
    """model_fn_unirope's geometry arguments for 4 x (width / 2) tokens."""
    c2w = cameras() if c2w is None else c2w
    fxfycxcy = torch.tensor([[60.0, 60.0, width * 4.0, 32.0]]).repeat(1, 5, 1)
    geom = prepare_unirope(c2w, fxfycxcy, (64, width * 8), (4, width // 2), footprint=True,
                           camera_model=camera_model, adaptive_freqs=True)
    return dict(unirope_c2w=geom['c2w'], unirope_w2c=geom['w2c'], unirope_directions=geom['directions'],
                unirope_patch_angle=geom['patch_angle'])


def trained_depth(dit):
    """Non-zero depth heads, so each token has its own depth."""
    torch.manual_seed(3)
    for block in dit.blocks:
        torch.nn.init.normal_(block.self_attn.unirope_depth.weight, std=0.01)
    return dit


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


class InstallTest(unittest.TestCase):
    def test_parameters_and_inputs(self):
        dit = tiny_wan()
        names = [n for n, _ in dit.named_parameters() if '.unirope_depth.' in n]
        self.assertEqual(len(names), 2 * len(dit.blocks))
        head = dit.blocks[0].self_attn.unirope_depth
        self.assertEqual(tuple(head.weight.shape), (2, 256))
        self.assertTrue(torch.equal(head.weight.cpu(), torch.zeros(2, 256)))
        self.assertEqual(head.bias.tolist(), [0.0, 3.0])
        for extra in ({'raymap': torch.zeros(1)}, {'scope_rays': torch.zeros(1)}):
            with self.assertRaises(ValueError):
                model_fn_unirope(dit, **wan_inputs(), **geometry(), **extra)
        with self.assertRaises(ValueError):
            model_fn_unirope(dit, **wan_inputs())
        with self.assertRaises(RuntimeError):
            install_unirope(dit)

    def test_needs_128_dim_heads(self):
        with self.assertRaises(ValueError):
            install_unirope(tiny_wan(dim=96, install=False))

    def test_token_camera_mismatch(self):
        with self.assertRaises(ValueError):
            model_fn_unirope(tiny_wan(), **wan_inputs(width=16), **geometry())


class CameraTest(unittest.TestCase):
    def run_model(self, dit, inputs, c2w, **kwargs):
        with torch.no_grad():
            return model_fn_unirope(dit, **inputs, **geometry(c2w, **kwargs))

    def test_world_frame_invariance(self):
        dit, inputs = trained_depth(tiny_wan()), wan_inputs()
        world = torch.eye(4)
        world[:3, :3] = rotation_y(0.7)
        world[:3, 3] = torch.tensor([3.0, -1.0, 2.0])
        moved = world @ cameras()
        # A world scale as well: the second ref sets the unit distance.
        moved[:, :, :3, 3] *= 2.5
        for camera_model in ('pinhole', 'equirect'):
            base = self.run_model(dit, inputs, cameras(), camera_model=camera_model)
            other = self.run_model(dit, inputs, moved, camera_model=camera_model)
            self.assertTrue(torch.allclose(base, other, atol=1e-3, rtol=1e-3), camera_model)

    def test_cameras_change_the_output(self):
        dit, inputs = trained_depth(tiny_wan()), wan_inputs()
        moved = cameras()
        moved[0, 3, 2, 3] += 0.5
        base = self.run_model(dit, inputs, cameras())
        self.assertFalse(torch.allclose(base, self.run_model(dit, inputs, moved), atol=1e-3, rtol=1e-3))

    def test_tokens_follow_their_cameras(self):
        """Swapping two query frames with their cameras swaps the output and nothing
        else; swapping the frames alone does not, so tokens are paired with their rays."""
        dit, inputs = trained_depth(tiny_wan()), wan_inputs()
        order = [0, 1, 3, 2, 4]
        base = self.run_model(dit, inputs, cameras())
        swapped_inputs = dict(inputs, latents=inputs['latents'][:, :, order], y=inputs['y'][:, :, order])
        both = self.run_model(dit, swapped_inputs, cameras()[:, order])
        self.assertTrue(torch.allclose(base[:, :, order], both, atol=1e-3, rtol=1e-3))
        frames_only = self.run_model(dit, swapped_inputs, cameras())
        self.assertFalse(torch.allclose(base[:, :, order], frames_only, atol=1e-3, rtol=1e-3))

    def test_adaptive_band_differs_per_camera_model(self):
        pinhole, equirect = geometry(), geometry(camera_model='equirect', width=16)
        self.assertEqual(tuple(pinhole['unirope_patch_angle'].shape), (1, 5))
        self.assertAlmostEqual(equirect['unirope_patch_angle'][0, 0].item(), torch.pi / 4, places=4)
        self.assertTrue((_direction_freqs(pinhole['unirope_patch_angle'], 4)[..., -1]
                         > _direction_freqs(equirect['unirope_patch_angle'], 4)[..., -1]).all())


class TrainingTest(unittest.TestCase):
    def test_checkpoint_across_token_counts(self):
        """The real run's sequence: gradient checkpointing and two token counts in turn.
        With torch.compile enabled, this also hits a shape-triggered recompile between backward passes."""
        dit = tiny_wan().train()
        for camera_model, width in (('pinhole', 8), ('equirect', 16), ('pinhole', 8), ('equirect', 16)):
            dit.zero_grad()
            prediction = model_fn_unirope(dit, **wan_inputs(width), **geometry(camera_model=camera_model, width=width),
                                          use_gradient_checkpointing=True)
            prediction.square().mean().backward()
            for block in dit.blocks:
                head = block.self_attn.unirope_depth
                for gradient in (head.weight.grad, head.bias.grad):
                    self.assertTrue(torch.isfinite(gradient).all(), camera_model)
                self.assertGreater(head.weight.grad.abs().sum().item(), 0.0, camera_model)

    def test_checkpointing_matches_plain_forward(self):
        dit, inputs = trained_depth(tiny_wan()).train(), wan_inputs()
        with torch.no_grad():
            plain = model_fn_unirope(dit, **inputs, **geometry())
        checkpointed = model_fn_unirope(dit, **inputs, **geometry(), use_gradient_checkpointing=True)
        self.assertTrue(torch.allclose(plain, checkpointed.detach(), atol=1e-5))

    def test_bf16_dit_keeps_fp32_geometry(self):
        dit = tiny_wan().bfloat16()
        self.assertEqual(dit.blocks[0].self_attn.unirope_depth.weight.dtype, torch.bfloat16)
        geom = unirope_geometry(*(torch.zeros(1, 5, 4, 4, dtype=torch.float64) for _ in range(4)), 'cpu')
        self.assertTrue(all(v.dtype == torch.float32 for v in geom.values()))
        with torch.no_grad():
            out = model_fn_unirope(dit, **wan_inputs(dtype=torch.bfloat16), **geometry())
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_depth_group_keeps_its_ratio_through_warmup(self):
        dit = tiny_wan()
        groups = optimizer_groups(dit, 4e-4, 0.5)
        names = sum(1 for n, p in dit.named_parameters() if p.requires_grad)
        self.assertEqual(len(groups[1]['params']), 2 * len(dit.blocks))
        self.assertEqual(len(groups[0]['params']) + len(groups[1]['params']), names)
        optimizer = torch.optim.AdamW(groups, lr=1e-5, weight_decay=0.01)
        self.assertEqual([g['weight_decay'] for g in optimizer.param_groups], [0.01, 0.5])
        warmup = 500
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda done: min(1.0, (done + 1) / warmup))
        for step in range(1, warmup + 10):
            lr, depth_lr = (g['lr'] for g in optimizer.param_groups)
            self.assertAlmostEqual(depth_lr / lr, 40.0, places=6)
            self.assertAlmostEqual(lr, 1e-5 * min(1.0, step / warmup), places=12)
            optimizer.step()
            scheduler.step()


class AdapterTest(unittest.TestCase):
    def test_pinhole_and_equirect_sets(self):
        for (height, width), camera_model in (((288, 512), 'pinhole'), ((256, 512), 'equirect')):
            batch = synthetic_batch(height, width, camera_model)
            data = to_omnicrafter_unirope(batch)
            self.assertEqual(data['camera_model'], camera_model)
            tokens = (height // 16) * (width // 16)
            self.assertEqual(tuple(data['unirope_directions'].shape), (5, tokens, 4, 3))
            self.assertEqual(tuple(data['unirope_c2w'].shape), (5, 4, 4))
            self.assertEqual(tuple(data['unirope_patch_angle'].shape), (5,))
            # The nearer ref (index 1, 0.5 from the queries) comes first and is the identity.
            self.assertTrue(torch.equal(TF.to_tensor(data['input_images'][0]) * 2 - 1, batch['ref'][0, 1]))
            self.assertTrue(torch.allclose(data['unirope_c2w'][0], torch.eye(4), atol=1e-5))
            # The other ref is at unit distance, so the near-depth scale cancels out.
            self.assertAlmostEqual(data['unirope_c2w'][1, :3, 3].norm().item(), 1.0, places=5)
            self.assertNotIn('scope_rays', data)
            self.assertEqual(len(data['target_images']), 5)
            black = to_omnicrafter_unirope(batch, black_queries=True)
            self.assertFalse(any(np.asarray(image).any() for image in black['conditioning_images']))


if __name__ == '__main__':
    unittest.main()
