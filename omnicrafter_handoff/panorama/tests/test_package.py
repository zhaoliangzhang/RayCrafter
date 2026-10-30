"""CPU tests only: no WAN weights, CUDA inference, or network downloads."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from common import check_resolution, read_manifest, sha256, wan_assets
from make_toy_data import create
from train import build_command, parser
from diffsynth.core.data.pose_conditioned_render_query_dataset import WanPanoramaPoseConditionedRenderQueryDataset as Dataset
from diffsynth.core.data.dataset import normalize_w2c_make_cam_last_origin, get_equirectangular_plucker_rays


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        create(self.root)

    def dataset(self, **kw):
        return Dataset(base_path=self.root, metadata_path=self.root / "train.jsonl",
                       pose_manifest_path=self.root / "poses/manifest.json",
                       height=32, width=64, deterministic_degradation=True, **kw)

    def test_train_shape_and_order(self):
        sample = self.dataset()[0]
        self.assertEqual(tuple(sample["raymap"].shape), (5, 384, 4, 8))
        self.assertEqual(sample["slot_frame_indices"], [0, 4, 1, 2, 3])
        self.assertEqual(len(sample["target_images"]), 5)
        self.assertTrue(torch.isfinite(sample["raymap"]).all())

    def test_stream_matches_eager(self):
        streamed = self.dataset(stream_metadata=True)
        a, b = self.dataset()[0], streamed[0]
        self.addCleanup(streamed._metadata_handle.close)
        self.assertTrue(torch.equal(a["raymap"], b["raymap"]))
        for x, y in zip(a["conditioning_images"], b["conditioning_images"]):
            np.testing.assert_array_equal(x, y)

    def test_query_nvs_endpoint(self):
        sample = self.dataset(nvs_ratio=1)[0]
        self.assertEqual(sample["target_task_types"], ["nvs"] * 3)
        self.assertTrue(all(not np.asarray(im).any() for im in sample["conditioning_images"]))
        self.assertTrue(np.asarray(sample["target_images"][-1]).any())

    def test_sample_nvs_endpoint(self):
        sample = self.dataset(nvs_probability=1)[0]
        self.assertEqual(sample["query_pixel_dropout_ratios"], [1, 1, 1])

    def test_cleanup_zero_drop_preserves_render(self):
        from PIL import Image
        sample = self.dataset(max_pixel_dropout=0)[0]
        for im, frame in zip(sample["conditioning_images"], (1, 2, 3)):
            with Image.open(self.root / f"noisy/{frame:06d}.png") as noisy:
                np.testing.assert_array_equal(im, noisy)

    def test_cleanup_dropout_can_reach_black(self):
        sample = self.dataset(min_pixel_dropout=1, max_pixel_dropout=1)[0]
        self.assertTrue(all(not np.asarray(im).any() for im in sample["conditioning_images"]))
        self.assertEqual(sample["query_pixel_dropout_ratios"], [1, 1, 1])

    def test_rays_at_release_resolution(self):
        sample = self.dataset()[0]
        poses = torch.eye(4).repeat(5, 1, 1)
        poses[:, 0, 3] = torch.arange(5)
        _, c2w, _ = normalize_w2c_make_cam_last_origin(poses)
        rays = get_equirectangular_plucker_rays(c2w, height=256, width=512)
        self.assertEqual(tuple(rays.shape), (5, 384, 32, 64))
        self.assertTrue(torch.isfinite(rays).all())

    def test_no_source_mutation(self):
        before = sha256(self.root / "noisy/000001.png")
        self.dataset(nvs_ratio=1)[0]
        self.assertEqual(before, sha256(self.root / "noisy/000001.png"))

    def test_independent_queries_can_mix(self):
        ds = self.dataset(repeat=30, nvs_ratio=0.5, max_pixel_dropout=0.99)
        tasks = [ds[i]["sample_task_type"] for i in range(len(ds))]
        self.assertIn("mixed_cleanup_nvs", tasks)

    def test_reject_combined_switches(self):
        with self.assertRaises(ValueError):
            self.dataset(nvs_ratio=0.5, nvs_probability=0.5)

    def test_query_limit(self):
        self.assertEqual(tuple(self.dataset(max_query_slots=1)[0]["raymap"].shape), (3, 384, 4, 8))

    def test_pose_anchor(self):
        poses = torch.eye(4).repeat(5, 1, 1)
        poses[:, 0, 3] = torch.arange(5)
        _, c2w, _ = normalize_w2c_make_cam_last_origin(poses)
        self.assertTrue(torch.allclose(c2w[-1], torch.eye(4)))
        self.assertAlmostEqual(float(c2w[:, :3, 3].norm(dim=-1).mean()), 1)

    def test_gt_not_required_for_inference(self):
        manifest = json.loads((self.root / "eval.json").read_text())
        for view in manifest["sets"][0]["targets"]:
            view["image"] = "nonexistent_gt.png"
        path = self.root / "no_gt.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(len(read_manifest(path, "cleanup")["sets"]), 1)

    def test_invalid_resolution(self):
        with self.assertRaises(ValueError):
            check_resolution(176, 354)

    def test_reject_unsafe_id(self):
        data = json.loads((self.root / "eval.json").read_text())
        data["sets"][0]["id"] = "../escape"
        (self.root / "bad.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            read_manifest(self.root / "bad.json")

    def test_train_command_global_steps(self):
        args = parser().parse_args(["--data-root", str(self.root), "--train-manifest", str(self.root / "train.jsonl"),
            "--pose-manifest", str(self.root / "poses/manifest.json"), "--wan-root", "/weights", "--init-checkpoint", "/adapter.safetensors",
            "--output", str(self.root / "out"), "--steps", "400000", "--initial-step", "250000"])
        cmd = build_command(args, ["shards"], "tokenizer")
        self.assertEqual(cmd[cmd.index("--max_train_steps") + 1], "650000")
        self.assertEqual(cmd[cmd.index("--nvs_ratio") + 1], "0.5")
        self.assertIn("--initialize_model_on_cpu", cmd)
        self.assertNotIn("--use_gradient_checkpointing_offload", cmd)

    def test_missing_wan_fails(self):
        with self.assertRaises(FileNotFoundError):
            wan_assets(self.root)

    def test_inference_cli_wiring_with_mock_backbone(self):
        import infer
        calls = []
        class FakeModel:
            def __init__(self, **kwargs):
                pass
            def generate_render_queries(self, refs, queries, raymap, **kwargs):
                calls.append((refs, queries, raymap, kwargs))
                return queries
        fake = types.ModuleType("model_pose_conditioned_render_queries")
        fake.PoseConditionedRenderQueryCrafter = FakeModel
        for task in ("nvs", "cleanup"):
            args = ["infer.py", "--manifest", str(self.root / "eval.json"), "--task", task,
                    "--checkpoint", str(self.root / "train.jsonl"), "--wan-root", str(self.root),
                    "--output", str(self.root / task), "--height", "32", "--width", "64", "--device", "cpu"]
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {"model_pose_conditioned_render_queries": fake}), patch.object(infer, "wan_assets"):
                infer.main()
            self.assertTrue((self.root / task / "_SUCCESS").is_file())
            self.assertEqual(len(list((self.root / task / "color/demo_set").glob("*.png"))), 3)
        self.assertTrue(all(not np.asarray(im).any() for im in calls[0][1]))
        self.assertTrue(all(np.asarray(im).any() for im in calls[1][1]))
        self.assertTrue(torch.equal(calls[0][2], calls[1][2]))


if __name__ == "__main__":
    unittest.main()
