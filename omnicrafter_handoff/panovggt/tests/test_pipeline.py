"""CPU tests using synthetic images and a fake predictor; no learned inference."""
from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import align_cameras, render_erp
from render_priors import (completed, load_manifest, lock_run_contract, process_set,
                           sample_signature, validate_benchmark)
from build_manifest import build


def camera(x: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = x
    return result


def original_geometry() -> dict:
    path = ROOT / "provenance/original/render_jhu_heldout.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"project_to_so3", "align_cameras", "render_erp"}]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class GeometryTests(unittest.TestCase):
    def test_recovers_known_similarity(self):
        predicted = np.stack([camera(0), camera(1)])
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        translation = np.array([2., -3., 4.])
        truth = predicted.copy()
        truth[:, :3, :3] = rotation
        truth[:, :3, 3] = 2.5 * (predicted[:, :3, 3] @ rotation.T) + translation
        scale, r, t, diagnostics = align_cameras(predicted, truth)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(r, rotation)
        np.testing.assert_allclose(t, translation)
        np.testing.assert_allclose(diagnostics["condition_center_errors"], 0)

    def test_geometry_exactly_matches_original(self):
        original = original_geometry()
        predicted = np.stack([camera(0), camera(1)])
        truth = np.stack([camera(1), camera(3)])
        a, b = align_cameras(predicted, truth), original["align_cameras"](predicted, truth)
        self.assertEqual(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(a[3], b[3])
        rng = np.random.default_rng(7)
        points, colors = rng.normal(size=(2000, 3)), rng.random((2000, 3))
        a = render_erp(points, colors, camera(.3), 16, 32)
        b = original["render_erp"](points, colors, camera(.3), 16, 32)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)

    def test_nearest_radial_depth_wins_and_holes_stay_black(self):
        rgb, depth, mask = render_erp(np.array([[0, 0, 2], [0, 0, 1.]]),
                                     np.array([[1, 0, 0], [0, 1., 0]]), np.eye(4), 4, 8)
        np.testing.assert_array_equal(rgb[2, 4], [0, 255, 0])
        self.assertEqual(depth[2, 4], 1.)
        self.assertEqual(np.count_nonzero(mask), 1)
        self.assertTrue((rgb[mask == 0] == 0).all())

    def test_zero_predicted_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            align_cameras(np.stack([camera(0), camera(0)]), np.stack([camera(0), camera(1)]))

    def test_empty_cloud(self):
        rgb, depth, mask = render_erp(np.zeros((0, 3)), np.zeros((0, 3)), np.eye(4), 4, 8)
        self.assertFalse(mask.any())
        self.assertFalse(rgb.any())
        self.assertTrue(np.isnan(depth).all())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="panovggt_package_test_")
        self.root = Path(self.temporary.name)
        self.paths = [self.root / f"ref{i}.png" for i in range(2)]
        for path in self.paths:
            Image.new("RGB", (16, 8), "orange").save(path)
        self.item = {
            "id": "sample_000000", "scene_name": "synthetic", "source_sample": "synthetic",
            "references": [{"image": str(p), "frame_index": i, "w2c": camera(-i).tolist()}
                           for i, p in enumerate(self.paths)],
            "targets": [{"frame_index": i + 2, "target_number": i, "w2c": camera(-.2 * i).tolist(),
                         "image": "/TARGET_RGB_MUST_NEVER_BE_OPENED.png"} for i in range(3)],
        }
        self.manifest = {"pose_convention": "OpenCV world-to-camera", "resolution": {"height": 8, "width": 16},
                         "logical_sets": 1, "target_images": 3, "sets": [self.item]}

    def tearDown(self):
        self.temporary.cleanup()

    def test_target_rgb_not_required(self):
        validate_benchmark(self.manifest, 8, 16)
        for target in self.item["targets"]:
            target.pop("image")
        validate_benchmark(self.manifest, 8, 16)

    def test_bad_pose_rejected(self):
        self.item["references"][0]["w2c"][0][0] = 2.
        with self.assertRaises(ValueError):
            validate_benchmark(self.manifest, 8, 16)

    def test_relative_paths(self):
        for ref in self.item["references"]:
            ref["image"] = Path(ref["image"]).name
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        result = load_manifest(self.root)
        validate_benchmark(result, 8, 16)
        self.assertEqual(result["sets"][0]["references"][0]["image"], str(self.paths[0]))

    def test_reference_content_changes_signature(self):
        signature = sample_signature(self.item)
        Image.new("RGB", (16, 8), "green").save(self.paths[0])
        self.assertNotEqual(signature, sample_signature(self.item))

    def test_contract_rejects_changed_checkpoint(self):
        lock_run_contract(self.root, {"checkpoint": "a"})
        lock_run_contract(self.root, {"checkpoint": "a"})
        with self.assertRaises(ValueError):
            lock_run_contract(self.root, {"checkpoint": "b"})

    def test_build_selects_exact_frames(self):
        sequence = self.root / "sequence.json"
        frames = copy.deepcopy(self.item["references"] + self.item["targets"])
        sequence.write_text(json.dumps({"pose_convention": "OpenCV world-to-camera", "frames": frames}))
        result = build(sequence, [0, 1], [4, 2], 8, 16)
        self.assertEqual([v["frame_index"] for v in result["sets"][0]["targets"]], [4, 2])
        with self.assertRaises(ValueError):
            build(sequence, [0, 1], [0], 8, 16)

    def test_fake_predictor_end_to_end_and_resume(self):
        paths = self.paths
        class FakePredictor:
            def predict_from_paths(self, actual):
                assert actual == paths, "Only reference images may enter the predictor"
                return {"camera_poses": np.stack([camera(0), camera(1)])[None],
                        "world_points": np.array([[[[[0., 0., 2.]]], [[[1., 0., 2.]]]]]),
                        "images": np.ones((1, 2, 3, 1, 1)) * .5}
        record = process_set(self.item, FakePredictor(), self.root / "output", 8, 16, 1e-4, None)
        self.assertFalse(record["target_rgb_accessed"])
        self.assertEqual(len(record["targets"]), 3)
        directory = self.root / "output/samples/sample_000000"
        self.assertTrue(completed(directory, 16, 8, 3, sample_signature(self.item)))
        self.assertFalse(completed(directory, 16, 8, 3, "changed"))
        with Image.open(directory / "prior_0.png") as image:
            self.assertEqual(image.size, (16, 8))

    def test_calibrated_degenerate_pair_uses_external_scale(self):
        self.item["references"][1]["w2c"] = camera(0).tolist()
        class FakePredictor:
            def predict_from_paths(self, paths):
                return {"camera_poses": np.stack([camera(0), camera(1)])[None],
                        "world_points": np.array([[[[[0., 0., 2.]]], [[[1., 0., 2.]]]]]),
                        "images": np.ones((1, 2, 3, 1, 1)) * .5}
        record = process_set(self.item, FakePredictor(), self.root / "fallback", 8, 16, 1e-4, 2.)
        self.assertEqual(record["alignment"]["scale"], 2.)
        self.assertEqual(record["input_signature"], sample_signature(self.item, 2.))
        self.assertNotEqual(sample_signature(self.item, 2.), sample_signature(self.item, 3.))

    def test_degenerate_pair_requires_fallback(self):
        self.item["references"][1]["w2c"] = camera(0).tolist()
        with self.assertRaises(ValueError):
            process_set(self.item, None, self.root / "unused", 8, 16, 1e-4, None)


if __name__ == "__main__":
    unittest.main()
