"""Regression tests for all-black cleanup sources, without loading WAN/CUDA."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from dataset import MixedProjectionDataset


class DatasetResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cameras = patch('dataset.ordered_cameras', side_effect=lambda row: (
            row, {'row_id': row['id'], 'near_depth': row['near_depth']}))
        cameras.start()
        self.addCleanup(cameras.stop)
        random.seed(20261103)

    def make_dataset(self, specifications, probability=0.):
        rows = []
        for i, (dataset, scene, nonempty) in enumerate(specifications):
            views = []
            for j in range(5):
                clean = self.root / f'{i}_{j}_clean.png'
                noisy = self.root / f'{i}_{j}_noisy.png'
                Image.new('RGB', (8, 4), (i*30+j+1, 10, 20)).save(clean)
                Image.new('RGB', (8, 4), (80, 20, 10) if nonempty else (0, 0, 0)).save(noisy)
                views.append({'image': str(clean), 'noisy': str(noisy)})
            rows.append({'id': f'sample{i}', 'dataset': dataset, 'scene_name': scene,
                         'projection': 'panorama', 'height': 256, 'width': 512,
                         'split': 'train', 'near_depth': i+1.,
                         'references': views[:2], 'targets': views[2:]})
        manifest = self.root / 'train.jsonl'
        offsets = self.root / 'offsets.npy'
        positions = []
        with manifest.open('wb') as stream:
            for row in rows:
                positions.append(stream.tell())
                stream.write((json.dumps(row)+'\n').encode())
        np.save(offsets, np.asarray(positions, dtype=np.int64))
        ds = MixedProjectionDataset(manifest, offsets, probability)
        self.addCleanup(lambda: ds._handle.close() if ds._handle else None)
        return ds

    def test_same_scene_replacement_keeps_whole_sample_and_single_task_draw(self):
        ds = self.make_dataset([('HM3D', 'sceneA', False), ('HM3D', 'sceneA', True),
                                ('HM3D', 'sceneB', True)])
        before = ds.manifest.read_bytes()
        with patch('dataset.random.random', return_value=.9) as draw, patch('builtins.print'):
            sample = ds[0]
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(sample['sample_id'], 'sample1')
        self.assertEqual(sample['original_sample_id'], 'sample0')
        self.assertEqual(sample['camera_inputs'], {'row_id':'sample1', 'near_depth':2.})
        self.assertEqual(sample['input_images'][0].getpixel((0, 0))[0], 31)
        self.assertEqual(sample['target_images'][2].getpixel((0, 0))[0], 33)
        self.assertTrue(sample['cleanup_source_resampled'])
        self.assertFalse(sample['cleanup_scene_changed'])
        self.assertEqual(sample['sample_task_type'], 'cleanup')
        self.assertFalse(sample['all_queries_black'])
        self.assertEqual(before, ds.manifest.read_bytes())
        self.assertEqual(len(ds), 3)

    def test_fallback_keeps_dataset_and_resolution(self):
        ds = self.make_dataset([('JHU', 'sceneA', False), ('JHU', 'sceneB', True),
                                ('HM3D', 'sceneA', True)])
        with patch('builtins.print'):
            sample = ds[0]
        self.assertEqual((sample['sample_id'], sample['dataset']), ('sample1', 'JHU'))
        self.assertTrue(sample['cleanup_scene_changed'])
        self.assertEqual(sample['conditioning_images'][0].size, (512, 256))

    def test_black_source_remains_available_to_nvs(self):
        ds = self.make_dataset([('HM3D', 'sceneA', False)], probability=1.)
        sample = ds[0]
        self.assertEqual(sample['sample_id'], 'sample0')
        self.assertTrue(sample['all_queries_black'])
        self.assertFalse(sample['cleanup_source_resampled'])
        self.assertIsNone(ds._cleanup_groups)

    def test_valid_cleanup_never_resamples(self):
        ds = self.make_dataset([('HM3D', 'sceneA', True)])
        sample = ds[0]
        self.assertEqual(sample['sample_id'], 'sample0')
        self.assertFalse(sample['cleanup_source_resampled'])
        self.assertFalse(sample['all_queries_black'])

    def test_no_cross_dataset_fallback_when_pool_is_empty(self):
        ds = self.make_dataset([('JHU', 'sceneA', False), ('HM3D', 'sceneA', True)])
        with self.assertRaisesRegex(RuntimeError, 'No nonempty cleanup source'):
            ds[0]

    def test_missing_render_is_not_silently_replaced(self):
        ds = self.make_dataset([('HM3D', 'sceneA', True), ('HM3D', 'sceneA', True)])
        with patch.object(ds, 'load_image', side_effect=FileNotFoundError('missing')):
            with self.assertRaises(FileNotFoundError):
                ds[0]

    def test_validation_rows_are_never_replacement_candidates(self):
        ds = self.make_dataset([('HM3D', 'sceneA', False), ('HM3D', 'sceneA', True)])
        lines = ds.manifest.read_text().splitlines()
        last = json.loads(lines[1]); last['split'] = 'val'
        ds.manifest.write_text(lines[0]+'\n'+json.dumps(last)+'\n')
        with self.assertRaisesRegex(ValueError, 'validation/test'):
            ds.prepare_cleanup_groups()


if __name__ == '__main__':
    unittest.main(verbosity=2)
