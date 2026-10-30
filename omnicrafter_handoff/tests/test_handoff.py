import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from handoff_common import ROOT, EVAL_COUNTS, rebase, gpu_tokens, checkpoint_header
from relocate_manifest import rewrite


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((ROOT/'metadata/eval_manifest.json').read_text())['sets']

    def test_asset_relocation_preserves_every_nonpath_field(self):
        for dataset in EVAL_COUNTS:
            original = next(r for r in self.rows if r['dataset'] == dataset)
            changed = rewrite(original, {'/cis':'/new/assets/cis'})
            restored = rewrite(changed, {'/new/assets/cis':'/cis'})
            self.assertEqual(original, restored)
            self.assertNotEqual(original['references'][0]['image'], changed['references'][0]['image'])

    def test_longest_prefix_and_component_boundary(self):
        m = {'/a':'/x', '/a/b':'/y'}
        self.assertEqual(rebase('/a/b/file.png', m), '/y/file.png')
        self.assertEqual(rebase('/a/c/file.png', m), '/x/c/file.png')
        with self.assertRaises(ValueError):
            rebase('/ab/file.png', m)

    def test_unsafe_or_unmapped_path_rejected(self):
        for value in ('../a', '/a/../b', '/other/path', '/a/b\nfile'):
            with self.assertRaises(ValueError):
                rebase(value, {'/a':'/target'})

    def test_train_does_not_touch_provenance_addresses(self):
        row = copy.deepcopy(self.rows[0]); row['split'] = 'train'
        altered = rewrite(row, {'/cis':'/new/assets/cis'}, train=True)
        self.assertEqual(altered['prior_metadata'], row['prior_metadata'])
        self.assertEqual(altered.get('source_manifest'), row.get('source_manifest'))

    def test_wrong_train_split_rejected(self):
        row = copy.deepcopy(self.rows[0]); row['split'] = 'val'
        with self.assertRaises(ValueError):
            rewrite(row, {'/cis':'/new/cis'}, train=True)

    def test_projection_size_order_enforced(self):
        row = copy.deepcopy(self.rows[0]); row['height'], row['width'] = 512, 256
        with self.assertRaises(ValueError):
            rewrite(row, {'/cis':'/new/cis'})

    def test_scheduler_allocation_respected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b,GPU-c,GPU-d'}):
            self.assertEqual(gpu_tokens(2), ['GPU-a','GPU-b'])
            with self.assertRaises(ValueError):
                gpu_tokens(5)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):
                gpu_tokens(1)

    def test_header_rejects_scope_checkpoint(self):
        data = {'patch_embedding.weight': {'shape':[5120,36,1,2,2]}}
        payload = json.dumps(data).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.safetensors'
            path.write_bytes(struct.pack('<Q',len(payload))+payload)
            with self.assertRaises(ValueError):
                checkpoint_header(path)

    def test_training_binding_uses_new_paths_and_disables_memory_parking(self):
        import train_entry
        config = {'data_dir':'/new/data', 'output':'/new/run','final_step':60000,
                  'save_every':2000,'wan_root':'/new/wan','base_initialization':'/new/init'}
        bound = train_entry.bind(config)
        self.assertEqual(bound.DATA, Path('/new/data'))
        self.assertEqual(bound.OUT, Path('/new/run'))
        self.assertEqual(bound.FINAL_STEP, 60000)
        self.assertEqual(bound.GPU_UUIDS, [])

    def test_evaluation_binding_has_no_old_runtime_paths(self):
        import eval_entry
        config = {'checkpoint':'/new/weights/step.safetensors','output':'/new/out',
                  'wan_root':'/new/wan','cache_dir':'/new/cache','dreamsim_cache':'/new/dream',
                  'fid_weights':'/new/fid.pt','num_gpus':4}
        bound = eval_entry.bind(config)
        for name in ('CHECKPOINT','OUT','WAN','CACHE','DREAM_CACHE','FID_SOURCE'):
            self.assertTrue(str(getattr(bound, name)).startswith('/new/'))
        self.assertEqual(bound.NUM_SHARDS, 4)

    def test_jsonl_relocation_rebuilds_offsets(self):
        import numpy as np
        import relocate_manifest as tool
        row = copy.deepcopy(self.rows[0]); row['split'] = 'train'
        other = copy.deepcopy(row); other['id'] += '_another'
        with tempfile.TemporaryDirectory(prefix='omni_handoff_test_') as directory:
            root = Path(directory)
            source = root/'source.jsonl'; mapping = root/'mapping.json'
            source.write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
            mapping.write_text(json.dumps({'/cis':'/completely/different/new/location/cis'}))
            output = root/'new/train.jsonl'
            argv = ['relocate_manifest.py','--input',str(source),'--mapping',str(mapping),
                    '--output',str(output),'--kind','train']
            with patch.object(sys, 'argv', argv), patch.object(tool, 'COUNTS', {row['dataset']:2}):
                tool.main()
            offsets = np.load(root/'new/train.offsets.npy')
            self.assertEqual(len(offsets), 2)
            with output.open('rb') as f:
                for position, expected in zip(offsets, [row['id'], other['id']]):
                    f.seek(int(position))
                    self.assertEqual(json.loads(f.readline())['id'], expected)
            with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
                tool.main()

    def test_dry_runs_work_without_gpu_allocation_or_output_writes(self):
        import train_mixed
        import evaluate
        with tempfile.TemporaryDirectory(prefix='omni_handoff_dry_') as directory:
            root = Path(directory)
            config = root/'config.json'
            config.write_text(json.dumps({'output':str(root/'not_created'),'num_gpus':4}))
            args = ['wrapper','--config',str(config),'--dry-run']
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}), patch.object(sys,'argv',args):
                with patch.object(train_mixed,'validate'):
                    train_mixed.main()
                with patch.object(evaluate,'validate',return_value=[]):
                    evaluate.main()
            self.assertFalse((root/'not_created').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
