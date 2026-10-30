"""Optional read-only preflight against original mounts; not needed on a new cluster."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


@unittest.skipUnless(os.environ.get('OC_VERIFY_ORIGINAL_MOUNTS') == '1', 'Original-mount checks opt in only')
class OriginalMountPreflight(unittest.TestCase):
    def test_training_preflight_no_model_load(self):
        from train_mixed import validate
        base = '/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace'
        validate({'data_dir':'/cis/net/io104a/data/xliu253/omnicrafter_mixed_local_cache/data_manifest',
                  'wan_root':base+'/checkpoints/wan21',
                  'base_initialization':base+'/weights/framecrafter_released/framecrafter.safetensors',
                  'initial_step':0,'final_step':300000,'save_every':2000,'num_gpus':4,'batch_per_gpu':1})

    def test_evaluation_preflight_no_model_load(self):
        from evaluate import validate
        from handoff_common import ROOT
        base = '/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace'
        out = '/cis/net/io81/data/xliu253/omniraycrafter/mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104'
        teacher = '/cis/net/io81/data/xliu253/framecrafter/OC_2ref3_250000_hm3d_jhu_cleanup_nvs_256x512_continue/evaluations/step650000_hm3d_replica_jhu_q3_256x512_20260917'
        validate({'manifest':str(ROOT/'metadata/eval_manifest.json'),
                  'checkpoint':out+'/step-50000.safetensors', 'wan_root':base+'/checkpoints/wan21',
                  'fid_weights':'/cis/net/io96/data/xliu253/oc650k_eval_cache/fid/inception-2015-12-05.pt',
                  'metric_python':teacher+'/runtime/dreamsim_env/bin/python',
                  'dreamsim_cache':teacher+'/runtime', 'num_gpus':4})

    def test_scope_training_preflight_and_restore_no_model_load(self):
        from train_mixed import validate
        base = '/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace'
        config = {'architecture':'scope',
            'data_dir':'/cis/net/io104a/data/xliu253/omnicrafter_mixed_local_cache/data_manifest',
            'wan_root':base+'/checkpoints/wan21',
            'base_initialization':base+'/weights/framecrafter_released/framecrafter.safetensors',
            'initial_step':0,'final_step':300000,'save_every':2000,'num_gpus':4,'batch_per_gpu':1}
        validate(config)
        config.update(initial_step=48000,restore_checkpoint=
            '/cis/net/io81/data/xliu253/omniraycrafter/ORC_scope_qk_2ref3_hm3d_jhu_nvs50_256x512_b1_100k_io104/step-48000.safetensors')
        validate(config)

    def test_scope_conversion_of_real_published_adapter(self):
        import tempfile
        import torch
        from safetensors.torch import load_file
        from scope_initialization import prepare_scope_initialization
        from handoff_common import PUBLISHED_SHA256, checkpoint_header, sha256
        source = Path('/cis/home/xliu253/Documents/PanoLagerNVS/panorama_nvs_transfer_20260818/workspace/weights/framecrafter_released/framecrafter.safetensors')
        with tempfile.TemporaryDirectory(prefix='oc_scope_conversion_') as folder:
            destination = Path(folder)/'image36.safetensors'
            prepare_scope_initialization(source,destination)
            checkpoint_header(destination,'scope',initialization=True)
            original = load_file(str(source),device='cpu')
            actual = load_file(str(destination),device='cpu')
            self.assertEqual(set(original),set(actual))
            for key,value in original.items():
                expected = value[:,:36] if key == 'patch_embedding.weight' else value
                self.assertTrue(torch.equal(actual[key],expected),key)
            self.assertEqual(sha256(source),PUBLISHED_SHA256)
            self.assertFalse(torch.cuda.is_initialized())
            before = sha256(destination)
            with self.assertRaises(FileExistsError):
                prepare_scope_initialization(source,destination)
            self.assertEqual(sha256(destination),before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
