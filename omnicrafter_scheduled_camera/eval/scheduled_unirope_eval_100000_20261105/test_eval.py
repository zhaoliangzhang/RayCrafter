"""CPU checkpoint, matched protocol and camera geometry regression tests."""
from collections import Counter
import unittest
from eval_shared import *
imports()
import torch
from safetensors import safe_open
from camera import ordered_cameras,prepare_camera_context
from contract import check_contract,verify_vendor
from test_scheduled import ScheduledTests

class EvaluationTests(unittest.TestCase):
    def test_full_matched_manifest(self):
        self.assertEqual(sha256(OUT/'benchmark/manifest.json'),sha256(MATCHED/'benchmark/manifest.json'))
        rows=manifest()['sets']
        self.assertEqual(Counter(r['dataset'] for r in rows),COUNTS)
        self.assertEqual(len(rows),3791)
        indices=[r['index'] for r in rows]
        self.assertEqual(indices,list(range(3791)))
        self.assertEqual(sum(len([i for i in indices if i%NUM_SHARDS==k]) for k in range(NUM_SHARDS)),len(rows))
        self.assertEqual(code_hashes(),json.loads((OUT/'provenance.json').read_text())['code_sha256'])
    def test_checkpoint_and_schedule(self):
        verify_vendor()
        with safe_open(str(CHECKPOINT),framework='pt',device='cpu') as f:
            check_contract(f.metadata())
            self.assertEqual(int(f.metadata()['step']),EXPECTED_STEP)
            self.assertEqual(len(f.keys()),EXPECTED_KEYS)
            self.assertEqual(json.loads(f.metadata()['injection_schedule']),dict(base_step=50000,completed_updates=50000,ramp_steps=10000,type='linear',weight=1.))
    def test_geometry_and_inputs(self):
        for row in smoke_rows(manifest()['sets']):
            ordered,cameras=ordered_cameras(dict(row,near_depth=near_depth_for(row)))
            self.assertEqual(ordered['references'],row['references'])
            context=prepare_camera_context(cameras,'cpu')
            torch.testing.assert_close(context['normalized_c2w'][0,-1],torch.eye(4),atol=3e-6,rtol=0)
            self.assertTrue(torch.isfinite(context['raymap']).all())
            self.assertEqual(context['raymap'].shape,(1,384,5,row['height']//8,row['width']//8))
            from worker import load_inputs
            refs,nvs=load_inputs(row,'nvs')
            self.assertTrue(all(im.getbbox() is None for im in nvs))
            self.assertEqual(len(refs),2)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
