"""CPU-only tests: reservation budgeting, ownership, cleanup, and diagnostics."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref
from runtime_guard import GIB, IdleMemoryReservation, LocalTrace, event, planned_reservation


class FakeAllocation:
    def __init__(self, size):
        self.size = size


class FakeCUDA:
    class OutOfMemoryError(Exception):
        pass

    def __init__(self):
        self.initial_free = 40 * GIB
        self.allocations = weakref.WeakSet()
        self.oom_at = None
        self.empty_cache_calls = 0

    def mem_get_info(self, device):
        return self.initial_free - sum(a.size for a in self.allocations), 96 * GIB

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    uint8 = 'uint8'

    def __init__(self):
        self.cuda = FakeCUDA()
        self.allocation_calls = []

    def empty(self, size, **kwargs):
        if self.cuda.oom_at is not None and len(self.cuda.allocations) >= self.cuda.oom_at:
            raise self.cuda.OutOfMemoryError()
        self.allocation_calls.append(kwargs)
        result = FakeAllocation(size)
        self.cuda.allocations.add(result)
        return result


class Tests(unittest.TestCase):
    def test_headroom_budget(self):
        self.assertEqual(planned_reservation(40 * GIB), 24 * GIB)
        self.assertEqual(planned_reservation(16 * GIB), 0)
        self.assertEqual(planned_reservation(12 * GIB), 0)
        self.assertEqual(planned_reservation(int(40.9 * GIB)), 24 * GIB)
        with self.assertRaises(ValueError):
            planned_reservation(40 * GIB, 0)

    def test_in_process_empty_allocation_and_release(self):
        fake = FakeTorch()
        guard = IdleMemoryReservation(fake, 'cuda:0')
        result = guard.acquire('gpu', lambda _: True)
        self.assertEqual(result['reserved_bytes'], 24 * GIB)
        self.assertEqual(result['free_bytes'], 16 * GIB)
        self.assertTrue(all(call == {'dtype': 'uint8', 'device': 'cuda:0', 'requires_grad': False}
                            for call in fake.allocation_calls))
        self.assertIsNone(guard.release_under_pressure())
        guard.release()
        self.assertEqual(guard.bytes, 0)
        self.assertEqual(fake.cuda.mem_get_info(None)[0], 40 * GIB)

    def test_never_competes_with_foreign_process(self):
        fake = FakeTorch()
        guard = IdleMemoryReservation(fake, 'cuda:0')
        self.assertEqual(guard.acquire('gpu', lambda _: False)['status'], 'skipped_shared_gpu')
        self.assertEqual(len(fake.allocation_calls), 0)
        self.assertEqual(guard.acquire('gpu', lambda _: True)['status'], 'already_attempted')

    def test_query_failure_skips_reservation(self):
        fake = FakeTorch()
        guard = IdleMemoryReservation(fake, 'cuda:0')
        def fail(_):
            raise OSError('query failed')
        self.assertEqual(guard.acquire('gpu', fail)['status'], 'skipped_ownership_unknown')
        self.assertFalse(fake.allocation_calls)

    def test_allocation_oom_releases_all_buffers(self):
        fake = FakeTorch()
        fake.cuda.oom_at = 2
        guard = IdleMemoryReservation(fake, 'cuda:0')
        self.assertEqual(guard.acquire('gpu', lambda _: True)['status'], 'released_after_reservation_oom')
        self.assertEqual(guard.bytes, 0)
        self.assertEqual(fake.cuda.mem_get_info(None)[0], 40 * GIB)

    def test_pressure_releases_without_reacquiring(self):
        fake = FakeTorch()
        guard = IdleMemoryReservation(fake, 'cuda:0')
        guard.acquire('gpu', lambda _: True)
        fake.cuda.initial_free -= 10 * GIB
        result = guard.release_under_pressure()
        self.assertEqual(result['released_bytes'], 24 * GIB)
        self.assertEqual(guard.bytes, 0)
        self.assertEqual(guard.acquire('gpu', lambda _: True)['status'], 'already_attempted')

    def test_trace_records_sample_and_stage_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('runtime_guard.faulthandler'):
                trace = LocalTrace(directory, 2)
                event('awaiting_batch', step=28001)
                event('image_read', path='/example.png', sample_id='sample')
                trace.close()
                records = [json.loads(line) for line in (Path(directory) / 'phases_rank2.jsonl').read_text().splitlines()]
                self.assertEqual(records[-1]['step'], 28001)
                self.assertEqual(records[-1]['stage'], 'image_read')
                self.assertEqual(records[-1]['pid'], os.getpid())
                self.assertEqual(records[-1]['sample_id'], 'sample')

    def test_scope_recipe_and_resume_guards(self):
        root = Path(__file__).resolve().parent
        train = (root / 'train.py').read_text()
        constants = (root / 'common.py').read_text()
        launcher = (root / 'run_io100.sh').read_text()
        queue = (root / 'queue_io100.py').read_text()
        self.assertIn('FINAL_STEP = 100000', constants)
        self.assertIn('WORLD_SIZE = 4', constants)
        self.assertIn('SAVE_STEPS = 2000', constants)
        self.assertIn('--architecture scope --initialization published', launcher)
        self.assertNotIn('OC_RESERVE_HEADROOM_GIB=', launcher)
        self.assertIn("os.environ.get('OC_RESERVE_HEADROOM_GIB','0')", queue)
        self.assertIn('len(checkpoint.keys()) == 1202', queue)
        self.assertIn('[5120,36,1,2,2]', queue)
        self.assertIn('Requested resume must be the latest checkpoint', queue)
        self.assertIn('lr=1e-5,weight_decay=.01', train)
        self.assertIn("assert zero['stage'] == 2", train)

    def test_gpu_zero2_config(self):
        import yaml
        root=Path(__file__).resolve().parent
        config=yaml.safe_load((root/'zero2.yaml').read_text())
        ds=json.loads(Path(config['deepspeed_config']['deepspeed_config_file']).read_text())
        self.assertEqual(config['num_processes'],4)
        self.assertEqual(ds['zero_optimization']['stage'],2)
        self.assertEqual(ds['zero_optimization']['offload_optimizer']['device'],'none')
        self.assertEqual(ds['zero_optimization']['offload_param']['device'],'none')
        self.assertEqual(ds['train_batch_size'],4)
        self.assertEqual(ds['train_micro_batch_size_per_gpu'],1)
        self.assertTrue(ds['bf16']['enabled'])
        self.assertFalse(ds['zero_optimization']['overlap_comm'])
        self.assertEqual(ds['zero_optimization']['reduce_bucket_size'],10000000)
        # With an external DS file, bf16 is specified on CLI to avoid Accelerate's
        # duplicate-config conflict. The effective setting is checked at runtime.
        self.assertNotIn('mixed_precision',config)
        queue=(root/'queue_io100.py').read_text()
        self.assertIn("'--mixed_precision','bf16'",queue)


if __name__ == '__main__':
    unittest.main(verbosity=2)
