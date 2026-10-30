"""No WAN, data mounts or GPU needed to test portable SCoPE dispatch."""
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
from handoff_common import camera_architecture, checkpoint_header, scope_parameter_shapes
from train_entry import training_arguments


def header(architecture='concat', initialization=False):
    data = {'patch_embedding.weight': {'shape': [5120,36 if architecture=='scope' else 420,1,2,2]},
            'patch_embedding.bias': {'shape': [5120]}}
    data.update({f'fake_lora_{i}': {'shape': [1]} for i in range(800)})
    if architecture == 'scope' and not initialization:
        data.update({k: {'shape': v} for k,v in scope_parameter_shapes().items()})
    data['__metadata__'] = {'step': '48000', 'camera_injection': architecture}
    return data


def write_header(path, data):
    payload = json.dumps(data).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack('<Q',len(payload))+payload)


class ScopePortabilityTests(unittest.TestCase):
    def test_explicit_selection_and_legacy_default(self):
        self.assertEqual(camera_architecture({}), 'concat')
        self.assertEqual(camera_architecture({'architecture':'scope'}), 'scope')
        for config in ({'architecture':'SCoPE'}, {'camera_injection':'scope'},
                       {'architecture':'scope','camera_injection':'concat'}):
            with self.assertRaises(ValueError):
                camera_architecture(config)

    def test_trained_scope_header_and_cross_architecture_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scope.safetensors'
            write_header(path, header('scope'))
            self.assertEqual(checkpoint_header(path,'scope')['step'], '48000')
            with self.assertRaises(ValueError):
                checkpoint_header(path,'concat')
            write_header(path, header('concat'))
            checkpoint_header(path,'concat')
            with self.assertRaises(ValueError):
                checkpoint_header(path,'scope')

    def test_image_only_initialization_is_not_a_trained_scope_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'init.safetensors'
            write_header(path, header('scope',initialization=True))
            checkpoint_header(path,'scope',initialization=True)
            with self.assertRaises(ValueError):
                checkpoint_header(path,'scope')

    def test_scope_names_shapes_and_metadata_fail_closed(self):
        good = header('scope')
        bad_name = copy.deepcopy(good)
        bad_name['blocks.0.self_attn.scope_pe.ev.weight'] = bad_name.pop('blocks.0.self_attn.scope_pe.eq.weight')
        bad_shape = copy.deepcopy(good)
        bad_shape['blocks.0.self_attn.scope_pe.eq.weight']['shape'] = [5120,6]
        bad_metadata = copy.deepcopy(good)
        bad_metadata['__metadata__']['camera_injection'] = 'concat'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.safetensors'
            for data in (bad_name,bad_shape,bad_metadata):
                write_header(path, data)
                with self.assertRaises(ValueError):
                    checkpoint_header(path,'scope')

    def test_training_arguments_keep_scope_and_converted_initialization(self):
        config = {'architecture':'scope','wan_root':'/new/wan','base_initialization':'/new/published',
            'prepared_initialization':'/new/run/initialization/image36.safetensors',
            'initial_step':48000,'restore_checkpoint':'/new/scope48000.safetensors'}
        arguments = training_arguments(config)
        self.assertEqual(arguments[arguments.index('--architecture')+1], 'scope')
        self.assertEqual(arguments[arguments.index('--base-initialization')+1], config['prepared_initialization'])
        self.assertEqual(arguments[arguments.index('--restore-checkpoint')+1], config['restore_checkpoint'])
        del config['prepared_initialization']
        with self.assertRaises(ValueError):
            training_arguments(config)
        config['architecture'] = 'concat'
        self.assertIn('/new/published', training_arguments(config))

    def test_scope_dry_run_never_converts_or_writes(self):
        import train_mixed
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root/'request.json'
            cfg.write_text(json.dumps({'architecture':'scope','output':str(root/'output')}))
            with patch.object(sys,'argv',['train_mixed','--config',str(cfg),'--dry-run']), \
                 patch.object(train_mixed,'validate'), \
                 patch('scope_initialization.prepare_scope_initialization') as convert, \
                 patch.object(train_mixed.subprocess,'call') as spawn:
                train_mixed.main()
            convert.assert_not_called()
            spawn.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_launcher_writes_effective_config_and_passes_scheduler_gpus(self):
        import train_mixed
        def convert(source, destination):
            write_header(destination, header('scope',initialization=True))
            return Path(destination)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {'architecture':'scope','output':str(root/'output'),
                'base_initialization':str(root/'published'),'wan_root':'/new/wan','initial_step':0}
            cfg = root/'request.json'; cfg.write_text(json.dumps(config))
            with patch.object(sys,'argv',['train_mixed','--config',str(cfg)]), \
                 patch.object(train_mixed,'validate'), patch.object(train_mixed,'protocol_hashes',return_value={}), \
                 patch('scope_initialization.prepare_scope_initialization',side_effect=convert) as conversion, \
                 patch.object(train_mixed.subprocess,'call',return_value=0) as spawn, \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b,GPU-c,GPU-d'}), \
                 self.assertRaises(SystemExit) as result:
                train_mixed.main()
            self.assertEqual(result.exception.code,0)
            conversion.assert_called_once()
            command = spawn.call_args.args[0]
            effective_path = root/'output/runtime/train_config.json'
            self.assertEqual(command[-1],str(effective_path))
            effective = json.loads(effective_path.read_text())
            self.assertEqual(effective['architecture'],'scope')
            args = training_arguments(effective)
            self.assertIn('framecrafter_image36.safetensors',args[args.index('--base-initialization')+1])
            self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-a,GPU-b,GPU-c,GPU-d')
            self.assertTrue((root/'output/portable_request.json').exists())

    def test_scope_evaluation_is_explicitly_not_supported(self):
        from evaluate import validate
        with self.assertRaisesRegex(ValueError,'SCoPE inference is not wired'):
            validate({'architecture':'scope'})


@unittest.skipUnless(os.environ.get('OC_TEST_TORCH')=='1','CPU tensor conversion test opt in')
class ScopeTensorTests(unittest.TestCase):
    def test_conversion_preserves_source_lora_bias_and_image_channels(self):
        import torch
        from scope_initialization import image_only_state
        state = {f'fake_lora_{i}':torch.tensor([i],dtype=torch.float32) for i in range(800)}
        state['patch_embedding.weight'] = torch.arange(5120*420*4,dtype=torch.float32).reshape(5120,420,1,2,2)
        state['patch_embedding.bias'] = torch.arange(5120,dtype=torch.float32)
        converted = image_only_state(state)
        self.assertEqual(set(converted),set(state))
        self.assertEqual(state['patch_embedding.weight'].shape[1],420)
        self.assertEqual(converted['patch_embedding.weight'].shape[1],36)
        torch.testing.assert_close(converted['patch_embedding.weight'],state['patch_embedding.weight'][:,:36],rtol=0,atol=0)
        self.assertTrue(converted['patch_embedding.weight'].is_contiguous())
        for key in state:
            if key != 'patch_embedding.weight':
                self.assertIs(converted[key],state[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
