"""Portable launcher for the near-depth mixed-projection SCoPE run (snapshot/ = io100's code).

Replaces the io100-only queue_io100.py / run_io100.sh: the same input checks, initialization
and preflight, then one foreground 4-GPU accelerate launch of the untouched snapshot/train.py.
Running it again with the same config continues in the same output directory from the latest
step checkpoint (adapter weights and step; optimizer, scheduler and data order restart, as on
io100). Once final_step is saved it only reports completion.

    python omnicrafter_neardepth/tools/train_neardepth.py --config train.local.json [--dry-run | --preflight-only]
"""
import argparse
from datetime import datetime
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT/'snapshot'
sys.path.insert(0, str(ROOT.parent/'omnicrafter_handoff/tools'))
sys.path.insert(0, str(SNAPSHOT))
from handoff_common import PUBLISHED_SHA256, checkpoint_header, gpu_tokens, required_file
from common import *

KEYS = {'data_dir', 'output', 'wan_root', 'base_initialization', 'final_step', 'save_every', 'preflight', 'triton_cache'}
TESTS = ('test_runtime_guard.py', 'test_protocol.py', 'test_frozen_encoders.py')


def load_config(path):
    config = {'final_step': 100000, 'save_every': 2000, 'preflight': True, 'triton_cache': None,
              **json.loads(Path(path).read_text())}
    if set(config) - KEYS:
        raise ValueError(f'Unknown config keys: {sorted(set(config) - KEYS)}')
    final, every = config['final_step'], config['save_every']
    if not 0 < every <= final or final % every:
        raise ValueError('Use a positive save_every that divides final_step')
    return config


def validate(config):
    data = Path(config['data_dir'])
    summary = json.loads((data/'summary.json').read_text())
    assert (data/'_SUCCESS').is_file(), 'Run tools/prepare_manifest.py first'
    assert summary['dataset_counts'] == COUNTS and summary['logical_sets'] == 200000
    assert summary['projection_counts'] == {'panorama': 100000, 'perspective': 100000}
    assert sha256(data/'train.jsonl') == summary['sha256'], 'train.jsonl checksum mismatch'
    assert sha256(data/'train.offsets.npy') == summary['offsets_sha256'], 'Rebuild offsets after any JSONL change'
    assert summary['pose_normalization'] == POSE_NORMALIZATION
    assert summary['near_depth_coverage_sets'] == 200000 and summary['near_depth_missing_scenes'] == 0
    from prepare_manifest import IO100_CANONICAL_SHA256
    assert summary.get('canonical_sha256') == IO100_CANONICAL_SHA256, 'Not the io100 near-depth manifest'
    for name, digest in summary['near_depth_source_sha256'].items():
        assert sha256(data/'near_depth'/f'{name}.json') == digest
    base = required_file(config['base_initialization'])
    checkpoint_header(base)
    assert sha256(base) == PUBLISHED_SHA256, 'base_initialization must be the published FrameCrafter adapter'
    model_paths(config['wan_root'])
    ds = json.loads((SNAPSHOT/'deepspeed_zero2.json').read_text())
    zero = ds['zero_optimization']
    assert zero['stage'] == 2 and zero['offload_optimizer']['device'] == zero['offload_param']['device'] == 'none'
    assert ds['train_batch_size'] == GLOBAL_BATCH and ds['train_micro_batch_size_per_gpu'] == BATCH
    return summary


def request(config, summary):
    tools = {p.name: sha256(p) for p in sorted((ROOT/'tools').glob('*.py'))}
    return {'config': config, 'snapshot_sha256': code_hashes(), 'tools_sha256': tools,
            'manifest_sha256': summary['sha256'], 'canonical_manifest_sha256': summary['canonical_sha256'],
            'base_initialization_sha256': PUBLISHED_SHA256, 'camera_injection': 'scope',
            'pose_normalization': POSE_NORMALIZATION, 'frozen_encoder_policy': FROZEN_ENCODER_POLICY,
            'nvs_probability': .5, 'world_size': WORLD_SIZE, 'global_batch_size': GLOBAL_BATCH}


def latest_checkpoint(output):
    checkpoints = sorted(output.glob('step-*.safetensors'), key=lambda p: int(p.stem.split('-')[1]))
    if not checkpoints:
        return None, 0
    meta = checkpoint_header(checkpoints[-1], 'scope')
    step = int(checkpoints[-1].stem.split('-')[1])
    assert int(meta['step']) == step and meta['pose_normalization'] == POSE_NORMALIZATION
    return checkpoints[-1], step


def preflight(config, output, run_id):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(REPO),
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    checks = [(name, [str(SNAPSHOT/name)]) for name in TESTS]
    checks.append(('check_data.py', [str(ROOT/'tools/entry.py'), '--config', str(output/'runtime/train_config.json'),
                                     '--script', 'check_data', '--', '--architecture', 'scope']))
    for name, arguments in checks:
        log = output/f'logs/{Path(name).stem}_{run_id}.log'
        print(f'{now()} preflight {name} (log {log})', flush=True)
        with log.open('x') as handle:
            subprocess.run([sys.executable, '-B', '-u', *arguments], cwd=SNAPSHOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    write_json(output/f'runtime/preflight_{run_id}.json', {'time': now(), 'passed': True})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Check inputs only; create nothing')
    p.add_argument('--preflight-only', action='store_true', help='Prepare the output and run the CPU preflight, no training')
    args = p.parse_args()
    config = load_config(args.config)
    summary = validate(config)
    output = Path(config['output'])
    wanted = request(config, summary)
    if output.exists():
        recorded = json.loads((output/'portable_request.json').read_text())
        assert recorded['request'] == wanted, 'Config, code or data differ from this output directory; use a new output'
        init = output/'initialization/framecrafter_image36.safetensors'
        assert sha256(init) == recorded['initialization_sha256']
    elif args.dry_run:
        print('Inputs verified. A new run would start at step 0 in ' + str(output))
        return
    restore, step = latest_checkpoint(output) if output.exists() else (None, 0)
    if step >= config['final_step']:
        assert json.loads((output/'_SUCCESS').read_text())['step'] == config['final_step']
        print(f'Complete: step-{step}.safetensors in {output}')
        return
    if args.dry_run:
        print(f'Inputs verified. Would continue from {restore or "the initialization"} at step {step} in {output}')
        return
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    fresh = not output.exists()
    if fresh:
        if shutil.disk_usage(output.parent).free < 60 * 1024**3:
            raise RuntimeError('Need at least 60 GiB free for 50 checkpoints')
        output.mkdir(parents=False)
        for name in ('logs', 'runtime', 'reports', 'initialization'):
            (output/name).mkdir()
        from scope_initialization import prepare_scope_initialization
        init = prepare_scope_initialization(config['base_initialization'], output/'initialization/framecrafter_image36.safetensors')
        write_json(output/'portable_request.json', {'time': now(), 'request': wanted, 'initialization_sha256': sha256(init)})
        write_json(output/'runtime/train_config.json', config)
    if (fresh and config['preflight']) or args.preflight_only:
        preflight(config, output, run_id)
    if args.preflight_only:
        print('Preflight passed; no training started.')
        return
    import yaml
    accelerate = yaml.safe_load((SNAPSHOT/'zero2.yaml').read_text())
    accelerate['deepspeed_config']['deepspeed_config_file'] = str(SNAPSHOT/'deepspeed_zero2.json')
    assert accelerate['num_processes'] == WORLD_SIZE
    (output/'runtime/zero2.yaml').write_text(yaml.safe_dump(accelerate))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(output/'runtime/zero2.yaml'),
        '--num_processes', str(WORLD_SIZE), '--mixed_precision', 'bf16', '--main_process_port', str(port),
        str(ROOT/'tools/entry.py'), '--config', str(output/'runtime/train_config.json'), '--script', 'train', '--',
        '--architecture', 'scope', '--wan', config['wan_root'],
        '--base-initialization', str(output/'initialization/framecrafter_image36.safetensors'),
        '--initial-step', str(step)]
    if restore:
        command += ['--restore-checkpoint', str(restore)]
    diagnostics = output/'runtime/diagnostics'/run_id
    diagnostics.mkdir(parents=True)
    triton = config['triton_cache'] or f"{os.environ.get('TMPDIR', '/tmp')}/{os.environ.get('USER', 'user')}/triton_neardepth"
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(gpu_tokens(WORLD_SIZE)), 'PYTHONPATH': str(REPO),
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'TRITON_CACHE_DIR': triton,
        'TORCH_NCCL_TRACE_BUFFER_SIZE': '4096', 'TORCH_NCCL_DUMP_ON_TIMEOUT': '1',
        'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': str(diagnostics/'nccl_trace_rank_'), 'NCCL_DEBUG': 'WARN'}
    write_json(output/f'runtime/command_{run_id}.json', {'command': command, 'initial_step': step,
        'restore_checkpoint': str(restore) if restore else None, 'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES']})
    print(f'{now()} launching from step {step}: {shlex.join(command)}', flush=True)
    result = subprocess.call(command, cwd=SNAPSHOT, env=env)
    if result == 0:
        assert (output/f"step-{config['final_step']}.safetensors").is_file()
        assert json.loads((output/'_SUCCESS').read_text())['step'] == config['final_step']
        print(f'{now()} complete at step {config["final_step"]}', flush=True)
    raise SystemExit(result)


if __name__ == '__main__':
    main()
