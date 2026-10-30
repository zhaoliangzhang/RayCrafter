"""Portable concat/SCoPE training: fixed four GPUs and batch one per GPU."""
import argparse
import shlex
import subprocess
import sys
from handoff_common import *


def validate(config):
    architecture = camera_architecture(config)
    if config.get('num_gpus', 4) != 4 or config.get('batch_per_gpu', 1) != 1:
        raise ValueError('This exact trainer supports four GPUs and batch one only')
    initial, final, every = (config[k] for k in ('initial_step', 'final_step', 'save_every'))
    if not 0 <= initial < final or every <= 0 or final % every:
        raise ValueError('Use 0 <= initial < final, positive save interval, and aligned final step')
    data = Path(config['data_dir'])
    for n in ('train.jsonl', 'train.offsets.npy', 'summary.json'):
        required_file(data/n)
    summary = read_json(data/'summary.json')
    if summary['logical_sets'] != 200000 or summary['dataset_counts'] != COUNTS:
        raise ValueError('Use the exact 100k DL3DV / 60k HM3D / 40k JHU manifest')
    if sha256(data/'train.jsonl') != summary['sha256']:
        raise ValueError('Training manifest checksum mismatch')
    if sha256(data/'train.offsets.npy') != summary['offsets_sha256']:
        raise ValueError('Rebuild offsets after rewriting any JSONL paths')
    base = required_file(config['base_initialization'])
    checkpoint_header(base)
    if sha256(base) != PUBLISHED_SHA256:
        raise ValueError('base_initialization must be the exact published FrameCrafter adapter')
    if config.get('restore_checkpoint'):
        meta = checkpoint_header(required_file(config['restore_checkpoint']), architecture)
        if int(meta.get('step', -1)) != initial:
            raise ValueError('initial_step must match restore checkpoint metadata')
    elif initial != 0:
        raise ValueError('A nonzero initial_step requires a restore checkpoint')
    model = Path(config['wan_root'])/'Wan-AI/Wan2.1-I2V-14B-480P'
    for i in range(1, 8):
        required_file(model/f'diffusion_pytorch_model-{i:05d}-of-00007.safetensors')
    for name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'):
        required_file(model/name)
    required_file(Path(config['wan_root'])/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json')


def command(config_path):
    return [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
            str(ROOT/'snapshots/mixed/zero2.yaml'), '--num_processes', '4',
            str(ROOT/'tools/train_entry.py'), '--config', str(config_path)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    path = args.config.resolve()
    config = read_json(path)
    validate(config)
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError('Choose a NEW output directory; continuation restores weights into a new run')
    architecture = camera_architecture(config)
    effective_path = output/'runtime/train_config.json'
    cmd = command(effective_path)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        print(f'Architecture: {architecture}; ' + ('will create a separate 36-channel initialization on CPU.'
              if architecture == 'scope' else 'published 420-channel initialization unchanged.'))
        print('Preflight passed. No output directory, process, or CUDA allocation created.')
        return
    tokens = gpu_tokens(4)
    print('CUDA_VISIBLE_DEVICES=' + ','.join(tokens), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ('runtime', 'reports', 'logs'):
        (output/subdir).mkdir(parents=True, exist_ok=True)
    base = Path(config['base_initialization'])
    if architecture == 'scope':
        from scope_initialization import prepare_scope_initialization
        base = prepare_scope_initialization(base, output/'initialization/framecrafter_image36.safetensors')
    effective = {**config, 'architecture': architecture, 'prepared_initialization': str(base)}
    write_json(effective_path, effective)
    write_json(output/'portable_request.json', {'config': config, 'effective_config': effective,
        'prepared_initialization_sha256': sha256(base), 'code_sha256': protocol_hashes()})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(tokens), 'PYTHONUNBUFFERED': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    # This is a foreground command. Use your scheduler or tmux for persistence.
    raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
