"""xliu253's mixed-projection concat run from Wan's weights instead of the published FrameCrafter adapter.

The trainer is ``omnicrafter_handoff/snapshots/mixed/train.py``, unmodified: the code behind xliu253's
``mixed_projection_2ref3_dl3dv100k_hm3d60k_jhu40k_b1_300k_io104`` checkpoints, with the same 200k-set
manifest, seed, per-set NVS, 0-100% dropout, ConstantLR and DeepSpeed config. This mirrors the handoff's
``tools/train_mixed.py`` (its checks, output layout and accelerate launch) except that
  - ``base_initialization`` may be any 802-tensor concat file, not only the published adapter's hash;
  - ``entry.py`` wraps ``tools/train_entry.py`` so collectives wait 2 h, not 10 min.

    python launch.py --config <json> [--dry-run]    # foreground; detach it on the GPU node
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'omnicrafter_handoff/tools'))
import train_mixed  # noqa: E402
from handoff_common import (ROOT, camera_architecture, checkpoint_header, gpu_tokens, protocol_hashes,  # noqa: E402
                            read_json, sha256, write_json)

COLLECTIVE_TIMEOUT_MINUTES = 120


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = read_json(args.config.resolve())
    if camera_architecture(config) != 'concat':
        raise ValueError('This launcher is for the concat run')
    base = Path(config['base_initialization'])
    checkpoint_header(base, 'concat', initialization=True)
    # The one check relaxed: train_mixed accepts only the published FrameCrafter adapter's hash.
    train_mixed.PUBLISHED_SHA256 = sha256(base)
    train_mixed.validate(config)
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError('Choose a NEW output directory; continuation restores weights into a new run')
    effective_path = output/'runtime/train_config.json'
    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
           str(ROOT/'snapshots/mixed/zero2.yaml'), '--num_processes', '4',
           str(HERE/'entry.py'), '--config', str(effective_path)]
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        print(f'Preflight passed: initialization {base} ({sha256(base)[:12]}). No output, process or CUDA allocation.')
        return
    tokens = gpu_tokens(4)
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ('runtime', 'reports', 'logs'):
        (output/subdir).mkdir(parents=True, exist_ok=True)
    effective = {**config, 'architecture': 'concat', 'prepared_initialization': str(base)}
    write_json(effective_path, effective)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    write_json(output/'portable_request.json', {'config': config, 'effective_config': effective,
        'prepared_initialization_sha256': sha256(base), 'code_sha256': protocol_hashes(),
        'launcher': {'commit': commit, 'launch.py': sha256(__file__), 'entry.py': sha256(HERE/'entry.py')},
        'collective_timeout_minutes': COLLECTIVE_TIMEOUT_MINUTES})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(tokens), 'PYTHONUNBUFFERED': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
           'DEEPSPEED_TIMEOUT': str(COLLECTIVE_TIMEOUT_MINUTES)}
    print(f'commit {commit}  CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}', flush=True)
    raise SystemExit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
