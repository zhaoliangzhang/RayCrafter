"""Full matched 2ref3 evaluation. Foreground, resumable; never SSHs or queues."""
import argparse
from collections import Counter
import fcntl
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from handoff_common import *
from eval_entry import bind


def validate(config):
    if camera_architecture(config) != 'concat':
        raise ValueError('The portable evaluation wrapper currently supports concat only. '
                         'SCoPE training is supported by train_mixed.py; SCoPE inference is not wired here.')
    rows = read_json(required_file(config['manifest']))['sets']
    if Counter(r['dataset'] for r in rows) != EVAL_COUNTS:
        raise ValueError('Use the full frozen HM3D423/Replica68/JHU500/DL3DV2800 benchmark')
    for i, row in enumerate(rows):
        validate_row(row)
        if row['index'] != i or row['seed'] != 20260904+i:
            raise ValueError('Evaluation order/seeds changed')
    if not 1 <= config['num_gpus'] <= 32:
        raise ValueError('Invalid number of GPUs')
    checkpoint_header(required_file(config['checkpoint']))
    required_file(config['fid_weights'])
    if sha256(config['fid_weights']) != 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4':
        raise ValueError('FID weights differ from the source evaluation')
    required_file(config['metric_python'])
    if not Path(config['dreamsim_cache']).is_dir():
        raise FileNotFoundError(config['dreamsim_cache'])
    model = Path(config['wan_root'])/'Wan-AI/Wan2.1-I2V-14B-480P'
    for i in range(1, 8):
        required_file(model/f'diffusion_pytorch_model-{i:05d}-of-00007.safetensors')
    for name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'):
        required_file(model/name)
    required_file(Path(config['wan_root'])/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json')
    return rows


def prepare(config):
    shared = bind(config)
    out = shared.OUT
    fingerprint = {'config': config, 'manifest_sha256': sha256(config['manifest']),
                   'checkpoint_sha256': sha256(config['checkpoint']), 'code_sha256': protocol_hashes()}
    request = out/'portable_request.json'
    if request.exists() and read_json(request) != fingerprint:
        raise ValueError('Existing output belongs to another config/code/manifest/checkpoint')
    write_json(request, fingerprint)
    if (out/'benchmark/_SUCCESS').is_file():
        contract = read_json(out/'provenance.json')
        if sha256(out/'benchmark/manifest.json') != contract['manifest_sha256']:
            raise ValueError('Prepared benchmark changed')
        return shared
    from PIL import Image
    data = read_json(config['manifest'])
    for directory in ('runtime', 'logs', 'metrics', 'benchmark'):
        (out/directory).mkdir(parents=True, exist_ok=True)
    shared.CACHE.mkdir(parents=True, exist_ok=True)
    for src, dest in [(shared.CHECKPOINT, shared.CACHE/shared.CHECKPOINT.name),
                      (shared.FID_SOURCE, shared.CACHE/'fid/inception-2015-12-05.pt')]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_name('.'+dest.name+'.copying')
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
        if sha256(src) != sha256(dest):
            raise ValueError(f'Local cache contains another weight file: {dest}')
    for i, row in enumerate(data['sets']):
        for obj, key in image_fields(row):
            required_file(obj[key])
        size = (row['width'], row['height'])
        for target in row['targets']:
            # Reuse the EXACT resized GT from the prior evaluation, no new resize.
            with Image.open(target['metric_gt']) as image:
                if image.mode != 'RGB' or image.size != size:
                    raise ValueError('Metric GT changed mode/resolution')
        for task in shared.TASKS:
            directory = shared.bundle(row, task)
            (directory/'predictions').mkdir(parents=True, exist_ok=True)
            for j, view in enumerate(row['references']):
                shared.link(view['image'], directory/'references'/f'ref_{j}.png')
            for j, view in enumerate(row['targets']):
                shared.link(view['metric_gt'], directory/'ground_truth'/f'query_{j}.png')
                shared.link(view['image'], directory/'ground_truth_original'/f'query_{j}.png')
                if task == 'cleanup':
                    shared.link(view['noisy'], directory/'noisy_inputs'/f'query_{j}.png')
            if task == 'cleanup':
                shared.link(row['prior_metadata'], directory/'noisy_inputs/provenance.json')
            shared.write_json(directory/'input_metadata.json', {**row, 'task': task,
                'online_pixel_dropout': False, 'GT_access_during_inference': False}, immutable=True)
        if (i+1) % 100 == 0:
            print(f'Prepared {i+1}/{len(data["sets"])} sets', flush=True)
    shared.write_json(out/'benchmark/manifest.json', data, immutable=True)
    contract = {**fingerprint, 'manifest_sha256': sha256(out/'benchmark/manifest.json'),
        'GT_resize': 'Reuse byte-identical PIL-bilinear resized metric GT from original benchmark',
        'fid_note': 'Clean-FID clean 2048-D; small correlated test sets (especially Replica204) limit reliability.',
        'strict_HM3D': 'Exclude the same 21 reference/target-overlap targets; retain inference groups.'}
    shared.write_json(out/'provenance.json', contract, immutable=True)
    write_json(out/'benchmark/_SUCCESS', {'sets': len(data['sets']), 'passed': True})
    return shared


def run_phase(config, config_path, phase):
    tokens = gpu_tokens(config['num_gpus'])
    out = Path(config['output'])
    children = []
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    count = 1 if phase in ('aggregate', 'smoke') else len(tokens)
    try:
        for rank in range(count):
            python = config['metric_python'] if phase not in ('worker', 'smoke') else sys.executable
            command = [python, '-B', '-u', str(ROOT/'tools/eval_entry.py'), '--config', str(config_path),
                       '--phase', 'worker' if phase == 'smoke' else phase, '--rank', str(rank)]
            if phase == 'smoke':
                command += ['--smoke']
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if phase == 'aggregate' else tokens[rank],
                   'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                   'TORCH_HOME': config['torch_home']}
            path = out/f'logs/{phase}_rank{rank}_{stamp}.log'
            with path.open('x') as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            children.append((proc, path))
        while True:
            failed = [(p.returncode, str(log)) for p, log in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'{phase} failed: {failed}; completed outputs preserved')
            running = [p.pid for p, _ in children if p.poll() is None]
            write_json(out/'runtime/portable_status.json', {'phase': phase, 'running_pids': running,
                       'time': datetime.now().astimezone().isoformat()})
            if not running:
                break
            time.sleep(15)
    finally:
        # Stop only children of THIS invocation; never other users' jobs.
        for proc, _ in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'inference', 'metrics', 'aggregate', 'all'], default='all')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = read_json(args.config)
    rows = validate(config)
    print(f'{len(rows)} sets, {len(rows)*6} generated targets; phase={args.phase}', flush=True)
    if args.dry_run:
        print('Static config/weight/manifest checks passed; no output writes or GPU calls.')
        return
    if args.phase != 'prepare':
        gpu_tokens(config['num_gpus'])
    out = Path(config['output'])
    (out/'runtime').mkdir(parents=True, exist_ok=True)
    with (out/'runtime/portable_launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(config)
        phases = {'prepare': [], 'smoke': ['smoke'], 'inference': ['worker'], 'metrics': ['score'],
                  'aggregate': ['aggregate'], 'all': ['worker', 'score', 'aggregate']}[args.phase]
        for phase in phases:
            run_phase(config, args.config.resolve(), phase)
    print('Requested phases complete.', flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
