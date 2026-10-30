"""Data contract for the mixed perspective/ERP 2ref3 experiment."""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime

CODE = Path(__file__).resolve().parent
OUT = Path('/cis/net/io81/data/xliu253/omniraycrafter/mixed_scope_neardepth_q3_dl3dv100k_hm3d60k_jhu40k_b1_100k_io100')
DATA = OUT / 'data_manifest'
PANO_ROOT = Path('/cis/net/io80/data/xliu253/framecrafter/5input_model')
PERSPECTIVE_ROOT = Path('/cis/home/ayadav/wriva_runtimes_all/for_Zhaoliang/dl3dv_points/train')
PERSPECTIVE_MANIFEST = PERSPECTIVE_ROOT / 'benchmark/manifest.json'
PANO_SOURCES = {
    'HM3D': PANO_ROOT / 'manifests/hm3d_train.jsonl',
    'JHU': PANO_ROOT / 'manifests/jhu_holdout10_seed20260822/jhu_train_2ref1_excluding_10_scenes.jsonl',
}
HELDOUT = DATA / 'heldout_scenes.json'
POSES = PANO_ROOT / 'poses/manifest.json'
COUNTS = {'DL3DV': 100000, 'HM3D': 60000, 'JHU': 40000}
SEED = 20261023
PYTHON = Path('/cis/home/xliu253/micromamba/envs/framecrafter/bin/python')


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def stable_rank(value):
    return int.from_bytes(hashlib.sha256(f'{SEED}:{value}'.encode()).digest()[:8], 'big')


def balanced_quotas(capacities, total):
    """Water-fill scene quotas, retaining every nonempty scene, without replacement."""
    if not capacities or any(n < 1 for n in capacities.values()):
        raise ValueError('Need positive capacities')
    if not len(capacities) <= total <= sum(capacities.values()):
        raise ValueError('Requested count cannot cover all scenes without replacement')
    quotas = dict.fromkeys(capacities, 0)
    order = sorted(capacities, key=stable_rank)
    remaining = total
    while remaining:
        active = [s for s in order if quotas[s] < capacities[s]]
        for scene in active:
            quotas[scene] += 1
            remaining -= 1
            if not remaining:
                break
    return quotas
