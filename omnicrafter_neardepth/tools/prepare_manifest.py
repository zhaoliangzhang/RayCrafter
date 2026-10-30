"""Portable snapshot/prepare_cached_manifest.py: the io100 near-depth manifest at a new cache root.

Reads the handoff's exact cached manifest (omnicrafter_handoff/metadata/train_cached.jsonl.gz,
200k sets), moves the eight RGB paths per set from io104's cache prefix to --cache-root and
attaches each scene's near depth from near_depth/*.json, row by row as the original script.
With io100's cache root the output is byte-identical to io100's manifest; with any root its
path-independent digest must equal io100's, which this script checks before marking success.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'snapshot'))
import numpy as np
from common import *

HANDOFF = ROOT.parent/'omnicrafter_handoff'
OLD_PREFIX = '/export/io104a/data/xliu253/omnicrafter_mixed_local_cache/'
NEAR_DEPTH = {'perspective': ROOT/'near_depth/perspective.json', 'panorama': ROOT/'near_depth/panorama.json'}
SIDE_FILES = ('heldout_scenes.json', 'hm3d_scene_selection.json', 'jhu_scene_selection.json', 'perspective_source_audit.json')
# io100's data_manifest (mixed_scope_neardepth_q3_..._100k_io100), image paths made relative to its cache root.
IO100_CANONICAL_SHA256 = '07275412712f04e95f05a482bebf478fca4de0c146586a5398dfe5e3f9c206f0'
IO100_SELECTED_SCENES_SHA256 = '75579e8f4f08c92e1e6f8932d9be3857ae539534b365cb7f19c7dcc8aefd3e68'
IO100_NEAR_DEPTH_SHA256 = {'perspective': 'e548a777a7b9b1c3ad75a50193d3c87e6a53fa367799be8e11a7c3908d9c964c',
                           'panorama': '278161cb30842988c0a40486269f314f87f6c161261d469173294ff43e883f64'}


def image_views(row):
    return ((row['references'], ('image',)), (row['targets'], ('image', 'noisy')))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-root', type=Path, required=True, help='Copy of omnicrafter_mixed_local_cache on this cluster')
    p.add_argument('--output', type=Path, required=True, help='New data directory (the training config data_dir)')
    p.add_argument('--source', type=Path, default=HANDOFF/'metadata/train_cached.jsonl.gz')
    p.add_argument('--check-files', action='store_true', help='Also require every RGB file to exist (1.6M stats)')
    args = p.parse_args()
    cache, data = args.cache_root.resolve(strict=True), args.output
    source_summary = json.loads((HANDOFF/'metadata/train_cache/summary.json').read_text())
    depth_hashes = {name: sha256(path) for name, path in NEAR_DEPTH.items()}
    assert depth_hashes == IO100_NEAR_DEPTH_SHA256, 'near_depth/*.json differ from the io100 run'
    if (data/'_SUCCESS').exists():
        existing = json.loads((data/'summary.json').read_text())
        assert existing['parent_cached_manifest_sha256'] == source_summary['sha256']
        assert sha256(data/'train.jsonl') == existing['sha256']
        assert sha256(data/'train.offsets.npy') == existing['offsets_sha256']
        assert existing['canonical_sha256'] == IO100_CANONICAL_SHA256
        print('Existing manifest verified', flush=True)
        return
    with gzip.open(args.source, 'rb') as src:
        digest = hashlib.sha256()
        for block in iter(lambda: src.read(8 * 1024**2), b''):
            digest.update(block)
    assert digest.hexdigest() == source_summary['sha256'], 'train_cached.jsonl.gz is not the handoff manifest'
    depth_tables = {name: json.loads(path.read_text()) for name, path in NEAR_DEPTH.items()}
    data.mkdir(parents=True, exist_ok=True)
    offsets, counts, changed, missing = [], Counter(), 0, 0
    scene_depths, canonical = {}, hashlib.sha256()
    temporary = data/f'.train.{os.getpid()}.jsonl.tmp'
    with gzip.open(args.source, 'rt') as src, temporary.open('xb') as dst:
        for line in src:
            row = json.loads(line)
            assert row['split'] == 'train'
            assert len(row['references']) == 2 and len(row['targets']) == 3
            scene = row['scene_name']
            key = scene.removeprefix('DL3DV::') if row['projection'] == 'perspective' else scene
            entry = depth_tables[row['projection']][key]
            depth = float(entry['near_depth'])
            assert math.isfinite(depth) and depth > 0, (scene, entry)
            if scene in scene_depths:
                assert scene_depths[scene]['near_depth'] == depth
            scene_depths[scene] = {'near_depth': depth, 'source_key': key,
                'projection': row['projection'], 'dataset': row['dataset'],
                'source_sha256': depth_hashes[row['projection']], 'source_record': entry}
            row.update(near_depth=depth, near_depth_source_key=key, pose_normalization=POSE_NORMALIZATION)
            for views, keys in image_views(row):
                for view in views:
                    for name in keys:
                        assert view[name].startswith(OLD_PREFIX), view[name]
                        view[name] = view[name].removeprefix(OLD_PREFIX)
                        changed += 1
            canonical.update((json.dumps(row, separators=(',', ':')) + '\n').encode())
            for views, keys in image_views(row):
                for view in views:
                    for name in keys:
                        view[name] = str(cache/view[name])
                        if args.check_files and not Path(view[name]).is_file():
                            missing += 1
            offsets.append(dst.tell())
            dst.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
            counts[row['dataset']] += 1
    assert counts == COUNTS and changed == 1600000
    assert missing == 0, f'{missing} RGB files missing under {cache}'
    assert canonical.hexdigest() == IO100_CANONICAL_SHA256, 'Manifest differs from io100 beyond the cache root'
    temporary.replace(data/'train.jsonl')
    temporary_offsets = data/f'.offsets.{os.getpid()}.tmp'
    with temporary_offsets.open('xb') as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64))
    temporary_offsets.replace(data/'train.offsets.npy')
    for name in SIDE_FILES:
        shutil.copy2(HANDOFF/'metadata/train_cache'/name, data/name)
    (data/'near_depth').mkdir(exist_ok=True)
    for name, path in NEAR_DEPTH.items():
        shutil.copy2(path, data/'near_depth'/f'{name}.json')
    write_json(data/'near_depth/selected_training_scenes.json', scene_depths)
    assert sha256(data/'near_depth/selected_training_scenes.json') == IO100_SELECTED_SCENES_SHA256
    summary = dict(source_summary)
    summary.update(time=now(), parent_cached_manifest_sha256=source_summary['sha256'],
        sha256=sha256(data/'train.jsonl'), offsets_sha256=sha256(data/'train.offsets.npy'),
        canonical_sha256=canonical.hexdigest(), image_prefix=str(cache),
        cache_ready_marker=str(cache/'_READY.json'), rgb_files_checked=args.check_files,
        relocation='Eight RGB paths relocated; near-depth scalar/provenance attached. IDs, poses, splits, scene quotas and order unchanged',
        pose_normalization=POSE_NORMALIZATION, near_depth_source_sha256=depth_hashes,
        near_depth_source_paths={k: str(v) for k, v in NEAR_DEPTH.items()},
        near_depth_scene_counts=dict(Counter(v['dataset'] for v in scene_depths.values())),
        near_depth_coverage_sets=sum(counts.values()), near_depth_missing_scenes=0)
    write_json(data/'summary.json', summary)
    write_json(data/'_SUCCESS', {'time': now(), 'logical_sets': len(offsets), 'dataset_counts': dict(counts),
        'changed_image_paths': changed, 'source_manifest_sha256': source_summary['sha256']})
    print(json.dumps({'manifest': str(data/'train.jsonl'), 'sets': len(offsets), 'counts': dict(counts),
                      'canonical_sha256': canonical.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
