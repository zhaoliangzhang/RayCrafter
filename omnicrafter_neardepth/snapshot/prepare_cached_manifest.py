"""Preserve exact training rows, relocate RGB paths, and attach scene near depth.

io104's /export paths are not local to io100. Point to the same verified copies
through /cis/net/io104a. Never resample or modify the original data.
"""
from collections import Counter
import shutil
import math
import numpy as np
from common import *

CACHE = Path('/cis/net/io104a/data/xliu253/omnicrafter_mixed_local_cache')
OLD_PREFIX = '/export/io104a/data/xliu253/omnicrafter_mixed_local_cache/'


def main():
    source_data = CACHE/'data_manifest'
    source = source_data/'train.jsonl'
    ready = json.loads((CACHE/'_READY.json').read_text())
    assert ready['checksum_verified'] and ready['logical_sets'] == 200000
    summary = json.loads((source_data/'summary.json').read_text())
    assert sha256(source) == summary['sha256']
    DATA.mkdir(parents=True, exist_ok=True)
    depth_hashes = {name:sha256(path) for name,path in NEAR_DEPTH_SOURCES.items()}
    depth_tables = {name:json.loads(path.read_text()) for name,path in NEAR_DEPTH_SOURCES.items()}
    if (DATA/'_SUCCESS').exists():
        existing = json.loads((DATA/'summary.json').read_text())
        assert existing['parent_cached_manifest_sha256'] == summary['sha256']
        assert sha256(DATA/'train.jsonl') == existing['sha256']
        assert sha256(DATA/'train.offsets.npy') == existing['offsets_sha256']
        assert existing['pose_normalization'] == POSE_NORMALIZATION
        assert existing['near_depth_source_sha256'] == depth_hashes
        print('Existing manifest verified', flush=True)
        return
    offsets, counts, changed = [], Counter(), 0
    scene_depths = {}
    destination = DATA/'train.jsonl'
    temporary = DATA/f'.train.{os.getpid()}.jsonl.tmp'
    with source.open() as src, temporary.open('xb') as dst:
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
            scene_depths[scene] = {'near_depth':depth,'source_key':key,
                'projection':row['projection'],'dataset':row['dataset'],
                'source_sha256':depth_hashes[row['projection']], 'source_record':entry}
            row.update(near_depth=depth,near_depth_source_key=key,pose_normalization=POSE_NORMALIZATION)
            for views, keys in ((row['references'], ('image',)), (row['targets'], ('image','noisy'))):
                for view in views:
                    for key in keys:
                        original = view[key]
                        assert original.startswith(OLD_PREFIX), original
                        view[key] = str(CACHE/original.removeprefix(OLD_PREFIX))
                        changed += 1
            offsets.append(dst.tell())
            dst.write((json.dumps(row, separators=(',',':'))+'\n').encode())
            counts[row['dataset']] += 1
    assert counts == COUNTS and changed == 1600000
    temporary.replace(destination)
    temporary_offsets = DATA/f'.offsets.{os.getpid()}.tmp'
    with temporary_offsets.open('xb') as handle:
        np.save(handle, np.asarray(offsets,dtype=np.int64))
    temporary_offsets.replace(DATA/'train.offsets.npy')
    for name in ('heldout_scenes.json','hm3d_scene_selection.json','jhu_scene_selection.json','perspective_source_audit.json'):
        shutil.copy2(source_data/name, DATA/name)
    (DATA/'near_depth').mkdir(exist_ok=True)
    for name,path in NEAR_DEPTH_SOURCES.items():
        destination_depth = DATA/'near_depth'/f'{name}.json'
        shutil.copy2(path,destination_depth)
        assert sha256(destination_depth) == depth_hashes[name]
    write_json(DATA/'near_depth/selected_training_scenes.json',scene_depths)
    summary.update(time=now(), parent_cached_manifest_sha256=summary['sha256'],
        sha256=sha256(destination),offsets_sha256=sha256(DATA/'train.offsets.npy'),
        image_prefix=str(CACHE),cache_ready_marker=str(CACHE/'_READY.json'),
        relocation='Eight RGB paths relocated; near-depth scalar/provenance attached. IDs, poses, splits, scene quotas and order unchanged',
        pose_normalization=POSE_NORMALIZATION,near_depth_source_sha256=depth_hashes,
        near_depth_source_paths={k:str(v) for k,v in NEAR_DEPTH_SOURCES.items()},
        near_depth_scene_counts=dict(Counter(v['dataset'] for v in scene_depths.values())),
        near_depth_coverage_sets=sum(counts.values()),near_depth_missing_scenes=0)
    write_json(DATA/'summary.json', summary)
    write_json(DATA/'_SUCCESS', {'time':now(),'logical_sets':len(offsets),'dataset_counts':dict(counts),
        'changed_image_paths':changed,'source_manifest_sha256':summary['parent_cached_manifest_sha256']})
    print(json.dumps({'manifest':str(destination),'sets':len(offsets),'counts':dict(counts)}),flush=True)


if __name__ == '__main__':
    main()
