"""Build a no-copy, audited manifest; never modify source datasets or splits."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import heapq
import json
import os
from pathlib import Path
import random
import stat
import time

import numpy as np
from PIL import Image
from settings import *


def progress(phase, **values):
    report = {'time': now(), 'phase': phase, **values}
    write_json(DATA / 'preparation_status.json', report)
    print(json.dumps(report), flush=True)


@lru_cache(maxsize=1000000)
def image_size(path):
    with Image.open(path) as image:
        if image.format != 'PNG':
            raise ValueError(f'Not PNG: {path}')
        return image.size


@lru_cache(maxsize=1000000)
def check_file(path):
    info = os.stat(path)
    assert stat.S_ISREG(info.st_mode) and info.st_size > 0, path
    assert os.access(path,os.R_OK), path


PERSPECTIVE_METADATA = {}
FULL_CHECK_IDS = set()


def validate_poses(views):
    poses = np.array([v['w2c'] for v in views], dtype=np.float64)
    if poses.shape != (5, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Expected five finite 4x4 poses')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous poses')
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3):
        raise ValueError('Non-orthonormal camera rotation')
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError('Camera reflection or singular rotation')


def read_pano_groups(dataset):
    """Recombine the three original JHU target rows; do not sample single views."""
    previous, rows = None, []
    with PANO_SOURCES[dataset].open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get('original_source_sample', row['source_sample'])
            number = row.get('original_target_number', row['target_number'])
            if identity != previous:
                if rows:
                    assert len(rows) == 3
                    yield previous, rows
                previous, rows = identity, []
            assert number == len(rows) and row['dataset'] == dataset
            if rows:
                assert row['scene_name'] == rows[0]['scene_name'] and row['refs'] == rows[0]['refs']
            rows.append(row)
    if rows:
        assert len(rows) == 3
        yield previous, rows


def select_panorama(dataset):
    capacities = Counter()
    seen = set()
    heldout = set(json.loads(HELDOUT.read_text()))
    for identity, rows in read_pano_groups(dataset):
        assert identity not in seen
        seen.add(identity)
        scene = rows[0]['scene_name']
        assert dataset != 'JHU' or scene.removeprefix('JHU::') not in heldout
        capacities[scene] += 1
    quotas = balanced_quotas(capacities, COUNTS[dataset])
    heaps = defaultdict(list)
    for identity, rows in read_pano_groups(dataset):
        scene = rows[0]['scene_name']
        entry = (-stable_rank(identity), identity, rows)
        heap = heaps[scene]
        if len(heap) < quotas[scene]:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    assert set(heaps) == set(capacities)
    assert all(len(heaps[s]) == quotas[s] for s in capacities)
    write_json(DATA / f'{dataset.lower()}_scene_selection.json', {
        'dataset': dataset, 'scene_unit': 'original scene/pose-sequence folder',
        'source_sets': sum(capacities.values()), 'selected_sets': COUNTS[dataset],
        'all_training_scenes_covered': True, 'scene_count': len(capacities),
        'minimum_per_scene': min(quotas.values()), 'maximum_per_scene': max(quotas.values()),
        'available': dict(capacities), 'selected': quotas,
        'selection': 'deterministic hash rank, scene-balanced water filling, no replacement',
    })
    for heap in heaps.values():
        FULL_CHECK_IDS.add(min(entry[1] for entry in heap))
    return [(identity, rows) for scene in sorted(heaps) for _, identity, rows in heaps[scene]]


POSE_MANIFEST = None


@lru_cache(maxsize=5000)
def scene_poses(scene):
    record = POSE_MANIFEST['scenes'][scene]
    with np.load(PANO_ROOT / record['pose_path']) as archive:
        poses = np.asarray(archive['w2c_poses'], dtype=np.float32)
    assert poses.shape == (record['num_frames'], 4, 4)
    return poses


def make_panorama(selected):
    identity, rows = selected
    dataset, scene = rows[0]['dataset'], rows[0]['scene_name']
    poses = scene_poses(scene)
    refs, targets = [], []
    full_check = identity in FULL_CHECK_IDS or stable_rank(identity)%100 == 0
    for path in rows[0]['refs']:
        index = int(Path(path).stem)
        absolute = str(PANO_ROOT / path)
        check_file(absolute)
        if full_check:
            assert image_size(absolute) == (512, 256)
        refs.append({'image': absolute, 'frame_index': index, 'w2c': poses[index].tolist()})
    for number, row in enumerate(rows):
        gt, noisy = str(PANO_ROOT / row['target']), str(PANO_ROOT / row['src'])
        check_file(gt)
        check_file(noisy)
        if full_check:
            assert image_size(gt) == image_size(noisy) == (512, 256)
        index = row['frame_index']
        assert index == int(Path(row['target']).stem)
        targets.append({'image': gt, 'noisy': noisy, 'target_number': number,
                        'frame_index': index, 'w2c': poses[index].tolist()})
    validate_poses(refs + targets)
    return {'id': identity, 'source_id': identity, 'dataset': dataset, 'scene_name': scene,
        'projection': 'panorama', 'camera_model': 'equirectangular', 'split': 'train',
        'height': 256, 'width': 512, 'references': refs, 'targets': targets,
        'pose_convention': 'OpenCV world-to-camera', 'repeat_index': 0,
        'source_manifest': str(PANO_SOURCES[dataset])}


def make_perspective(item):
    directory = PERSPECTIVE_ROOT / 'priors/samples' / item['id']
    assert (directory / '_SUCCESS').is_file(), f'Incomplete perspective set: {item["id"]}'
    metadata = PERSPECTIVE_METADATA[item['id']]
    full_check = item['id'] in FULL_CHECK_IDS or stable_rank(item['id'])%100 == 0
    check_file(str(directory/'metadata.json'))
    if full_check:
        assert json.loads((directory/'metadata.json').read_text()) == metadata
    assert metadata['set_id'] == item['id'] and metadata['scene_name'] == item['scene_name']
    assert metadata['split'] == item['split'] == 'train'
    assert metadata['distortion_applied'] is True
    assert metadata['target_rgb_used_for_reconstruction'] is False
    assert len(item['references']) == 2 and len(item['targets']) == 3
    refs, targets = [], []
    for role, records, source, destination in (
        ('ref', metadata['references'], item['references'], refs),
        ('gt', metadata['targets'], item['targets'], targets),
    ):
        for number, (actual, expected) in enumerate(zip(records, source, strict=True)):
            for key in ('w2c', 'frame_index', 'camera', 'image'):
                assert actual[key] == expected[key], (item['id'], key)
            camera = actual['camera']
            size = (int(camera['width']), int(camera['height']))
            K = np.asarray(camera['K'])
            assert K.shape == (3, 3) and np.isfinite(K).all() and K[0,0] > 0 and K[1,1] > 0
            assert len(camera['dist']) == 4 and np.isfinite(camera['dist']).all()
            path = str(directory / f'{role}_{number}.png')
            check_file(path)
            if full_check:
                assert image_size(path) == size, path
            view = {'image': path, 'frame_index': actual['frame_index'],
                    'w2c': actual['w2c'], 'camera': camera}
            if role == 'gt':
                noisy = str(directory / f'prior_{number}.png')
                check_file(noisy)
                if full_check:
                    assert image_size(noisy) == size, noisy
                assert actual['prior'] == noisy and actual['gt'] == path
                view.update(noisy=noisy, target_number=number)
            destination.append(view)
    validate_poses(refs + targets)
    return {'id': 'DL3DV::' + item['id'], 'source_id': item['id'], 'dataset': 'DL3DV',
        'scene_name': 'DL3DV::' + item['scene_name'], 'projection': 'perspective',
        'camera_model': 'opencv_radial_tangential', 'split': 'train', 'height': 288, 'width': 512,
        'references': refs, 'targets': targets, 'pose_convention': 'OpenCV world-to-camera',
        'repeat_index': 0, 'source_manifest': str(PERSPECTIVE_MANIFEST),
        'prior_metadata': str(directory / 'metadata.json')}


def checked_map(function, items, workers, phase):
    result = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(items), 1000):
            result.extend(pool.map(function, items[begin:begin+1000]))
            progress(phase, completed=len(result), total=len(items))
    return result


def main():
    global POSE_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / 'train.jsonl').exists():
        raise FileExistsError('Refusing to overwrite an existing mixed training manifest')
    progress('reading_source_manifests')
    source_hashes = {str(p): sha256(p) for p in [PERSPECTIVE_MANIFEST, POSES, HELDOUT, *PANO_SOURCES.values()]}
    POSE_MANIFEST = json.loads(POSES.read_text())
    assert POSE_MANIFEST['pose_convention'] == 'w2c_opencv'
    perspective = json.loads(PERSPECTIVE_MANIFEST.read_text())
    assert perspective['pose_convention'] == 'OpenCV world-to-camera'
    assert perspective['projection'] == 'perspective'
    assert perspective['logical_sets'] == len(perspective['sets'])
    contract_path = PERSPECTIVE_ROOT / 'priors/run_contract.json'
    contract = json.loads(contract_path.read_text())
    assert contract['manifest_sha256'] == source_hashes[str(PERSPECTIVE_MANIFEST)]
    assert contract['ignore_distortion'] is False and contract['save_images'] is True
    skipped = None
    for shard in range(8):
        assert (PERSPECTIVE_ROOT / f'priors/_SUCCESS.shard{shard:02d}').is_file()
        configuration = PERSPECTIVE_ROOT / f'priors/configuration_shard_{shard:02d}.json'
        info = json.loads(configuration.read_text())
        current = set(info['skipped_degenerate_sets'])
        assert skipped is None or current == skipped
        skipped = current
        source_hashes[str(configuration)] = sha256(configuration)
    source_hashes[str(contract_path)] = sha256(contract_path)
    inputs = [item for item in perspective['sets'] if item['id'] not in skipped]
    # The renderer appends the exact metadata.json record to its progress logs.
    # Audit these sequentially instead of opening 97k separate metadata files;
    # inspect every selected image's stat/access, and PNG headers for every
    # scene plus a deterministic 1% sample. Completed exports remain untouched.
    for shard in range(8):
        log = PERSPECTIVE_ROOT/f'priors/progress_shard_{shard:02d}.jsonl'
        with log.open() as handle:
            for line in handle:
                record = json.loads(line)
                PERSPECTIVE_METADATA[record['set_id']] = record
        source_hashes[str(log)] = sha256(log)
    assert {item['id'] for item in inputs} == set(PERSPECTIVE_METADATA)
    scene_first = {}
    for item in inputs:
        scene_first.setdefault(item['scene_name'],item['id'])
    FULL_CHECK_IDS.update(scene_first.values())
    records = checked_map(make_perspective, inputs, args.workers, 'audit_perspective_images_and_cameras')
    perspective_unique = len(records)
    assert perspective_unique <= COUNTS['DL3DV']
    needed = COUNTS['DL3DV'] - perspective_unique
    # Keep every usable perspective set. Transparently fill the small deficit
    # with scene-balanced repeated entries; image data is never copied.
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record['scene_name']].append(record)
    if needed:
        quotas = balanced_quotas({s: len(v) for s,v in by_scene.items()}, needed)
        repeated = []
        for scene in sorted(by_scene):
            candidates = sorted(by_scene[scene], key=lambda r: stable_rank('repeat:'+r['id']))
            for record in candidates[:quotas[scene]]:
                repeated.append({**record, 'id': record['id']+'::repeat_1', 'repeat_index': 1})
        assert len(repeated) == needed
        records.extend(repeated)
    write_json(DATA / 'perspective_source_audit.json', {
        'manifest_sets': perspective['logical_sets'], 'usable_unique_sets': perspective_unique,
        'degenerate_sets_excluded': sorted(skipped), 'repeated_entries': needed,
        'unique_scenes': len(by_scene), 'selected_entries': COUNTS['DL3DV'],
        'repeat_policy': 'all usable sets once plus deterministic scene-balanced repetitions',
        'available_sets_per_scene': {s:len(v) for s,v in by_scene.items()},
        'distortion': 'preserve RGB/prior images; invert supplied OpenCV distortion when making rays',
        'image_header_audited_sets': sum(item['id'] in FULL_CHECK_IDS or stable_rank(item['id'])%100 == 0 for item in inputs),
    })
    del perspective, inputs, by_scene
    PERSPECTIVE_METADATA.clear()
    for dataset in ('HM3D', 'JHU'):
        progress('select_scene_balanced_panorama', dataset=dataset)
        selected = select_panorama(dataset)
        records.extend(checked_map(make_panorama, selected, args.workers, 'audit_'+dataset))
        del selected
    assert len(records) == sum(COUNTS.values()) == len({r['id'] for r in records})
    assert Counter(r['dataset'] for r in records) == COUNTS
    random.Random(SEED).shuffle(records)
    destination = DATA / 'train.jsonl'
    temporary = DATA / f'.train.{os.getpid()}.jsonl.tmp'
    offsets, digest = [], hashlib.sha256()
    with temporary.open('wb') as handle:
        for record in records:
            offsets.append(handle.tell())
            line = (json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n').encode()
            digest.update(line)
            handle.write(line)
    # Fail closed if a source manifest was replaced during preparation.
    assert all(sha256(path) == expected for path,expected in source_hashes.items())
    index_temp = DATA / f'.offsets.{os.getpid()}.tmp'
    with index_temp.open('wb') as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64))
    index_temp.replace(DATA / 'train.offsets.npy')
    temporary.replace(destination)
    summary = {'time': now(), 'logical_sets': len(records), 'query_images': len(records)*3,
        'dataset_counts': COUNTS, 'projection_counts': dict(Counter(r['projection'] for r in records)),
        'perspective_unique_sets': perspective_unique, 'perspective_repeated_entries': needed,
        'panorama_scene_balance': 'all training scene/sequence folders, without replacement',
        'resolutions_width_height': {'panorama': [512,256], 'perspective': [512,288]},
        'pose_convention': 'OpenCV world-to-camera', 'queries_per_set': 3, 'references_per_set': 2,
        'source_sha256': source_hashes, 'sha256': digest.hexdigest(),
        'offsets_sha256': sha256(DATA/'train.offsets.npy'), 'seed': SEED,
        'black_images_stored': False, 'copies_of_source_images': 0,
        'validation_used_for_training': False, 'jhu_heldout_scenes_excluded': True,
        'image_validation': 'all selected images are nonempty readable regular files; PNG headers/dimensions checked for every scene plus deterministic 1% sets; real-data smoke fully decodes examples',
    }
    write_json(DATA / 'summary.json', summary)
    write_json(DATA / '_SUCCESS', summary)
    progress('complete', **{k:summary[k] for k in ('logical_sets','dataset_counts','sha256')})


if __name__ == '__main__':
    main()
