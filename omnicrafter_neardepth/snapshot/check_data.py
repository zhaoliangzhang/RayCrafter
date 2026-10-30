"""Preflight: full manifest audit, all camera calibrations, real-image smoke tests."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import random
import math
import cv2
import numpy as np
import torch
from common import *
from mixed_camera import perspective_directions
from mixed_dataset import MixedProjectionDataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',choices=['scope','concat'],required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    summary = json.loads((DATA/'summary.json').read_text())
    assert sha256(DATA/'train.jsonl') == summary['sha256']
    assert sha256(DATA/'train.offsets.npy') == summary['offsets_sha256']
    assert summary['pose_normalization'] == POSE_NORMALIZATION
    depth_tables = {k:json.loads((DATA/'near_depth'/f'{k}.json').read_text()) for k in ('perspective','panorama')}
    for projection in depth_tables:
        assert sha256(DATA/'near_depth'/f'{projection}.json') == summary['near_depth_source_sha256'][projection]
    heldout = set(json.loads(HELDOUT.read_text()))
    counts,scenes,ids,cameras,examples = Counter(),{},set(),{},{}
    dataset = MixedProjectionDataset(args.architecture)
    assert len(dataset) == 200000
    for index in range(len(dataset)):
        row = dataset.record(index)
        assert row['id'] not in ids
        ids.add(row['id'])
        counts[row['dataset']] += 1
        scenes.setdefault(row['dataset'],Counter())[row['scene_name']] += 1
        assert row['projection'] == ('perspective' if row['dataset']=='DL3DV' else 'panorama')
        assert (row['width'],row['height']) == (512,288 if row['projection']=='perspective' else 256)
        assert row['dataset'] != 'JHU' or row['scene_name'].removeprefix('JHU::') not in heldout
        assert row['pose_normalization'] == POSE_NORMALIZATION
        assert math.isfinite(row['near_depth']) and row['near_depth'] > 0
        key = row['scene_name'].removeprefix('DL3DV::') if row['projection']=='perspective' else row['scene_name']
        assert row['near_depth_source_key'] == key
        assert row['near_depth'] == depth_tables[row['projection']][key]['near_depth']
        bucket = examples.setdefault(row['dataset'],[])
        if len(bucket)<4:
            bucket.append(index)
        for view in row['targets']:
            assert view['noisy'] and view['noisy'] != view['image']
        if row['projection']=='perspective':
            for view in row['references']+row['targets']:
                camera = view['camera']
                key = json.dumps({k:camera[k] for k in ('K','width','height','dist')},sort_keys=True)
                cameras.setdefault(key,camera)
    assert counts == COUNTS
    for name in ('HM3D','JHU'):
        selection = json.loads((DATA/f'{name.lower()}_scene_selection.json').read_text())
        assert dict(scenes[name]) == selection['selected']
        assert set(scenes[name]) == set(selection['available'])
    def check_camera(camera):
        rays = perspective_directions(camera,288,512,16 if args.architecture=='scope' else 1)
        assert torch.isfinite(rays).all()
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(check_camera,cameras.values()))
    records = []
    for name,indices in examples.items():
        for index in indices:
            for probability in (0.,1.):
                dataset.nvs_probability = probability
                random.seed(SEED+index)
                sample = dataset[index]
                expected = (512,288 if name=='DL3DV' else 256)
                assert all(im.size==expected for im in sample['input_images']+sample['conditioning_images']+sample['target_images'])
                if probability == 1:
                    assert sample['sample_task_type']=='nvs'
                    assert all(not np.asarray(im).any() for im in sample['conditioning_images'])
                else:
                    assert sample['sample_task_type']=='cleanup'
                    assert all(0 <= ratio <= 1 for ratio in sample['query_pixel_dropout_ratios'])
                feature = sample['scope_rays'] if args.architecture=='scope' else sample['raymap']
                assert torch.isfinite(feature).all()
                records.append({'sample':sample['sample_id'],'projection':sample['projection'],
                    'task':sample['sample_task_type'],'resolution_wh':expected,'camera_shape':list(feature.shape)})
    assert not torch.cuda.is_initialized()
    write_json(OUT/'reports/data_preflight.json',{'time':now(),'passed':True,'cpu_only':True,
        'dataset_counts':dict(counts),'scene_counts':{n:len(s) for n,s in scenes.items()},
        'camera_calibrations_checked':len(cameras),'examples':records,'architecture':args.architecture,
        'pose_normalization':POSE_NORMALIZATION,'near_depth_verified_sets':sum(counts.values()),
        'near_depth_source_sha256':summary['near_depth_source_sha256']})
    print(f'PASS: 200000 entries, all scene quotas, {len(cameras)} calibrations, {len(records)} decoded examples',flush=True)


if __name__ == '__main__':
    main()
