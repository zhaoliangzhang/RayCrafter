"""Reuse the byte-identical 22k benchmark and verify every saved input hash."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import shutil
from eval_shared import *


def progress(phase, **values):
    record = {'time': now(), 'phase': phase, **values}
    write_json(OUT/'runtime/preparation.json', record)
    print(json.dumps(record), flush=True)


def main():
    from safetensors import safe_open
    assert (MATCHED/'_SUCCESS').is_file(), 'Original matched evaluation is incomplete'
    old = json.loads((MATCHED/'provenance.json').read_text())
    assert sha256(MATCHED/'benchmark/manifest.json') == old['manifest_sha256']
    # Preserve the exact camera/model implementation used by the original run.
    for relative, expected in old['code_sha256'].items():
        if relative.startswith('training_snapshot/'):
            assert sha256(CODE/relative) == expected, relative
    with safe_open(str(CHECKPOINT), framework='pt', device='cpu') as state:
        assert len(state.keys()) == 802
        assert state.get_slice('patch_embedding.weight').get_shape() == [5120,420,1,2,2]
        metadata = state.metadata()
        assert metadata['step'] == '50000'
        for field in ('camera_injection','manifest_sha256','model','projections','resolutions_wh'):
            assert metadata[field] == old['checkpoint_metadata'][field], field
    data = json.loads((MATCHED/'benchmark/manifest.json').read_text())
    rows = data['sets']
    assert Counter(row['dataset'] for row in rows) == COUNTS
    assert len(rows) == len({row['id'] for row in rows}) == 3791
    for i, row in enumerate(rows):
        assert row['index'] == i and row['seed'] == BASE_SEED + i
        assert len(row['references']) == 2 and len(row['targets']) == 3
        assert (row['width'],row['height']) == (512, 288 if row['projection']=='perspective' else 256)
    source_hashes = {p:h for p,h in old['source_hashes'].items() if p != old['checkpoint']}
    for path, expected in source_hashes.items():
        assert sha256(path) == expected, path
    source_hashes[str(CHECKPOINT)] = sha256(CHECKPOINT)
    expected_inputs = json.loads((MATCHED/'input_hashes.json').read_text())
    def check_input(item):
        path, expected = item
        assert sha256(path) == expected, f'An input changed since the 22k evaluation: {path}'
        return path
    progress('verifying_frozen_input_checksums', total=len(expected_inputs))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(check_input, expected_inputs.items()), 1):
            if i % 1000 == 0 or i == len(expected_inputs):
                progress('verifying_frozen_input_checksums', checked=i, total=len(expected_inputs))
    progress('creating_linked_bundles', sets=len(rows))
    for i, row in enumerate(rows, 1):
        for task in TASKS:
            dest = bundle(row, task)
            (dest/'predictions').mkdir(parents=True, exist_ok=True)
            for j, ref in enumerate(row['references']):
                link(ref['image'], dest/'references'/f'ref_{j}.png')
            for j, target in enumerate(row['targets']):
                link(target['metric_gt'], dest/'ground_truth'/f'query_{j}.png')
                link(target['image'], dest/'ground_truth_original'/f'query_{j}.png')
                if task == 'cleanup':
                    link(target['noisy'], dest/'noisy_inputs'/f'query_{j}.png')
            if task == 'cleanup':
                link(row['prior_metadata'], dest/'noisy_inputs/provenance.json')
            write_json(dest/'input_metadata.json', {**row, 'task':task,
                'input_query_rgb':'all_black' if task=='nvs' else 'cached_raw_point_render',
                'online_pixel_dropout':False, 'resize':'PIL bilinear to requested inference resolution',
                'GT_access_during_inference':False}, immutable=True)
        if i % 250 == 0 or i == len(rows):
            progress('creating_linked_bundles', completed=i, total=len(rows))
    (OUT/'benchmark').mkdir(parents=True, exist_ok=True)
    for name in ('benchmark/manifest.json', 'input_hashes.json'):
        dest = OUT/name
        if dest.exists():
            assert sha256(dest) == sha256(MATCHED/name)
        else:
            temp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
            shutil.copy2(MATCHED/name, temp)
            temp.replace(dest)
    link(MATCHED/'metric_ground_truth', OUT/'metric_ground_truth')
    assert sha256(OUT/'benchmark/manifest.json') == old['manifest_sha256']
    protocol = copy.deepcopy(old)
    protocol.update(checkpoint=str(CHECKPOINT), checkpoint_sha256=source_hashes[str(CHECKPOINT)],
                    checkpoint_metadata=metadata, source_hashes=source_hashes, code_sha256=code_hashes(),
                    matched_evaluation=str(MATCHED), matched_manifest_byte_identical=True,
                    input_checksums_reverified=len(expected_inputs), server='io95',
                    gpu_indices=list(range(NUM_SHARDS)), gpu_uuids=GPUS, num_shards=NUM_SHARDS)
    write_json(OUT/'provenance.json', protocol, immutable=True)
    write_json(OUT/'benchmark/_SUCCESS', {'passed':True, 'time':now(), 'sets':len(rows),
        'matched_manifest_sha256':old['manifest_sha256'], 'inputs_verified':len(expected_inputs)})
    progress('complete', sets=len(rows), targets_per_task=3*len(rows))


if __name__ == '__main__':
    main()
