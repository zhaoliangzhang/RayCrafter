"""CPU-only integrity checks and local cache preparation; no GPU allocations."""
from concurrent.futures import ThreadPoolExecutor
import shutil
import socket
from config_resume import *


def copy_verified(source, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
        shutil.copy2(source, temporary)
        assert original.sha256(temporary) == digest
        temporary.replace(destination)
    assert original.sha256(destination) == digest, destination


def main():
    assert socket.gethostname().split('.')[0] == 'io95'
    s = configure()
    assert adapter_hashes() == json.loads((RUNTIME / 'launch_contract.json').read_text())['adapter_hashes']
    contract = json.loads((OUT / 'provenance.json').read_text())
    assert s.code_hashes() == contract['code_sha256']
    assert s.sha256(OUT / 'benchmark/manifest.json') == contract['manifest_sha256']
    assert s.sha256(OUT / 'benchmark/near_depth_resolved.json') == contract['near_depth_sha256']
    for path in (OUT / 'metrics').glob('shard_*.json'):
        part = json.loads(path.read_text())
        assert part['num_shards'] == NUM_SHARDS and 0 <= part['rank'] < NUM_SHARDS

    rows = s.manifest()['sets']
    assert len(rows) == 3791 and [r['index'] for r in rows] == list(range(3791))
    counts = {d: {t: 0 for t in s.TASKS} for d in s.COUNTS}
    preserved = {}
    for row in rows:
        for task in s.TASKS:
            marker = s.completion(row, task)
            if not marker.exists():
                continue
            assert s.completed(row, task)
            record = json.loads(marker.read_text())
            assert record['query_conditions'] == (['all_black'] * 3 if task == 'nvs'
                                                  else [t['noisy'] for t in row['targets']])
            historical = json.loads((s.bundle_at(s.MATCHED, row, task) / '_SUCCESS').read_text())
            assert record['noise_sha256'] == historical['noise_sha256']
            preserved[str(marker)] = s.sha256(marker)
            for j, digest in enumerate(record['prediction_sha256']):
                preserved[str(s.prediction(row, task, j))] = digest
            counts[row['dataset']][task] += 1

    def verify(item):
        path, digest = item
        assert s.sha256(path) == digest, f'File changed: {path}'

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify, preserved.items()):
            pass
    preservation = RUNTIME / 'preserved_outputs.json'
    if not preservation.exists():
        s.write_json(preservation, dict(time=s.now(), counts=counts, file_sha256=preserved), immutable=True)
    else:
        # On relaunch, retain and verify the original migration boundary.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(verify, json.loads(preservation.read_text())['file_sha256'].items()):
                pass
    print(json.dumps(dict(time=s.now(), preserved_counts=counts)), flush=True)

    assert s.sha256(s.CHECKPOINT) == contract['checkpoint_sha256']
    copy_verified(s.CHECKPOINT, CACHE_ROOT / s.CHECKPOINT.name, contract['checkpoint_sha256'])
    canonical = json.loads((s.MATCHED / 'runtime/local_cache.json').read_text())
    copy_verified(FID_SOURCE, CACHE_ROOT / 'fid/inception-2015-12-05.pt', canonical['fid_sha256'])
    expected = json.loads((ORIGINAL / 'expected_wan_sha256.json').read_text())
    folder = WAN / 'Wan-AI/Wan2.1-I2V-14B-480P'
    sizes = {Path(p).name: size for p, size in canonical['wan_files'].items()}
    assert len(expected) == len(sizes) == 10
    for name, digest in expected.items():
        path = folder / name
        assert path.stat().st_size == sizes[name] and s.sha256(path) == digest, path
        print(json.dumps(dict(time=s.now(), verified_wan=name)), flush=True)
    assert (WAN / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json').is_file()
    assert DREAM_CACHE.is_dir()
    inputs = json.loads((OUT / 'input_hashes.json').read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify, inputs.items()):
            pass
    prior_hashes = json.loads((OUT / 'benchmark/prior_metadata_hashes.json').read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify, prior_hashes.items()):
            pass
    remaining = {str(rank): {task: sum(not s.completed(r, task) for r in rows
                                      if r['index'] % NUM_SHARDS == rank) for task in s.TASKS}
                 for rank in range(NUM_SHARDS)}
    s.write_json(RUNTIME / 'preflight.json', dict(passed=True, time=s.now(), preserved_counts=counts,
                 checkpoint_sha256=contract['checkpoint_sha256'], wan_sha256=expected,
                 input_files_verified=len(inputs), remaining_by_rank=remaining,
                 gpu_uuids=GPUS, logical_shards=NUM_SHARDS, weight_cache_gib=VRAM_LIMIT_GIB))
    print(json.dumps(dict(time=s.now(), stage='preflight_passed', remaining_by_rank=remaining)), flush=True)


if __name__ == '__main__':
    main()
