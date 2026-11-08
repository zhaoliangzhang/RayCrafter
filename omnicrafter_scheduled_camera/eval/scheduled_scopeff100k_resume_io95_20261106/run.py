"""Persistent eight-GPU resume; preserve io95's existing scoring processes."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import signal
import socket
import subprocess
import time
from config_resume import *

s = configure()
CHILDREN = []


def state(stage, **extra):
    counts = {d: {t: len(list((OUT / 'bundles' / d / t).glob('*/*/_SUCCESS')))
                  for t in s.TASKS} for d in s.COUNTS}
    record = dict(time=s.now(), stage=stage, server='io95', model=s.MODEL, completed_sets=counts,
                  expected_sets_per_task=s.COUNTS, resume_contract=str(RUNTIME / 'launch_contract.json'), **extra)
    s.write_json(OUT / 'runtime/status.json', record)
    s.write_json(RUNTIME / 'status.json', record)
    print(json.dumps(record), flush=True)


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    rows = [[v.strip() for v in line.split(',')] for line in raw.splitlines()]
    assert [r[1] for r in rows] == GPUS
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True, timeout=20)
    occupied = [line for line in apps.splitlines() if any(g in line for g in GPUS)]
    return dict(idle=not occupied and all(int(r[2]) < 1024 and int(r[3]) <= 5 for r in rows),
                gpus=rows, compute_apps=occupied)


def wait_idle(stage):
    count = 0
    while count < 3:
        current = inventory()
        count = count + 1 if current['idle'] else 0
        state('confirming_idle_gpus' if count else 'waiting_for_idle_gpus',
              next_stage=stage, consecutive_idle_checks=count, **current)
        if count < 3:
            time.sleep(15 if count else 60)


def spawn(label, script, args=(), gpu=None, metric=False, adapter=False):
    env = {**os.environ, 'CAMERA_VARIANT': 'scope_ff', 'OC_EVAL_MODEL': s.MODEL,
           'CUDA_VISIBLE_DEVICES': '' if gpu is None else GPUS[gpu], 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
           'TRITON_CACHE_DIR': str(CACHE_ROOT / f'triton_{gpu}')}
    if script == 'test_eval.py':
        env['TORCH_COMPILE_DISABLE'] = '1'
    command = [str(s.METRIC_PYTHON if metric else s.PYTHON), '-B', '-u']
    command += ([str(HERE / script)] if adapter else [str(HERE / 'entry.py'), script]) + list(args)
    log = OUT / 'logs' / f'io95_resume_{label}_{s.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(child)
    s.write_json(RUNTIME / f'command_{label}.json', dict(time=s.now(), command=command,
                 pid=child.pid, log=str(log), gpu_uuid=None if gpu is None else GPUS[gpu]))
    return label, child, log


def wait(jobs, stage):
    while True:
        errors = [(n, p.returncode, str(log)) for n, p, log in jobs if p.poll() not in (None, 0)]
        if errors:
            raise RuntimeError(f'{stage}: {errors}')
        running = [n for n, p, _ in jobs if p.poll() is None]
        state(stage, running=running)
        if not running:
            return
        time.sleep(30)


def main():
    assert socket.gethostname().split('.')[0] == 'io95'
    RUNTIME.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_status = OUT / 'runtime/status.json'
        archived_status = RUNTIME / 'original_io107_status.json'
        if original_status.exists() and not archived_status.exists():
            s.write_json(archived_status, json.loads(original_status.read_text()), immutable=True)
        launch = dict(host='io95', adapter_hashes=adapter_hashes(), original_engine=str(ORIGINAL),
                      original_code_sha256=s.code_hashes(), gpu_uuids=GPUS,
                      runtime_overrides=dict(weight_cache_gib=VRAM_LIMIT_GIB, logical_shards=NUM_SHARDS,
                                             wan=str(WAN), cache_root=str(CACHE_ROOT), dreamsim_cache=str(DREAM_CACHE)),
                      preserve=dict(samples=True, seeds=True, camera_math=True, resolutions=True,
                                    checkpoint=True, completed_outputs=True, original_provenance=True),
                      queue=dict(require_all_eight_idle=True, interval_seconds=60, expiry=None))
        s.write_json(RUNTIME / 'launch_contract.json', launch, immutable=True)
        if (OUT / '_SUCCESS').exists():
            state('complete', summary=str(OUT / 'metrics/summary.json'))
            return
        wait([spawn('preflight', 'stage.py', adapter=True)], 'verifying_preserved_results_and_local_weights')
        wait([spawn('cpu_tests', 'test_eval.py')], 'cpu_checks')
        wait_idle('resumed_inference')
        wait([spawn(f'inference_rank{rank}', 'worker.py', ['--rank', str(rank)], gpu=rank)
              for rank in range(NUM_SHARDS)], 'resumed_inference')
        wait_idle('metrics')
        wait([spawn(f'metrics_rank{rank}', 'score.py', ['--rank', str(rank)], gpu=rank, metric=True)
              for rank in range(NUM_SHARDS)], 'metrics')
        wait([spawn('aggregate', 'aggregate.py', metric=True)], 'aggregating')
        assert (OUT / '_SUCCESS').is_file()
        preserved = json.loads((RUNTIME / 'preserved_outputs.json').read_text())['file_sha256']
        def verify(item):
            path, digest = item
            assert s.sha256(path) == digest, f'Preserved output changed: {path}'
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(verify, preserved.items()):
                pass
        s.write_json(RUNTIME / 'preservation_audit.json', dict(passed=True, time=s.now(), files=len(preserved)))
        state('complete', summary=str(OUT / 'metrics/summary.json'))


def stop(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BlockingIOError:
        raise SystemExit('Another controller owns this evaluation')
    except BaseException as exc:
        for child in CHILDREN:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        state('failed', error=repr(exc))
        raise
