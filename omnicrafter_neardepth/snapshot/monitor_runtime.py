"""CPU-only GPU-process history. Monitoring never stops another user's job."""
import argparse
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import threading
import time


def query(fields, kind='gpu'):
    output = subprocess.check_output(
        ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    return [[value.strip() for value in row] for row in csv.reader(output.splitlines())]


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        user = pwd.getpwuid(path.stat().st_uid).pw_name
        comm = (path / 'comm').read_text().strip()
        return {'user': user, 'comm': comm}
    except (OSError, KeyError):
        return {'user': None, 'comm': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-root', required=True)
    parser.add_argument('--shared-root', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    local, shared = Path(args.local_root), Path(args.shared_root)
    local.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    logger = logging.getLogger('gpu_history')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(local / 'gpu_process_history.jsonl',
                                   maxBytes=32 * 1024 ** 2, backupCount=8)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    pending = []
    pending_lock = threading.Lock()

    def publish():
        # Network I/O is isolated from training and local monitoring.
        while not stop.wait(60):
            try:
                with pending_lock:
                    batch = pending[:]
                    pending.clear()
                shared.mkdir(parents=True, exist_ok=True)
                with (shared / 'gpu_process_history.jsonl').open('a') as handle:
                    handle.writelines(batch)
                shutil.copy2(local / 'gpu_latest.json', shared / 'gpu_latest.json')
            except OSError:
                pass  # The complete bounded history remains on the local disk.

    threading.Thread(target=publish, daemon=True).start()
    try:
        while not stop.is_set():
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                break
            record = {'unix_time': time.time(), 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            try:
                record['gpus'] = query('index,uuid,memory.used,memory.total,utilization.gpu')
                record['processes'] = [
                    {'gpu_uuid': row[0], 'pid': int(row[1]), 'memory_mib': row[2],
                     **process_identity(int(row[1]))}
                    for row in query('gpu_uuid,pid,used_gpu_memory', 'compute-apps')]
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                record['error'] = str(exc)
            line = json.dumps(record) + '\n'
            logger.info(line.rstrip())
            temporary = local / '.gpu_latest.tmp'
            temporary.write_text(line)
            temporary.replace(local / 'gpu_latest.json')
            with pending_lock:
                pending.append(line)
                if len(pending) > 3600:
                    del pending[:-3600]
            stop.wait(10)
    finally:
        stop.set()
        handler.close()


if __name__ == '__main__':
    main()
