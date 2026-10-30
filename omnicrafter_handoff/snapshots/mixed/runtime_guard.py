"""Opt-in, in-process memory reservation and bounded local stall diagnostics.

The reservation is not an exclusive GPU lock. It never starts CUDA subprocesses,
uses no random numbers or recurrent GPU kernels, and leaves explicit headroom.
"""
import atexit
import faulthandler
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import time

GIB = 1024 ** 3
_TRACE = None


def event(stage, **fields):
    if _TRACE is not None:
        _TRACE.mark(stage, **fields)


class LocalTrace:
    def __init__(self, root, rank):
        global _TRACE
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.context = {}
        self.logger = logging.getLogger(f'mixed_runtime_rank{rank}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = RotatingFileHandler(self.root / f'phases_rank{rank}.jsonl',
                                           maxBytes=8 * 1024 ** 2, backupCount=2)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
        self.stack = (self.root / f'stacks_rank{rank}.log').open('a')
        faulthandler.enable(file=self.stack)
        _TRACE = self
        atexit.register(self.close)

    def mark(self, stage, **fields):
        self.context.update(fields)
        record = {'unix_time': time.time(), 'pid': os.getpid(), 'rank': self.rank,
                  'stage': stage, **self.context}
        self.logger.info(json.dumps(record, allow_nan=False))

    def watchdog(self, enabled=True):
        faulthandler.cancel_dump_traceback_later()
        if enabled:
            faulthandler.dump_traceback_later(180, repeat=True, file=self.stack)

    def close(self):
        global _TRACE
        if _TRACE is not self:
            return
        self.watchdog(False)
        _TRACE = None
        self.handler.close()
        self.logger.removeHandler(self.handler)
        faulthandler.disable()
        self.stack.close()


def planned_reservation(free_bytes, headroom_gib=16, chunk_gib=1):
    if headroom_gib < 16 or chunk_gib <= 0:
        raise ValueError('At least 16 GiB headroom and positive chunks are required')
    chunk = int(chunk_gib * GIB)
    return max(0, (int(free_bytes) - int(headroom_gib * GIB)) // chunk) * chunk


def exclusive_process_on_gpu(uuid):
    """Fail closed if a foreign CUDA process appeared after the idle check."""
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    pids = [int(line.split(',')[1].strip()) for line in raw.splitlines()
            if line.split(',')[0].strip() == uuid]
    return bool(pids) and set(pids) == {os.getpid()}


class IdleMemoryReservation:
    def __init__(self, torch_module, device, headroom_gib=16):
        self.torch = torch_module
        self.device = device
        self.headroom = int(headroom_gib * GIB)
        self.chunks = []
        self.bytes = 0
        self.attempted = False
        planned_reservation(0, headroom_gib)

    def acquire(self, uuid, ownership_check=exclusive_process_on_gpu):
        if self.attempted:
            return {'status': 'already_attempted', 'reserved_bytes': self.bytes}
        self.attempted = True
        try:
            if not ownership_check(uuid):
                return {'status': 'skipped_shared_gpu', 'reserved_bytes': 0}
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            return {'status': 'skipped_ownership_unknown', 'error': str(exc), 'reserved_bytes': 0}
        free, total = self.torch.cuda.mem_get_info(self.device)
        budget = planned_reservation(free, self.headroom / GIB)
        status = 'allocated'
        # One-time allocation only, after both projection sizes have warmed up.
        # torch.empty does not fill the buffer or consume the training RNG.
        while self.bytes < budget:
            free, _ = self.torch.cuda.mem_get_info(self.device)
            if free < self.headroom + GIB:
                break
            try:
                self.chunks.append(self.torch.empty(GIB, dtype=self.torch.uint8,
                                                    device=self.device, requires_grad=False))
            except self.torch.cuda.OutOfMemoryError:
                status = 'released_after_reservation_oom'
                self.release()
                break
            self.bytes += GIB
        free, _ = self.torch.cuda.mem_get_info(self.device)
        return {'status': status, 'reserved_bytes': self.bytes,
                'free_bytes': free, 'total_bytes': total, 'headroom_bytes': self.headroom}

    def release_under_pressure(self):
        free, _ = self.torch.cuda.mem_get_info(self.device)
        if self.chunks and free < self.headroom // 2:
            released = self.bytes
            self.release()
            return {'status': 'released_low_headroom', 'released_bytes': released,
                    'free_before_bytes': free}
        return None

    def release(self):
        self.chunks.clear()
        self.bytes = 0
        # Release only this process's unused allocator cache; never other jobs.
        self.torch.cuda.empty_cache()
