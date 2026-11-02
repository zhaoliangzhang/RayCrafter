"""Warm the GPU node's page cache with the images xliu253's mixed trainer will read next.

The trainer loads each set inline (``num_workers=0``), so one slow NFS read on any rank stalls all four.
Its order is fixed by the seeded sampler, so this rebuilds each rank's epoch-0 order, follows the run's
step in ``reports/train_loss.jsonl`` and reads the next ``--ahead`` steps' files (references, targets and
noisy renders) with a thread pool. It only reads files: the training process and its math are untouched,
and over the epoch it reads what the trainer reads, plus the noisy renders of sets that turn out NVS.

    python -I prefetch.py --run <output dir>    # on the GPU node, detached
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SEED = 20261023  # snapshots/mixed/train.py
WORLD_SIZE = 4


def rank_orders(n):
    """Dataset indices per rank and step, as ``accelerator.prepare`` shards the trainer's loader."""
    from accelerate import Accelerator
    from accelerate.data_loader import prepare_data_loader

    class Index(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return i

    Accelerator(cpu=True)
    orders = []
    for rank in range(WORLD_SIZE):
        loader = DataLoader(Index(), batch_size=1, shuffle=True, drop_last=True, collate_fn=lambda rows: rows,
                            num_workers=0, generator=torch.Generator().manual_seed(SEED))
        shard = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=WORLD_SIZE, process_index=rank,
                                    split_batches=False, put_on_device=False, rng_types=['generator'],
                                    dispatch_batches=False, even_batches=True, use_seedable_sampler=False)
        shard.set_epoch(0)
        orders.append([b[0] for b in shard])
    return np.array(orders, dtype=np.int64)


def current_step(log):
    """Last step in the trainer's per-step loss log, or 0 before the first line."""
    try:
        with log.open('rb') as f:
            f.seek(max(0, f.seek(0, os.SEEK_END)-4096))
            lines = f.read().splitlines()
    except FileNotFoundError:
        return 0
    for line in reversed(lines):
        try:
            return json.loads(line)['step']
        except (ValueError, KeyError):
            continue
    return 0


def read_through(path):
    start, size = time.monotonic(), 0
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            size += len(chunk)
    return time.monotonic()-start, size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--ahead', type=int, default=300, help='steps to stay ahead of the trainer')
    p.add_argument('--chunk', type=int, default=10, help='steps per batch of reads')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--stale-hours', type=float, default=3, help='exit if the loss log stops moving this long')
    args = p.parse_args()
    config = json.loads((args.run/'runtime/train_config.json').read_text())
    data = Path(config['data_dir'])
    final = config['final_step']
    assert config['initial_step'] == 0, 'orders are rebuilt for a run that starts at step 0'
    offsets = np.load(data/'train.offsets.npy', mmap_mode='r')
    orders = rank_orders(len(offsets))
    assert orders.shape[1] >= final
    manifest = (data/'train.jsonl').open('rb')
    log = args.run/'reports/train_loss.jsonl'

    def files(step):
        out = []
        for rank in range(WORLD_SIZE):
            manifest.seek(int(offsets[orders[rank, step-1]]))
            row = json.loads(manifest.readline())
            out += [v['image'] for v in row['references']+row['targets']] + [v['noisy'] for v in row['targets']]
        return out

    print(f'{time.strftime("%F %T")} orders rebuilt for {WORLD_SIZE} ranks x {orders.shape[1]} steps; '
          f'ahead {args.ahead}, {args.threads} threads', flush=True)
    done, last_report, last_move, last_step = 0, 0., time.monotonic(), -1
    reads = slow = 0
    nbytes = 0
    with ThreadPoolExecutor(args.threads) as pool:
        while done < final and not (args.run/'_SUCCESS').exists():
            step = current_step(log)
            if step != last_step:
                last_step, last_move = step, time.monotonic()
            elif time.monotonic()-last_move > args.stale_hours*3600:
                print(f'{time.strftime("%F %T")} loss log stuck at step {step} for {args.stale_hours} h; exiting', flush=True)
                return
            done = max(done, step)
            target = min(final, step+args.ahead, done+args.chunk)
            if target > done:
                paths = [q for s in range(done+1, target+1) for q in files(s)]
                for seconds, size in pool.map(read_through, paths):
                    reads += 1
                    slow += seconds > 5
                    nbytes += size
                done = target
            else:
                time.sleep(5)
            if time.monotonic()-last_report > 60:
                print(f'{time.strftime("%F %T")} trainer {step} prefetched {done} (+{done-step}) '
                      f'reads {reads} slow>5s {slow} {nbytes/2**30:.1f} GiB', flush=True)
                last_report = time.monotonic()
    print(f'{time.strftime("%F %T")} done at prefetched step {done}', flush=True)


if __name__ == '__main__':
    main()
