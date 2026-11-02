"""Stream an xliu-recipe run's ``reports/train_loss.jsonl`` to W&B.

xliu253's trainer has no tracker (their launcher sets ``WANDB_MODE=disabled``), so this follows the
loss log from a separate process and logs each step, backfilling from step 1. The run id is kept in
``logs/wandb_id``, so a restart of this script resumes the same W&B run. Each line's ``loss`` is the
mean over the 4 ranks; ``rank0_*`` describe only rank 0's set, so they are not used to split the loss.

    python wandb_tail.py --run <output dir>    # any node with network; detached
"""
import argparse
import json
import secrets
import time
from datetime import datetime
from pathlib import Path

import wandb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--entity', default='jhu-aiem')
    p.add_argument('--project', default='raycrafter')
    p.add_argument('--stale-hours', type=float, default=3, help='finish if the loss log stops growing this long')
    args = p.parse_args()
    config = json.loads((args.run/'runtime/train_config.json').read_text())
    request = json.loads((args.run/'portable_request.json').read_text())
    id_file = args.run/'logs/wandb_id'
    run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
    id_file.write_text(run_id+'\n')
    run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume='allow', name=args.run.name,
                     config={**config, 'trainer': 'omnicrafter_handoff/snapshots/mixed/train.py (xliu253, unmodified)',
                             'launcher': request.get('launcher'),
                             'prepared_initialization_sha256': request.get('prepared_initialization_sha256')})
    run.define_metric('train/*', step_metric='step')
    run.define_metric('perf/*', step_metric='step')
    run.define_metric('data/*', step_metric='step')
    log, progress = args.run/'reports/train_loss.jsonl', args.run/'runtime/progress.json'
    last_step = int(run.summary.get('tail/last_step', 0))
    last_time, last_growth = None, time.monotonic()
    offset, pending, progress_step = 0, b'', None
    while True:
        with log.open('rb') as f:
            f.seek(offset)
            chunk = f.read()
        offset += len(chunk)
        lines = (pending+chunk).split(b'\n')
        pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            stamp = datetime.fromisoformat(record['time'])
            wall = (stamp-last_time).total_seconds() if last_time else None
            last_time = stamp
            if record['step'] <= last_step:
                continue
            metrics = {'step': record['step'], 'train/loss': record['loss'], 'train/epoch': record['epoch'],
                       'perf/step_seconds': record['step_seconds']}
            if wall is not None:
                metrics['perf/wall_seconds_per_step'] = wall
            run.log(metrics)
            last_step = record['step']
        if chunk:
            last_growth = time.monotonic()
            run.summary['tail/last_step'] = last_step
        try:
            state = json.loads(progress.read_text())
        except (OSError, ValueError):
            state = None
        if state and state.get('step') != progress_step and state.get('step', 0) <= last_step:
            progress_step = state['step']
            run.log({'step': progress_step, **{f'data/{k}': state[k] for k in
                     ('nvs_sample_fraction', 'perspective_fraction', 'hm3d_fraction', 'jhu_fraction', 'black_query_fraction')
                     if k in state}})
        if (args.run/'_SUCCESS').exists() and not chunk:
            break
        if time.monotonic()-last_growth > args.stale_hours*3600:
            print(f'loss log stopped growing at step {last_step}; finishing', flush=True)
            break
        time.sleep(30)
    run.finish()


if __name__ == '__main__':
    main()
