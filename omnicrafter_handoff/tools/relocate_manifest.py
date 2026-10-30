"""Rewrite only runtime asset paths; preserve groups, cameras, seeds and splits."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from handoff_common import *


def rewrite(row, mapping, train=False):
    validate_row(row, train)
    result = copy.deepcopy(row)
    # Cached training only contains the eight RGB assets per set. prior_metadata
    # remains an ORIGINAL provenance address, not a training-time dependency.
    for obj, key in image_fields(result, metadata=not train):
        obj[key] = rebase(obj[key], mapping)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['train', 'eval'], required=True)
    p.add_argument('--check-files', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; originals are never overwritten')
    if args.kind == 'train':
        for name in ('train.offsets.npy', 'summary.json'):
            if (args.output.parent/name).exists():
                raise FileExistsError(f'Use a fresh manifest directory: {args.output.parent/name}')
        if args.output.name != 'train.jsonl':
            raise ValueError('Training output must be named train.jsonl')
    mapping = read_json(args.mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name('.' + args.output.name + '.preparing')
    if tmp.exists():
        raise FileExistsError(f'Inspect unfinished output first: {tmp}')
    counts = Counter()
    if args.kind == 'train':
        import numpy as np
        offsets = []
        with text_open(args.input) as inp, tmp.open('xb') as out:
            for line in inp:
                row = rewrite(json.loads(line), mapping, train=True)
                counts[row['dataset']] += 1
                if args.check_files:
                    for obj, key in image_fields(row, metadata=False):
                        required_file(obj[key])
                offsets.append(out.tell())
                out.write((json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n').encode())
        if counts != COUNTS:
            raise ValueError(f'Wrong training composition: {counts}, expected {COUNTS}')
        tmp.replace(args.output)
        offset_path = args.output.with_name('train.offsets.npy')
        if offset_path.exists():
            raise FileExistsError(offset_path)
        with offset_path.open('xb') as out:
            np.save(out, np.asarray(offsets, dtype=np.int64))
        summary = {'sha256': sha256(args.output), 'logical_sets': len(offsets),
                   'dataset_counts': dict(counts), 'offsets_sha256': sha256(offset_path)}
    else:
        data = read_json(args.input)
        data['sets'] = [rewrite(row, mapping) for row in data['sets']]
        for row in data['sets']:
            counts[row['dataset']] += 1
            if args.check_files:
                for obj, key in image_fields(row):
                    required_file(obj[key])
        if counts != EVAL_COUNTS:
            raise ValueError(f'Wrong full evaluation composition: {counts}')
        write_json(tmp, data)
        tmp.replace(args.output)
        summary = {'sha256': sha256(args.output), 'logical_sets': len(data['sets']),
                   'dataset_counts': dict(counts)}
    summary.update(source=str(args.input), source_sha256=sha256(args.input), mapping=mapping,
                   changes='runtime asset paths only; poses, ordering, seeds and split unchanged')
    write_json(args.output.parent / ('summary.json' if args.kind == 'train' else 'relocation.json'), summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
