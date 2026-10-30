"""CPU-only metadata audit; does not open image files or allocate GPUs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from handoff_common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, default=ROOT/'metadata/train_cached.jsonl.gz')
    p.add_argument('--evaluation', type=Path, default=ROOT/'metadata/eval_manifest.json')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    counts, scenes, ids = Counter(), defaultdict(set), set()
    heldout = set(read_json(ROOT/'metadata/jhu_heldout_scenes.json'))
    train_digest = hashlib.sha256()
    opener = gzip.open if str(args.training).endswith('.gz') else open
    with opener(args.training, 'rb') as f:
        for line in f:
            train_digest.update(line)
            row = json.loads(line)
            validate_row(row, train=True)
            if row['id'] in ids:
                raise ValueError('Duplicate logical training ID (repeats need explicit unique IDs)')
            ids.add(row['id'])
            counts[row['dataset']] += 1
            scenes[row['dataset']].add(row['scene_name'])
            if row['dataset'] == 'JHU' and row['scene_name'].split('::')[-1] in heldout:
                raise ValueError('Held-out JHU sequence in training')
    if counts != COUNTS:
        raise ValueError(counts)
    evaluation = read_json(args.evaluation)
    ec = Counter()
    overlap = Counter()
    for i, row in enumerate(evaluation['sets']):
        validate_row(row)
        if row['index'] != i or row['seed'] != 20260904+i:
            raise ValueError('Evaluation seed/order differs')
        if row['scene_name'] in scenes[row['dataset']]:
            raise ValueError('Exact scene-name overlap between current training and evaluation')
        ec[row['dataset']] += 1
        overlap[row['dataset']] += sum(t['reference_overlap'] for t in row['targets'])
    if ec != EVAL_COUNTS or overlap != {'HM3D': 21, 'Replica': 0, 'JHU': 0, 'DL3DV': 0}:
        raise ValueError((ec, overlap))
    value = {'train_counts': dict(counts), 'train_scene_sequence_counts': {k:len(v) for k,v in scenes.items()},
             'training_uncompressed_sha256': train_digest.hexdigest(), 'eval_counts': dict(ec),
             'eval_targets_per_task': len(evaluation['sets'])*3, 'reference_overlap_targets': dict(overlap),
             'jhu_heldout_excluded': True, 'exact_scene_name_overlap': False,
             'scope': 'Metadata only; not a proof of physical-location disjointness or valid image copies'}
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
