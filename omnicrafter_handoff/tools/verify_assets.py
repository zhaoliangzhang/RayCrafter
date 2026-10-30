"""Verify relocated evaluation RGB against the original content hashes."""
import argparse
from handoff_common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hashes', type=Path, default=ROOT/'metadata/eval_input_hashes.json')
    p.add_argument('--mapping', type=Path, required=True)
    args = p.parse_args()
    mapping = read_json(args.mapping)
    hashes = read_json(args.hashes)
    for i, (original, expected) in enumerate(hashes.items(), 1):
        destination = rebase(original, mapping)
        if sha256(destination) != expected:
            raise ValueError(f'Content mismatch: {destination}')
        if i % 1000 == 0:
            print(f'Verified {i}/{len(hashes)}', flush=True)
    print(f'All {len(hashes)} original input-image checksums match.')


if __name__ == '__main__':
    main()
